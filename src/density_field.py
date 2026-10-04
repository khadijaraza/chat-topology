"""Continuous density field over the proximity layout via a Dirichlet
Process Gaussian Mixture Model (DPGMM).

An alternative reading of the same segment positions the Macro Domains tier
(src/macro_domains.py) already clusters discretely: instead of assigning
every segment to exactly one hard-boundaried domain, fit an infinite
mixture of 2D Gaussians over the UMAP (x, y) positions and evaluate its
total density on a regular grid. Segments that genuinely blend topics, or
sit between two real themes, read as a smooth gradient rather than being
forced across an arbitrary line -- recurring themes become density
"mountains," one-off tangents trail into "lowland plains." This is an
ADDITIONAL visualization layer, not a replacement for Macro Domains; the
frontend renders it underneath the existing point cloud, toggleable.

**Fit in 2D layout space, not in the original 768-dim embedding space** --
a deliberate, stated scope decision. The "infinite mixture over segment
embeddings" framing would produce components that don't correspond to any
single renderable surface (projecting a 768-dim Gaussian's covariance
structure into the 2D UMAP view for contouring is its own, much messier
problem, and UMAP already deliberately discards exact distances in favor of
readable 2D structure). Fitting directly on the (x, y) positions already
used for the point cloud keeps "what you see in 3D" and "what the model
fit" the same space, at the cost of the mixture only seeing proximity, not
the original semantic distances UMAP compressed away.

**"Density" rendered by the frontend is log-density, not raw density** --
also a deliberate, stated choice. `model.score_samples()` returns
log-probability; exponentiating it for raw density produces an extremely
peaked distribution (a few tight Gaussians can dwarf everything else by
orders of magnitude), which reads as one spike and a flat plain rather than
the rolling "mountain range" the visualization is meant to evoke. Log-
density compresses that dynamic range into something a heightfield mesh
can actually show texture in.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DENSITY_FIELD_CONFIG_PATH = PROJECT_ROOT / "config" / "density_field.yaml"

# Fallbacks used only if config/density_field.yaml is missing or malformed.
FALLBACK_MAX_COMPONENTS = 60
FALLBACK_COVARIANCE_TYPE = "full"
FALLBACK_WEIGHT_CONCENTRATION_PRIOR = 5.0
FALLBACK_COVARIANCE_PRIOR_SCALE = 0.3
FALLBACK_DEGREES_OF_FREEDOM_PRIOR = 10.0
FALLBACK_GRID_RESOLUTION = 150
FALLBACK_RANDOM_STATE = 42
FALLBACK_TERRAIN_Z_RANGE = 40.0

# A truncated stick-breaking component with weight below this is treated as
# "pruned" by the Dirichlet process prior -- this is what "the algorithm
# determines the number of components automatically" looks like in a
# variational (not literally infinite) implementation: fit max_components,
# then count how many the prior actually kept.
EFFECTIVE_WEIGHT_THRESHOLD = 1e-3


@lru_cache(maxsize=1)
def get_density_field_config(path: str = str(DEFAULT_DENSITY_FIELD_CONFIG_PATH)) -> dict:
    """Load DPGMM/grid settings from config/density_field.yaml."""
    config_path = Path(path)
    if not config_path.is_file():
        return {
            "max_components": FALLBACK_MAX_COMPONENTS,
            "covariance_type": FALLBACK_COVARIANCE_TYPE,
            "weight_concentration_prior": FALLBACK_WEIGHT_CONCENTRATION_PRIOR,
            "covariance_prior_scale": FALLBACK_COVARIANCE_PRIOR_SCALE,
            "degrees_of_freedom_prior": FALLBACK_DEGREES_OF_FREEDOM_PRIOR,
            "grid_resolution": FALLBACK_GRID_RESOLUTION,
            "random_state": FALLBACK_RANDOM_STATE,
            "terrain_z_range": FALLBACK_TERRAIN_Z_RANGE,
        }

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return {
        "max_components": int(payload.get("max_components", FALLBACK_MAX_COMPONENTS)),
        "covariance_type": payload.get("covariance_type", FALLBACK_COVARIANCE_TYPE),
        "weight_concentration_prior": float(
            payload.get("weight_concentration_prior", FALLBACK_WEIGHT_CONCENTRATION_PRIOR)
        ),
        "covariance_prior_scale": float(
            payload.get("covariance_prior_scale", FALLBACK_COVARIANCE_PRIOR_SCALE)
        ),
        "degrees_of_freedom_prior": float(
            payload.get("degrees_of_freedom_prior", FALLBACK_DEGREES_OF_FREEDOM_PRIOR)
        ),
        "grid_resolution": int(payload.get("grid_resolution", FALLBACK_GRID_RESOLUTION)),
        "random_state": int(payload.get("random_state", FALLBACK_RANDOM_STATE)),
        "terrain_z_range": float(payload.get("terrain_z_range", FALLBACK_TERRAIN_Z_RANGE)),
    }


def fit_dpgmm(positions: np.ndarray):
    """Fit a Dirichlet Process Gaussian Mixture Model (variational, via
    sklearn's BayesianGaussianMixture) over 2D positions.

    P(segment_i in component k) = pi_k * N(position_i | mu_k, Sigma_k) --
    each component's weight pi_k comes from a stick-breaking prior, which
    prefers explaining the data with as few components as
    weight_concentration_prior allows rather than spreading weight evenly
    across all max_components candidates.

    **Real finding from tuning against this corpus**: weight_concentration_
    prior alone could NOT be pushed high enough to produce more than ~7
    effective components, across three different initialization strategies
    (k-means++, random, random_from_data) and a 67x range of prior values
    (0.3 to 20) -- not a local-optimum artifact, a genuine model-selection
    preference. With unconstrained (full) covariance, a handful of large
    elliptical components can already cover the data's broad "continents"
    (the same macro-scale shape UMAP itself showed -- see
    proximity_layout.py's documented finding of a few broad regions) at
    higher likelihood than many small ones, and the data term dominates the
    ELBO once the prior is "generous enough." What actually controls
    effective component COUNT here is constraining component SIZE via
    covariance_prior/degrees_of_freedom_prior -- shrinking the prior
    covariance from the data's own scale (the sklearn default) down to a
    small fraction of coordinate_range forces the model to cover the same
    area with many smaller components instead of few large ones. Confirmed
    by direct sweep: covariance_prior scale 1.0 -> 16 effective components,
    0.1 -> 21, 0.02 -> 24; adding degrees_of_freedom_prior=10 (stronger
    pull toward the (now small) prior covariance) at scale 0.003 -> 32,
    landing close to Macro Domains' ~38 for a comparable "how many real
    territories" reading between the discrete and continuous layers.
    """
    from sklearn.mixture import BayesianGaussianMixture

    cfg = get_density_field_config()
    covariance_prior = np.eye(2) * cfg["covariance_prior_scale"]
    model = BayesianGaussianMixture(
        n_components=cfg["max_components"],
        covariance_type=cfg["covariance_type"],
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=cfg["weight_concentration_prior"],
        covariance_prior=covariance_prior,
        degrees_of_freedom_prior=cfg["degrees_of_freedom_prior"],
        max_iter=500,
        n_init=3,
        random_state=cfg["random_state"],
    )
    model.fit(positions)
    return model


def effective_component_count(model, weight_threshold: float = EFFECTIVE_WEIGHT_THRESHOLD) -> int:
    """How many of the max_components candidate components the Dirichlet
    process prior actually kept non-negligible weight on."""
    return int((model.weights_ > weight_threshold).sum())


def evaluate_density_grid(model, coordinate_range: float, grid_resolution: int) -> dict:
    """Evaluate the mixture's LOG-density (see module docstring for why not
    raw density) on a regular grid spanning
    [-coordinate_range, +coordinate_range] on both axes.

    Returns {"x": [...], "y": [...], "log_density": [[...]],
    "normalized_density": [[...]]}, both grid_resolution x grid_resolution,
    row-major in y (grid[j][i] is the value at (x[i], y[j])).

    **`normalized_density` exists because `log_density` alone still isn't
    legible for rendering, even after the log transform.** Real finding:
    the covariance_prior_scale/degrees_of_freedom_prior tuning needed to
    get enough EFFECTIVE components (see fit_dpgmm's docstring) produces
    some genuinely tight, high-precision Gaussians -- a tight component's
    peak height scales with 1/sqrt(det(Sigma)), so a few narrow "spikes"
    end up hundreds of nats taller than the broader ridges between
    components. A raw min-max (or even percentile-clipped) color/height
    scale over log_density gets dominated by those few spikes -- nearly
    the entire populated area saturates to "maximum" and the actual
    texture between real clusters disappears. `normalized_density` is
    log_density's PERCENTILE RANK across the grid (0 = lowest-density grid
    cell, 1 = highest) -- a histogram-equalizing transform that guarantees
    a legible visual spread regardless of how skewed the raw values are.
    This is a rendering-only construct, not a probability -- `log_density`
    is kept alongside it as the literal, undistorted math for anyone who
    wants it.
    """
    xs = np.linspace(-coordinate_range, coordinate_range, grid_resolution)
    ys = np.linspace(-coordinate_range, coordinate_range, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    log_density = model.score_samples(grid_points).reshape(grid_resolution, grid_resolution)

    order = log_density.ravel().argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(log_density.size)
    normalized_density = (ranks / (log_density.size - 1)).reshape(grid_resolution, grid_resolution)

    return {
        "x": xs.tolist(),
        "y": ys.tolist(),
        "log_density": log_density.tolist(),
        "normalized_density": normalized_density.tolist(),
    }


def segment_component_responsibilities(model, positions: np.ndarray) -> np.ndarray:
    """P(segment_i in component k) for every segment -- the soft
    probability distribution over mixture components each segment
    receives, instead of a single hard cluster id. Not currently consumed
    by the frontend (segments still render as discrete points positioned
    by UMAP, independent of this mixture), exposed for completeness /
    future use (e.g. blending a segment's point color across its top
    components instead of using its discrete macro-domain id)."""
    return model.predict_proba(positions)
