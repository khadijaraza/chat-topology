import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// No <StrictMode> here on purpose: with React 19.2.8 + @react-three/fiber
// 9.7.0, StrictMode's double-invoked mount/cleanup/mount leaves the R3F
// <canvas> element stuck at the browser's default un-sized state (static
// position, 300x150) instead of the resize-observed 100%/100% it should
// get -- confirmed by comparing computed canvas size with StrictMode
// present vs. removed. Revisit if a future @react-three/fiber release
// documents this as fixed.
createRoot(document.getElementById('root')!).render(<App />)
