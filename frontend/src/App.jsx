import { useState } from 'react'
import MapPage from './MapPage.jsx'
import './App.css'

export default function App() {
  const [mode, setMode] = useState(null)
  if (mode) return <MapPage sampleOnOpen={mode === 'sample'} onExit={() => setMode(null)} />
  return (
    <main className="launch-screen">
      <section className="launch-panel">
        <h1>Drone delivery simulator</h1>
        <p>Watch two orders leave the post office, reach their delivery points, and return to base. Or choose your own stops on the map.</p>
        <div className="launch-actions">
          <button onClick={() => setMode('sample')}>Try a sample flight</button>
          <button onClick={() => setMode('manual')}>Plan my own route</button>
        </div>
        <p className="launch-location">Demo coverage is limited to Hanamkonda, Telangana.</p>
      </section>
    </main>
  )
}
