import React, { useState } from 'react'

export default function IntroPage({ initialConfig, onStart }) {
  const [place, setPlace] = useState(initialConfig.placeQuery || '')
  const [n, setN] = useState(initialConfig.nPoints || 1)

  return (
    <div className="intro-root">
      <div className="intro-box">
        <h1>Real-World Drone Delivery</h1>
        <p>Enter the neighbourhood or place to load and the number of delivery points.</p>

        <label>Place / neighbourhood</label>
        <input value={place} onChange={(e)=>setPlace(e.target.value)} />

        <label>Number of delivery points</label>
        <input type="number" min={1} max={12} value={n} onChange={(e)=>setN(Math.max(1, Math.min(12, +e.target.value)))} />

        <div style={{marginTop:16}}>
          <button onClick={()=>onStart({ placeQuery: place, nPoints: n })}>Start planning</button>
        </div>

        <div style={{marginTop:12, fontSize:13}}>
          <strong>Note</strong>
          <p>Heights used by planner are strictly the OSM 'height' tag. If heights are missing in the area, the planner will refuse to plan (safe mode).</p>
        </div>
      </div>
    </div>
  )
}
