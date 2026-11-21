import { useEffect, useState } from 'react'
import { Marker } from 'react-leaflet'
import L from 'leaflet'

const droneIcon = L.icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/512/149/149059.png',
  iconSize: [32, 32],
  iconAnchor: [16, 16],
})

export default function DronePlayer({ path }) {
  const [idx, setIdx] = useState(0)

  useEffect(() => {
    if (!path || path.length === 0) return
    setIdx(0)
    const t = setInterval(() => {
      setIdx(i => (i + 1 < path.length ? i + 1 : i))
    }, 400)
    return () => clearInterval(t)
  }, [path])

  if (!path || path.length === 0) return null

  return <Marker position={[path[idx][0], path[idx][1]]} icon={droneIcon} />
}
