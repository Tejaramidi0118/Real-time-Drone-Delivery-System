import { MapContainer, GeoJSON, Tooltip, TileLayer, Marker, Polyline, useMapEvents, useMap } from 'react-leaflet'
import L from 'leaflet'
import { useState, useEffect, useMemo, useRef } from 'react'
import './flight-events.css'
import axios from 'axios'
import 'leaflet/dist/leaflet.css'

const API_BASE = 'http://localhost:5002'

// Realistic drone speed: 15 m/s (54 km/h) - typical delivery drone speed
const DRONE_SPEED_MS = 15.0  // meters per second
const SIMULATION_INTERVAL_MS = 100  // Update every 100ms

const SIMULATED_ENDURANCE_SECONDS = 30 * 60

// Icons
const baseIcon = L.icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
})

function orderIcon(id, delivered) {
  return L.divIcon({
    className: 'order-map-icon',
    html: '<span class="order-pin ' + (delivered ? 'done' : '') + '">' + (delivered ? '&#10003; ' : '') + Number(id) + '</span>',
    iconSize: [36, 24],
    iconAnchor: [18, 12],
  })
}

const droneIcon = L.icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
  iconSize: [30, 46],
  iconAnchor: [15, 46],
  popupAnchor: [1, -34],
})

// Fix for default marker icons
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

// Helper: Calculate distance between two lat/lon points in meters
function haversineDistance(lat1, lon1, lat2, lon2) {
  const R = 6371000 // Earth radius in meters
  const dLat = (lat2 - lat1) * Math.PI / 180
  const dLon = (lon2 - lon1) * Math.PI / 180
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2)
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  return R * c
}

function DeliveryClickHandler({ onAdd, enabled }) {
  useMapEvents({
    click(e) {
      if (enabled) {
        onAdd([e.latlng.lat, e.latlng.lng])
      }
    },
  })
  return null
}

function DroneMarker({ position }) {
  const map = useMap()
  const markerRef = useRef(null)

  useEffect(() => {
    if (position && markerRef.current) {
      markerRef.current.setLatLng(position)
      map.setView(position, map.getZoom())
    }
  }, [position, map])

  if (!position) return null

  return <Marker position={position} icon={droneIcon} ref={markerRef} />
}

export default function MapPage({ onExit, sampleOnOpen }) {
  const [base, setBase] = useState(null)
  const [deliveries, setDeliveries] = useState([])
  const [zones, setZones] = useState(null)
  const [zoneError, setZoneError] = useState(false)
  useEffect(() => {
    const controller = new AbortController()
    axios.get(API_BASE + '/environment/data/nfz.geojson', { signal: controller.signal })
      .then(({ data }) => setZones(data))
      .catch(() => { if (!controller.signal.aborted) setZoneError(true) })
    return () => controller.abort()
  }, [])
  const nextOrder = useRef(1)
  const [events, setEvents] = useState([])
  const completed = useMemo(() => new Set(events.filter(e => e.kind === "delivery").map(e => e.orderId)), [events])
  const icons = useMemo(() => new Map(deliveries.map(d => [d.orderId, orderIcon(d.orderId, completed.has(d.orderId))])), [deliveries, completed])
  const [noticeIndex, setNoticeIndex] = useState(0)
  const notice = events[noticeIndex]
  useEffect(() => {
    if (!notice) return
    const timeout = setTimeout(() => setNoticeIndex(i => i + 1), 2400)
    return () => clearTimeout(timeout)
  }, [notice])
  function notify(text, kind = 'info', seconds = 0, orderId = null) {
    setEvents(list => [...list, { text, kind, seconds, orderId }])
  }
  const [route, setRoute] = useState([])
  const [path, setPath] = useState([])
  const [isPlanning, setIsPlanning] = useState(false)
  const [isSimulating, setIsSimulating] = useState(false)
  const [dronePosition, setDronePosition] = useState(null)
  const [currentPathIndex, setCurrentPathIndex] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const distances = useMemo(() => {
    const cumulative = [0]
    for (let i = 1; i < path.length; i++) {
      cumulative.push(cumulative[i - 1] + haversineDistance(...path[i - 1].slice(0, 2), ...path[i].slice(0, 2)))
    }
    return cumulative
  }, [path])
  const total = distances[distances.length - 1]
  const flown = Math.min(total, elapsed * DRONE_SPEED_MS)
  const battery = Math.max(0, 100 * (1 - elapsed / SIMULATED_ENDURANCE_SECONDS))
  const [addingEnabled, setAddingEnabled] = useState(true)
  
  // Search functionality
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [showSearchResults, setShowSearchResults] = useState(false)

  // Load base location
  useEffect(() => {
    axios.get(`${API_BASE}/base`)
      .then(res => {
        setBase([res.data.lat, res.data.lon])
      })
      .catch(err => {
        console.error('Failed to load base:', err)
        alert('Failed to load base location. Please check backend.')
      })
  }, [])

  // Geocode search
  async function handleSearch(query) {
    if (!query.trim()) {
      setSearchResults([])
      setShowSearchResults(false)
      return
    }

    try {
      const res = await axios.get(`${API_BASE}/geocode`, {
        params: { q: query }
      })
      setSearchResults(res.data)
      setShowSearchResults(true)
    } catch (err) {
      console.error('Search failed:', err)
      setSearchResults([])
    }
  }

  function selectSearchResult(result) {
    resetFlight()
    setBase([result.lat, result.lon])
    setSearchQuery('')
    setSearchResults([])
    setShowSearchResults(false)
  }

  function addDelivery(point) {
    resetFlight()
    const orderId = nextOrder.current++
    setDeliveries(prev => [...prev, { lat: point[0], lon: point[1], orderId }])
  }

  function removeDelivery(index) {
    resetFlight()
    setDeliveries(prev => prev.filter((_, i) => i !== index))
  }

  function clearAll() {
    nextOrder.current = 1
    setDeliveries([])
    resetFlight()
  }

  function resetFlight() {
    setEvents([])
    setNoticeIndex(0)
    setElapsed(0)
    setAddingEnabled(true)
    setRoute([])
    setPath([])
    setDronePosition(null)
    setCurrentPathIndex(0)
    setIsSimulating(false)
  }

  const sampleStarted = useRef(false)
  useEffect(() => {
    if (base && sampleOnOpen && !sampleStarted.current) {
      sampleStarted.current = true
      sampleFlight()
    }
  }, [base, sampleOnOpen])

  function sampleFlight() {
    const stops = [
      { lat: 17.99500, lon: 79.54878, orderId: 1 },
      { lat: 17.99500, lon: 79.54843, orderId: 2 },
    ]
    nextOrder.current = 3
    setDeliveries(stops)
    planRoute(stops, true)
  }

  async function planRoute(stops = deliveries, autoplay = false) {
    if (stops.length === 0) {
      alert('Please add at least one delivery point')
      return
    }

    resetFlight()
    setIsPlanning(true)
    setAddingEnabled(false)

    try {
      // Set delivery points
      const accepted = await axios.post(`${API_BASE}/deliveries`, {
        points: stops
      })

      const pending = accepted.data.deliveries.map((d, i) => ({ ...d, orderId: stops[i].orderId }))

      // Plan route
      const res = await axios.post(`${API_BASE}/route/plan`)
      const ordered = res.data.delivery_order.map(stop => {
        const i = pending.findIndex(d => d.lat === stop.lat && d.lon === stop.lon)
        if (i < 0) throw new Error('Could not match the planned delivery to its order')
        return pending.splice(i, 1)[0]
      })
      setDeliveries(ordered)
      setPath(res.data.path)
      setRoute(res.data.delivery_waypoints.map((index, i) => ({ index, orderId: ordered[i].orderId })))
      if (autoplay) startSimulation(res.data.path)
      else notify("Route ready")
    } catch (err) {
      setAddingEnabled(true)
      console.error('Planning failed:', err)
      notify(`Planning failed: ${err.response?.data?.error || err.message}`, "error")
    } finally {
      setIsPlanning(false)
      if (!autoplay) setAddingEnabled(true)
    }
  }

  function startSimulation(flightPath = path) {
    if (flightPath.length === 0) {
      alert('Please plan a route first')
      return
    }

    setElapsed(0)
    setEvents([{ text: "Flight started", kind: "info", seconds: 0 }])
    setNoticeIndex(0)
    setIsSimulating(true)
    setAddingEnabled(false)
    setCurrentPathIndex(0)
    setDronePosition([flightPath[0][0], flightPath[0][1]])
  }

  function stopSimulation() {
    notify("Flight stopped", "info", elapsed)
    setIsSimulating(false)
    setAddingEnabled(true)
  }

  // Realistic simulation loop - moves based on distance and speed
  useEffect(() => {
    if (!isSimulating || path.length === 0) return

    const started = performance.now()
    let nextStop = 0
    let index = 0
    const interval = setInterval(() => {
      const seconds = Math.min((performance.now() - started) / 1000, total / DRONE_SPEED_MS)
      const meters = seconds * DRONE_SPEED_MS
      while (index < path.length - 1 && distances[index + 1] <= meters) index++
      while (nextStop < route.length && (index >= route[nextStop].index || seconds >= total / DRONE_SPEED_MS)) {
        const stop = route[nextStop++]
        notify(`Order #${stop.orderId} delivered`, 'delivery', distances[stop.index] / DRONE_SPEED_MS, stop.orderId)
      }
      const start = path[index]
      const end = path[index + 1] || start
      const length = (distances[index + 1] || total) - distances[index]
      const fraction = length > 0 ? (meters - distances[index]) / length : 0
      setElapsed(seconds)
      setCurrentPathIndex(index)
      setDronePosition([start[0] + (end[0] - start[0]) * fraction, start[1] + (end[1] - start[1]) * fraction])
      if (seconds >= total / DRONE_SPEED_MS) {
        clearInterval(interval)
        notify("Returned to base", "complete", seconds)
        setIsSimulating(false)
        setAddingEnabled(true)
      }
    }, SIMULATION_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [isSimulating, path, distances, total, route])

  if (!base) {
    return <div style={{ padding: 20 }}>Loading base location...</div>
  }

  return (
    <div className="planner-layout">
      <div className="planner-panel">
        <h2>Delivery planner</h2>
        {deliveries.length === 0 && <button className="sample-action" disabled={isPlanning} onClick={sampleFlight}>Try a sample flight</button>}
        {events.length > 0 && <section className="flight-history" aria-label="Flight history">
          <h3>Flight history</h3>
          <ol>{events.map((event, i) => <li key={i} className={event.kind}>
            <time>{event.seconds.toFixed(1)} s</time> {event.text}
          </li>)}</ol>
        </section>}
        <p className="zone-key">{zoneError ? 'Zone overlay unavailable' : zones ? zones.features.length + ' no-fly zones in red' : 'Loading no-fly zones...'}</p>
        {/* Search Bar */}
        <div style={{ marginBottom: 20 }}>
          <input
            type="text"
            placeholder="Search location..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value)
              handleSearch(e.target.value)
            }}
            style={{
              width: '100%',
              padding: '8px',
              borderRadius: 4,
              border: '1px solid #475569',
              background: '#334155',
              color: 'white',
              fontSize: 14
            }}
          />
          {showSearchResults && searchResults.length > 0 && (
            <div style={{
              position: 'absolute',
              zIndex: 1000,
              background: '#334155',
              border: '1px solid #475569',
              borderRadius: 4,
              marginTop: 4,
              maxHeight: 200,
              overflowY: 'auto',
              width: 280
            }}>
              {searchResults.map((result, i) => (
                <div
                  key={i}
                  onClick={() => selectSearchResult(result)}
                  style={{
                    padding: '8px 12px',
                    cursor: 'pointer',
                    borderBottom: i < searchResults.length - 1 ? '1px solid #475569' : 'none'
                  }}
                  onMouseEnter={(e) => e.target.style.background = '#475569'}
                  onMouseLeave={(e) => e.target.style.background = 'transparent'}
                >
                  <div style={{ fontSize: 13, fontWeight: 'bold' }}>
                    {result.display_name}
                  </div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
                    {result.lat.toFixed(6)}, {result.lon.toFixed(6)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        <div style={{ marginBottom: 20, padding: 10, background: '#334155', borderRadius: 4 }}>
          <strong>Base:</strong> {base ? `${base[0].toFixed(6)}, ${base[1].toFixed(6)}` : 'Loading...'}<br />
          <small style={{ color: '#94a3b8' }}>Click search to change base location</small>
        </div>

        <h3>Delivery Points ({deliveries.length})</h3>
        {deliveries.length === 0 ? (
          <p style={{ color: '#94a3b8' }}>Click on map to add delivery points</p>
        ) : (
          <div>
            {deliveries.map((d, i) => (
              <div key={i} style={{ 
                marginBottom: 8, 
                padding: 8, 
                background: '#334155', 
                borderRadius: 4,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <div>
                  <strong>#{d.orderId}{completed.has(d.orderId) ? ' (delivered)' : ''}</strong> {d.lat.toFixed(5)}, {d.lon.toFixed(5)}
                </div>
                <button 
                  onClick={() => removeDelivery(i)}
                  style={{ padding: '4px 8px', fontSize: 12 }}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}

        <div style={{ marginTop: 20 }}>
          <button
            onClick={() => planRoute()}
            disabled={isPlanning || deliveries.length === 0}
            style={{
              width: '100%',
              padding: 12,
              fontSize: 16,
              background: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: isPlanning ? 'not-allowed' : 'pointer',
              marginBottom: 10
            }}
          >
            {isPlanning ? 'Planning...' : 'Plan Route'}
          </button>

          {path.length > 0 && (
            <div style={{ marginBottom: 10, padding: 10, background: '#334155', borderRadius: 4 }}>
              <strong>Flight telemetry</strong>
              <div>Total round trip: {total.toFixed(1)} m</div>
              <div>Flown: {flown.toFixed(1)} m | Remaining: {(total - flown).toFixed(1)} m</div>
              <div>Speed: {isSimulating ? DRONE_SPEED_MS : 0} m/s</div>
              <div>Elapsed: {elapsed.toFixed(1)} s | ETA: {((total - flown) / DRONE_SPEED_MS).toFixed(1)} s</div>
              <div>Waypoint: {currentPathIndex + 1} / {path.length}</div>
              <div>Estimated battery: {battery.toFixed(2)}%</div>
              <small>Simulation assumes 30 minutes per full charge.</small><br />
              <small style={{ color: '#94a3b8' }}>Includes return to base</small>
            </div>
          )}

          {path.length > 0 && (
            <button
              onClick={isSimulating ? stopSimulation : () => startSimulation()}
              style={{
                width: '100%',
                padding: 12,
                fontSize: 16,
                background: isSimulating ? '#ef4444' : '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer',
                marginBottom: 10
              }}
            >
              {isSimulating ? 'Stop Simulation' : 'Start Simulation'}
            </button>
          )}

          <button
            onClick={clearAll}
            style={{
              width: '100%',
              padding: 12,
              fontSize: 16,
              background: '#64748b',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer',
              marginBottom: 10
            }}
          >
            Clear All
          </button>

          <button
            onClick={onExit}
            style={{
              width: '100%',
              padding: 12,
              fontSize: 16,
              background: '#475569',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer'
            }}
          >
            Exit
          </button>
        </div>
      </div>

      {/* Map */}
      <div className="planner-map">
        <div className="flight-announcements" role="status" aria-live="polite" aria-atomic="true">
          {notice && <div className={'flight-notice ' + notice.kind}>
            <span aria-hidden="true">{notice.kind === 'delivery' ? '\u2713' : '\u2022'}</span>
            {notice.text}
          </div>}
        </div>
        <MapContainer
          center={base}
          zoom={17}
          style={{ height: '100%', width: '100%' }}
        >
          {/* ESRI Satellite */}
          <TileLayer
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            attribution="&copy; Esri"
          />

          {/* OSM Labels Overlay */}
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution="&copy; OpenStreetMap"
            opacity={0.4}
          />

          {zones && <GeoJSON data={zones} interactive={false} style={{ color: '#cf2434', weight: 2, fillOpacity: 0.22 }} />}
          {/* Base marker */}
          <Marker position={base} icon={baseIcon} />

          {/* Delivery points */}
          {deliveries.map((d, i) => (
            <Marker
              key={i}
              position={[d.lat, d.lon]}
              icon={icons.get(d.orderId)}
              alt={'Order #' + d.orderId + (completed.has(d.orderId) ? ' delivered' : ' pending')}
            ><Tooltip>Order #{d.orderId} {completed.has(d.orderId) ? 'delivered' : 'pending'}</Tooltip></Marker>
          ))}

          {/* Planned path */}
          {path.length > 1 && (
            <Polyline
              positions={path}
              color="#3b82f6"
              weight={3}
              opacity={0.7}
            />
          )}

          {/* Drone marker */}
          <DroneMarker position={dronePosition} />

          {/* Click handler */}
          <DeliveryClickHandler onAdd={addDelivery} enabled={addingEnabled} />
        </MapContainer>
      </div>
    </div>
  )
}
