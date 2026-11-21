"""
Real-World Drone Delivery Simulation
Phases 0-6: Complete implementation
- Phase 0: Bootstrap
- Phase 1: Base & Geographic Truth
- Phase 2: Real Environment Extraction
- Phase 3: Delivery Input Contract
- Phase 4: Route Sequencing
- Phase 5: Collision-Free Path Planning
- Phase 6: Frontend (in frontend/)
"""
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import logging

from core.base import get_base
from core.environment import build_environment
from core.deliveries import get_delivery_manager
from core.sequencing import compute_route_sequence
from core.environment_index import EnvironmentIndex
from core.planner import RRTPlanner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    """Phase 0: Health check endpoint"""
    return jsonify({"status": "ok"})


@app.route("/base", methods=["GET"])
def base():
    """Phase 1: Get immutable base location"""
    base_loc = get_base()
    return jsonify(base_loc.to_dict())


@app.route("/environment/build", methods=["POST"])
def build_env():
    """Phase 2: Build environment from OSM"""
    data = request.get_json() or {}
    place = data.get("place", "Ashoka Colony, Hanamkonda, Telangana, India")
    
    try:
        metadata = build_environment(place)
        return jsonify(metadata)
    except Exception as e:
        logger.error(f"Error building environment: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/environment/data/<filename>", methods=["GET"])
def get_env_data(filename):
    """Phase 2: Get environment data file"""
    if filename not in ["buildings.geojson", "nfz.geojson", "metadata.json"]:
        return jsonify({"error": "Invalid filename"}), 400
    
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(path)


@app.route("/geocode", methods=["GET"])
def geocode():
    """Geocode search for locations"""
    import requests
    
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    
    try:
        url = "https://nominatim.openstreetmap.org/search"
        response = requests.get(
            url,
            params={
                "q": query,
                "format": "json",
                "limit": 10,
                "addressdetails": 1
            },
            headers={"User-Agent": "RealWorldDrone/1.0"},
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            # Format results
            formatted = [{
                "display_name": r.get("display_name", ""),
                "lat": float(r.get("lat", 0)),
                "lon": float(r.get("lon", 0)),
                "place_id": r.get("place_id", 0)
            } for r in results]
            return jsonify(formatted)
        else:
            return jsonify([])
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return jsonify([])


@app.route("/deliveries", methods=["POST"])
def set_deliveries():
    """Phase 3: Set delivery points (ordered)"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    
    points = data.get("points", [])
    if not points:
        return jsonify({"error": "Empty points list"}), 400
    
    base_loc = get_base()
    delivery_mgr = get_delivery_manager()
    
    try:
        # Load environment to check NFZ
        env_index = None
        try:
            env_index = EnvironmentIndex()
        except Exception as e:
            logger.debug(f"Could not load environment for NFZ checking: {e}")
        
        delivery_mgr.set_deliveries(points, base_loc.lat, base_loc.lon, env_index)
        return jsonify({
            "status": "ok",
            "count": delivery_mgr.count(),
            "deliveries": delivery_mgr.get_deliveries()
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/deliveries", methods=["GET"])
def get_deliveries():
    """Phase 3: Get delivery points"""
    delivery_mgr = get_delivery_manager()
    return jsonify({
        "count": delivery_mgr.count(),
        "deliveries": delivery_mgr.get_deliveries()
    })


@app.route("/deliveries", methods=["DELETE"])
def clear_deliveries():
    """Phase 3: Clear delivery points"""
    delivery_mgr = get_delivery_manager()
    delivery_mgr.clear()
    return jsonify({"status": "ok"})


@app.route("/route/sequence", methods=["GET"])
def get_route_sequence():
    """Phase 4: Get sequenced route (ordered delivery points)"""
    base_loc = get_base()
    delivery_mgr = get_delivery_manager()
    
    deliveries = delivery_mgr.get_deliveries_as_tuples()
    if not deliveries:
        return jsonify({"error": "No delivery points set"}), 400
    
    route = compute_route_sequence(base_loc.lat, base_loc.lon, deliveries)
    
    # Convert to JSON format
    route_json = [{"lat": lat, "lon": lon, "index": i} 
                  for i, (lat, lon) in enumerate(route)]
    
    return jsonify({
        "route": route_json,
        "count": len(route)
    })


@app.route("/route/plan", methods=["POST"])
def plan_route():
    """Phase 5: Plan collision-free path for sequenced route"""
    delivery_mgr = get_delivery_manager()
    base_loc = get_base()
    
    deliveries = delivery_mgr.get_deliveries_as_tuples()
    if not deliveries:
        return jsonify({"error": "No delivery points set"}), 400
    
    # Check if environment is built
    nfz_file = os.path.join(DATA_DIR, "nfz.geojson")
    if not os.path.exists(nfz_file):
        return jsonify({
            "error": "Environment not built. Please build environment first to extract no-fly zones.",
            "hint": "POST to /environment/build"
        }), 400
    
    # Get sequenced route
    route_points = compute_route_sequence(base_loc.lat, base_loc.lon, deliveries)
    
    # Load environment and plan path
    try:
        env_index = EnvironmentIndex()
        
        # Warn if NFZ is empty
        if env_index.nfz is None or (hasattr(env_index.nfz, 'empty') and env_index.nfz.empty):
            logger.warning("NFZ is empty - path planning will not avoid no-fly zones!")
            return jsonify({
                "error": "No-fly zones not loaded. Please rebuild environment to extract NFZ.",
                "hint": "POST to /environment/build with proper place name"
            }), 400
        
        # Use RRT planner with parameters matching old working code
        planner = RRTPlanner(
            env_index, 
            max_iterations=20000,  # Increased to give more time in complex environments
            step_size=10.0,  # Match old code (10m steps)
            goal_bias=0.2,  # Match old code (FOCUS_FACTOR = 0.2)
            min_altitude=10.0
        )
        
        path = planner.plan_route(route_points)
        
        logger.info(f"Successfully planned path with {len(path)} waypoints")
        
        # CRITICAL: Verify path doesn't intersect NFZ (safety check)
        if env_index.nfz is not None and not env_index.nfz.empty:
            violations = []
            point_violations = []
            
            # Check all segments
            for i in range(len(path) - 1):
                start_ll = path[i]
                end_ll = path[i + 1]
                start_3857 = env_index.point_to_3857(start_ll[1], start_ll[0])
                end_3857 = env_index.point_to_3857(end_ll[1], end_ll[0])
                
                # Check segment collision
                collision, reason = env_index.check_segment_collision_2d(start_3857, end_3857, 10.0)
                if collision:
                    if "no-fly zone" in reason.lower() or "nfz" in reason.lower():
                        violations.append(f"Segment {i}-{i+1}: {reason}")
                
                # Also check if points themselves are in NFZ
                in_nfz, nfz_type = env_index.check_point_in_nfz(start_3857)
                if in_nfz:
                    point_violations.append(f"Point {i} in NFZ: {nfz_type}")
            
            # Check last point
            if path:
                last_ll = path[-1]
                last_3857 = env_index.point_to_3857(last_ll[1], last_ll[0])
                in_nfz, nfz_type = env_index.check_point_in_nfz(last_3857)
                if in_nfz:
                    point_violations.append(f"Point {len(path)-1} in NFZ: {nfz_type}")
            
            if violations or point_violations:
                logger.error(f"Path validation FAILED:")
                logger.error(f"  Segment violations: {len(violations)}")
                logger.error(f"  Point violations: {len(point_violations)}")
                for v in violations[:5]:
                    logger.error(f"    {v}")
                for v in point_violations[:5]:
                    logger.error(f"    {v}")
                return jsonify({
                    "error": f"Path validation failed: {len(violations)} segment violations, {len(point_violations)} point violations",
                    "segment_violations": violations[:10],
                    "point_violations": point_violations[:10]
                }), 500
            
            logger.info("Path validation passed: no NFZ violations")
        
        return jsonify({
            "status": "ok",
            "path": path,
            "point_count": len(path),
            "delivery_waypoints": planner.delivery_waypoints,
            "delivery_order": [{"lat": lat, "lon": lon} for lat, lon in route_points[1:-1]],
            "nfz_count": len(env_index.nfz) if env_index.nfz is not None else 0
        })
    except RuntimeError as e:
        logger.error(f"Path planning failed: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in path planning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Path planning error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5002)
