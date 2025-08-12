from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import requests
import osmnx as ox
import geopandas as gpd
import numpy as np
import random
from shapely.geometry import Point, LineString
from pyproj import Proj, transform
from shapely.ops import nearest_points
from shapely.geometry import Point, Polygon
import math
from threading import Thread
import time


app = Flask(__name__)
CORS(app)



# Initialize SQLite tables
def initialize_db():
    conn = sqlite3.connect("delivery.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            weight REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def init_db():
    conn = sqlite3.connect('locations.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS delivery_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    latitude REAL,
                    longitude REAL
                )''')
    conn.commit()
    conn.close()

# Load base and NFZ data
nfz = gpd.read_file("nfz.geojson")
base_name = "Hanamkonda Head Post Office, Hanamkonda, Telangana, India"
lat, lon = ox.geocode(base_name)
base_point = Point(lon, lat)
base_gdf = gpd.GeoDataFrame(geometry=[base_point], crs="EPSG:4326")
base_buffer = gpd.GeoSeries([base_point], crs="EPSG:4326").to_crs(epsg=3857).buffer(5000).to_crs(epsg=4326)

utm_zone = "EPSG:32644"
utm_proj = Proj(proj='utm', zone=44, ellps='WGS84')
wgs84_proj = Proj(proj='latlong', datum='WGS84')

# Drone

drone_state = {
    "lat": lat,
    "lon": lon,
    "battery": 100.0,
    "altitude": 0,
    "speed": 0,
    "path": [],
    "active": False
}
def utm_to_latlon(x, y):
    lon, lat = transform(utm_proj, wgs84_proj, x, y)
    return lat, lon

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_collision_free(p1, p2, no_fly_zones, goal=None):
    line = LineString([p1, p2])
    for zone in no_fly_zones:
        if line.intersects(zone):
            if goal and Point(goal).within(zone) and p2 == goal:
                continue
            return False
    return True

class RRTStar:
    def __init__(self, start, goal, no_fly_zones, bounds, step_size=10, max_iter=10000, radius=15):
        self.start = start
        self.goal = goal
        self.no_fly_zones = no_fly_zones
        self.minx, self.miny, self.maxx, self.maxy = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.radius = radius
        self.FOCUS_FACTOR = 0.2
        self.tree = {start: None}
        self.costs = {start: 0}

    def find_path(self):
        for _ in range(self.max_iter):
            if random.random() < self.FOCUS_FACTOR:
                random_point = self.goal
            else:
                random_point = (random.uniform(self.minx, self.maxx), random.uniform(self.miny, self.maxy))
            new_node = self.extend_tree(random_point)
            if new_node and distance(new_node, self.goal) < self.step_size:
                if is_collision_free(new_node, self.goal, self.no_fly_zones, goal=self.goal):
                    self.tree[self.goal] = new_node
                    self.costs[self.goal] = self.costs[new_node] + distance(new_node, self.goal)
                    return self.reconstruct_path()
        return None

    def extend_tree(self, random_point):
        nearest_node = min(self.tree.keys(), key=lambda n: distance(n, random_point))
        direction = np.array(random_point) - np.array(nearest_node)
        length = np.linalg.norm(direction)
        if length == 0:
            return None
        new_node = tuple(np.array(nearest_node) + self.step_size * direction / length)
        if not self.is_within_bounds(new_node) or not is_collision_free(nearest_node, new_node, self.no_fly_zones):
            return None
        self.tree[new_node] = nearest_node
        self.costs[new_node] = self.costs[nearest_node] + distance(nearest_node, new_node)
        self.rewire_tree(new_node)
        return new_node

    def is_within_bounds(self, point):
        return self.minx <= point[0] <= self.maxx and self.miny <= point[1] <= self.maxy

    def rewire_tree(self, new_node):
        for node in self.tree.keys():
            if node == new_node:
                continue
            if distance(node, new_node) < self.radius and is_collision_free(new_node, node, self.no_fly_zones):
                new_cost = self.costs[new_node] + distance(new_node, node)
                if new_cost < self.costs[node]:
                    self.tree[node] = new_node
                    self.costs[node] = new_cost

    def reconstruct_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append(node)
            node = self.tree.get(node)
        return path[::-1]

@app.route("/geocode")
def geocode():
    query = request.args.get("q")
    url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=5"
    headers = {"User-Agent": "DroneDeliveryApp/1.0"}
    response = requests.get(url, headers=headers)
    results = response.json()
    places = [{"label": place["display_name"], "lat": float(place["lat"]), "lon": float(place["lon"])} for place in results]
    return jsonify(places)

@app.route("/submit", methods=["POST"])
def submit_location():
    data = request.json
    lat = data.get("lat")
    lon = data.get("lon")
    weight = data.get("weight") or 1.0
    if lat and lon:
        conn = sqlite3.connect("delivery.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO deliveries (lat, lon, weight) VALUES (?, ?, ?)", (lat, lon, weight))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"}), 200
    else:
        return jsonify({"error": "Invalid data"}), 400

@app.route("/save_location", methods=["POST"])
def save_location():
    data = request.json
    lat = data.get("lat")
    lon = data.get("lon")
    if lat is None or lon is None:
        return jsonify({"message": "Invalid coordinates"}), 400
    conn = sqlite3.connect("locations.db")
    c = conn.cursor()
    c.execute("INSERT INTO delivery_points (latitude, longitude) VALUES (?, ?)", (lat, lon))
    conn.commit()
    conn.close()
    return jsonify({"message": "Location saved successfully!"})



def is_inside_nfz(point, nfz_polygons):
    """Returns True if the point is inside any NFZ polygon."""
    for nfz in nfz_polygons:
        if nfz.contains(Point(point)):
            return True
    return False

@app.route("/plan_path", methods=["POST"])
def plan_path():
    data = request.json
    points = data.get("points", [])
    
    if not points:
        return jsonify({"message": "❌ No delivery points received."}), 400

    delivery_points = [Point(p.get("lon") or p.get("lng"), p["lat"]) for p in points]
    print(f"Input points: {[utm_to_latlon(p.x, p.y) for p in delivery_points]}")

    base_utm = base_gdf.to_crs(utm_zone).geometry.iloc[0]
    nfz_utm = nfz.to_crs(utm_zone)
    buffer_utm = base_buffer.to_crs(utm_zone)
    bounds = buffer_utm.total_bounds
    minx, miny, maxx, maxy = bounds

    delivery_points_utm = gpd.GeoSeries(delivery_points, crs="EPSG:4326").to_crs(utm_zone)
    start = (base_utm.x, base_utm.y)
    delivery_coords = [(p.x, p.y) for p in delivery_points_utm]

    full_path = []
    delivery_order = []
    deliveries_latlon = []
    segment_distances = []
    segment_labels = []  # New list for descriptive labels
    total_distance = 0
    max_range = 5000
    max_retries = 5
    search_radius = 50

    i = 0
    current = start
    delivery_num = 1
    processed_indices = set()

    while i < len(delivery_coords):
        if len(processed_indices) >= len(delivery_coords):
            break

        remaining_coords = [dc for idx, dc in enumerate(delivery_coords) if idx not in processed_indices]
        if not remaining_coords:
            break
        goal_idx = min((idx for idx in range(len(delivery_coords)) if idx not in processed_indices),
                       key=lambda j: distance(current, delivery_coords[j]))
        goal = delivery_coords[goal_idx]
        goal_point = Point(goal)
        original_goal = goal
        original_goal_latlon = utm_to_latlon(original_goal[0], original_goal[1])
        print(f"Original goal {delivery_num}: {original_goal_latlon}")

        retries = 0
        is_in_nfz = any(zone.contains(goal_point) for zone in nfz_utm.geometry)
        while is_in_nfz and retries < max_retries:
            print(f"🚫 Delivery {delivery_num} at {original_goal_latlon} is inside NFZ, finding nearest safe point.")
            safe_point = None
            for zone in nfz_utm.geometry:
                if zone.contains(goal_point):
                    angle_step = 10
                    for radius in [search_radius * (retry + 1) for retry in range(5)]:
                        for angle in range(0, 360, angle_step):
                            dx = radius * math.cos(math.radians(angle))
                            dy = radius * math.sin(math.radians(angle))
                            test_point = (goal[0] + dx, goal[1] + dy)
                            test_point_geom = Point(test_point)
                            if (minx <= test_point[0] <= maxx and miny <= test_point[1] <= maxy and
                                not any(z.contains(test_point_geom) for z in nfz_utm.geometry)):
                                safe_point = test_point
                                break
                        if safe_point:
                            break
                    if safe_point:
                        goal = safe_point
                        goal_point = Point(goal)
                        break

            if not safe_point:
                retries += 1
                print(f"🔄 Retry {retries}/{max_retries} for Delivery {delivery_num}")
                if retries == max_retries:
                    print(f"❌ Failed to find safe point for delivery {delivery_num} at {original_goal_latlon}, skipping.")
                    processed_indices.add(goal_idx)
                    break
            else:
                adjusted_goal_latlon = utm_to_latlon(goal[0], goal[1])
                print(f"Adjusted goal {delivery_num}: {adjusted_goal_latlon}")
                break

        if goal_idx in processed_indices:
            i += 1
            continue

        print(f"🛰️ Planning to Delivery {delivery_num} → {goal} (from {original_goal_latlon})")

        rrt = RRTStar(current, goal, nfz_utm.geometry.values, bounds)
        to_delivery = rrt.find_path()

        if not to_delivery:
            print(f"❌ Failed to reach delivery {delivery_num}, skipping.")
            processed_indices.add(goal_idx)
            i += 1
            continue

        seg_dist = sum(distance(to_delivery[j], to_delivery[j + 1]) for j in range(len(to_delivery) - 1))
        rrt_back = RRTStar(goal, start, nfz_utm.geometry.values, bounds)
        to_base = rrt_back.find_path()
        back_dist = sum(distance(to_base[j], to_base[j + 1]) for j in range(len(to_base) - 1)) if to_base else float('inf')

        if seg_dist + back_dist <= max_range:
            full_path += to_delivery[1:]
            deliveries_latlon.append(utm_to_latlon(goal[0], goal[1]))
            delivery_order.append(delivery_num)
            segment_distances.append(round(seg_dist / 1000, 2))
            # Assign a label based on delivery number or proximity (e.g., "Delivery to Hospital Area")
            segment_labels.append(f"Delivery to Point {delivery_num} (Near {original_goal_latlon})")
            total_distance += seg_dist
            current = goal
            processed_indices.add(goal_idx)
            delivery_num += 1
        else:
            print(f"🔋 Battery low. Returning to base from {current}")
            back_path = RRTStar(current, start, nfz_utm.geometry.values, bounds).find_path()
            if back_path:
                full_path += back_path[1:]
                back_seg_dist = sum(distance(back_path[j], back_path[j + 1]) for j in range(len(back_path) - 1))
                segment_distances.append(round(back_seg_dist / 1000, 2))
                segment_labels.append("Return to Base")
                total_distance += back_seg_dist
                current = start
            else:
                print("⚠️ Could not return to base, aborting this segment.")
            break

    if current != start and not any(zone.contains(Point(current)) for zone in nfz_utm.geometry):
        rrt_final = RRTStar(current, start, nfz_utm.geometry.values, bounds)
        final_path = rrt_final.find_path()
        if final_path:
            full_path += final_path[1:]
            final_dist = sum(distance(final_path[j], final_path[j + 1]) for j in range(len(final_path) - 1))
            segment_distances.append(round(final_dist / 1000, 2))
            segment_labels.append("Final Return to Base")
            total_distance += final_dist

    path_latlon = [utm_to_latlon(x, y) for x, y in full_path]

    # Set up simulation state
    drone_state["path"] = path_latlon
    drone_state["active"] = True
    drone_state["battery"] = 100.0
    drone_state["altitude"] = 0
    drone_state["speed"] = 0

    

    # assume segment_distances is a list of floats, e.g. [0.48, 0.78, 1.25]
    segment_labels = []
    n = len(segment_distances)
    for idx in range(n):
        if idx == 0:
            lbl = f"Base to Delivery Point 1"
        elif idx == n - 1:
            lbl = f"Delivery Point {idx} to Base"
        else:
            lbl = f"Delivery Point {idx} to Delivery Point {idx+1}"
        segment_labels.append(lbl)

    # then store in drone_state as before:
    drone_state["segment_labels"]    = segment_labels
    drone_state["segment_distances"] = segment_distances
    drone_state["delivery_logs"]     = []


    return jsonify({
        "path": [[lat, lon] for lat, lon in path_latlon],
        "deliveries": deliveries_latlon,
        "order": delivery_order,
        "distance": round(total_distance / 1000, 2),
        "segment_distances": segment_distances,
        "segment_labels": segment_labels  # New field for labels
    })

@app.route("/telemetry")
def telemetry():
    return jsonify({
        "lat": drone_state["lat"],
        "lon": drone_state["lon"],
        "battery": drone_state["battery"],
        "altitude": drone_state["altitude"],
        "speed": drone_state["speed"],
        "active": drone_state["active"]  # ← include this
    })


@app.route("/logs")
def get_logs():
    return jsonify(drone_state.get("delivery_logs", []))

@app.route("/reset", methods=["POST"])
def reset():
    global drone_state
    drone_state.update({
        "lat": lat,
        "lon": lon,
        "battery": 100.0,
        "altitude": 0,
        "speed": 0,
        "path": [],
        "active": False,
        "delivery_logs": [],
        "segment_labels": [],
        "segment_distances": []
    })
    return jsonify({"status": "reset"})
def simulate_drone():
    while True:
        if drone_state.get("active") and drone_state.get("path"):
            path       = drone_state["path"]
            labels     = drone_state["segment_labels"]
            distances  = drone_state["segment_distances"]

            # reset logs
            drone_state["delivery_logs"] = []

            battery_per_km  = 2.0
            speed_kmph      = 30.0
            step_duration   = 0.3   # 300ms per micro‑step
            earth_radius_km = 6371

            current_seg = 0
            dist_acc     = 0.0
            seg_length   = distances[0] if distances else 0.0
            total_dist   = 0.0

            for i in range(1, len(path)):
                lat1, lon1 = path[i-1]
                lat2, lon2 = path[i]

                # compute this segment’s total km
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a    = (math.sin(dlat/2)**2 +
                        math.cos(math.radians(lat1)) *
                        math.cos(math.radians(lat2)) *
                        math.sin(dlon/2)**2)
                c    = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                seg_km = earth_radius_km * c

                steps = max(int((seg_km / speed_kmph) * 3600 / step_duration), 1)

                # move in micro‑steps
                for s in range(steps):
                    f = (s+1)/steps
                    drone_state["lat"] = lat1 + (lat2 - lat1)*f
                    drone_state["lon"] = lon1 + (lon2 - lon1)*f
                    drone_state["altitude"] = round(random.uniform(45, 55), 1)
                    drone_state["speed"]    = speed_kmph

                    # battery drain
                    drain = (seg_km/steps) * battery_per_km
                    drone_state["battery"] = max(drone_state["battery"] - drain, 0)

                    dist_acc += seg_km/steps
                    time.sleep(step_duration)

                # segment complete?
                if (current_seg < len(distances) 
                    and abs(dist_acc - seg_length) < 0.01):
                    # log it
                    msg = f"{labels[current_seg]}: {seg_length} km"
                    print("✅", msg)
                    drone_state["delivery_logs"].append(msg)
                    total_dist += seg_length
                    dist_acc    = 0.0
                    current_seg += 1
                    # **2s pause at delivery point**
                    time.sleep(2)
                    seg_length = (distances[current_seg]
                                  if current_seg < len(distances) else 0.0)

                if drone_state["battery"] <= 0:
                    drone_state["delivery_logs"].append(
                        "🔋 Battery depleted. Drone stopped."
                    )
                    drone_state["active"] = False
                    return

            # all done → total distance
            drone_state["delivery_logs"].append(
                f"📏 Total Distance: {round(total_dist,2)} km"
            )
            drone_state["active"] = False

        time.sleep(0.0001)

    

Thread(target=simulate_drone, daemon=True).start()   

if __name__ == "__main__":
    init_db()
    initialize_db()
    app.run(debug=True)
