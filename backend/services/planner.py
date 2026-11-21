# backend/services/planner.py
import time
import math
import random
import json
import os
from scipy.spatial import cKDTree
import numpy as np

from services.environment_index import EnvironmentIndex
from services.collision import is_segment_collision_free_3d, intersects_nfz

DATA_DIR = "data"
PLAN_FILE = os.path.join(DATA_DIR, "last_plan.json")


class Node:
    __slots__ = ("x", "y", "z", "parent", "cost")
    def __init__(self, x, y, z, parent=None, cost=0.0):
        self.x = x; self.y = y; self.z = z
        self.parent = parent
        self.cost = cost

    def point2d(self):
        return (self.x, self.y)

    def to_tuple(self):
        return (self.x, self.y, self.z)


class RRTStarPlanner:
    def __init__(self, env_index: EnvironmentIndex, max_iter=5000, step_size=10.0, neighbor_radius=30.0, goal_sample_rate=0.05):
        self.env = env_index
        self.max_iter = max_iter
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.goal_sample_rate = goal_sample_rate

    def _distance2d(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _steer(self, from_node, to_point):
        # returns a new node stepped from from_node towards to_point by step_size
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dz = to_point[2] - from_node.z
        dxy = math.hypot(dx, dy)
        if dxy == 0:
            ratio = 0.0
        else:
            ratio = min(self.step_size / dxy, 1.0)
        nx = from_node.x + dx * ratio
        ny = from_node.y + dy * ratio
        nz = from_node.z + dz * ratio
        return Node(nx, ny, nz)

    def plan(self, start_latlonalt, goal_latlonalt, clearance=10.0, max_time=6.0):
        """
        start_latlonalt, goal_latlonalt: [lat, lon, alt_meters]
        returns dict with raw_path and smoothed_path (lat,lon,alt)
        """
        start_lon, start_lat, start_z = start_latlonalt[1], start_latlonalt[0], start_latlonalt[2]
        goal_lon, goal_lat, goal_z = goal_latlonalt[1], goal_latlonalt[0], goal_latlonalt[2]

        sx, sy = self.env.point_to_3857(start_lon, start_lat)
        gx, gy = self.env.point_to_3857(goal_lon, goal_lat)
        start_node = Node(sx, sy, start_z)
        goal_node = Node(gx, gy, goal_z)

        # quick check if start/goal inside building roof clearance
        if not is_segment_collision_free_3d(self.env, (sx, sy, start_z), (sx, sy, start_z), clearance=clearance):
            raise RuntimeError("Start inside obstacle or too low clearance")
        if not is_segment_collision_free_3d(self.env, (gx, gy, goal_z), (gx, gy, goal_z), clearance=clearance):
            raise RuntimeError("Goal inside obstacle or too low clearance")

        nodes = [start_node]
        best_goal = None
        best_cost = float("inf")
        start_time = time.time()
        pts = [(start_node.x, start_node.y)]
        kdtree = cKDTree(pts)

        for it in range(self.max_iter):
            if time.time() - start_time > max_time:
                break
            # sample point (goal bias)
            if random.random() < self.goal_sample_rate:
                sample = (goal_node.x, goal_node.y, goal_node.z)
            else:
                # sample uniformly in bounding box of environment (based on buildings bbox)
                minx, miny, maxx, maxy = self.env.buildings.total_bounds
                rx = random.uniform(minx, maxx)
                ry = random.uniform(miny, maxy)
                rz = random.uniform(min(start_node.z, goal_node.z) - 10, max(start_node.z, goal_node.z) + 50)
                sample = (rx, ry, rz)
            # nearest node
            dist_idx = kdtree.query([sample[0], sample[1]], k=1)
            dist = dist_idx[0]
            idx = int(dist_idx[1])
            # safety guard
            if idx >= len(nodes):
                continue
            nearest = nodes[idx]
            new_node = self._steer(nearest, sample)

            # NFZ check (2D)
            if intersects_nfz(self.env, nearest.to_tuple(), new_node.to_tuple()):
                continue

            # collision check between nearest and new_node (3D)
            if not is_segment_collision_free_3d(self.env, nearest.to_tuple(), new_node.to_tuple(), clearance=clearance):
                continue

            # compute cost from start
            new_node.parent = nearest
            new_node.cost = nearest.cost + self._distance2d(nearest.point2d(), new_node.point2d())

            # add node to list FIRST, then update pts and kdtree
            nodes.append(new_node)
            pts.append((new_node.x, new_node.y))
            kdtree = cKDTree(pts)

            # find neighbors within radius
            idxs = kdtree.query_ball_point([new_node.x, new_node.y], r=self.neighbor_radius)
            # choose best parent among neighbors
            min_cost = new_node.cost
            best_parent = nearest
            for i in idxs:
                other = nodes[i]
                # check collision from other to new_node and NFZ
                if intersects_nfz(self.env, other.to_tuple(), new_node.to_tuple()):
                    continue
                if not is_segment_collision_free_3d(self.env, other.to_tuple(), new_node.to_tuple(), clearance=clearance):
                    continue
                tentative_cost = other.cost + self._distance2d(other.point2d(), new_node.point2d())
                if tentative_cost < min_cost:
                    min_cost = tentative_cost
                    best_parent = other
            new_node.parent = best_parent
            new_node.cost = min_cost

            # rewire neighbors: see if new_node offers cheaper path to others
            for i in idxs:
                other = nodes[i]
                if other is new_node.parent:
                    continue
                if intersects_nfz(self.env, new_node.to_tuple(), other.to_tuple()):
                    continue
                new_cost = new_node.cost + self._distance2d(new_node.point2d(), other.point2d())
                if new_cost < other.cost:
                    if is_segment_collision_free_3d(self.env, new_node.to_tuple(), other.to_tuple(), clearance=clearance):
                        other.parent = new_node
                        other.cost = new_cost

            # check if new node can connect to goal (also check NFZ)
            if not intersects_nfz(self.env, new_node.to_tuple(), goal_node.to_tuple()) and \
               is_segment_collision_free_3d(self.env, new_node.to_tuple(), goal_node.to_tuple(), clearance=clearance):
                total_cost = new_node.cost + self._distance2d(new_node.point2d(), goal_node.point2d())
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_goal = new_node

        # build path
        if best_goal is None:
            if len(nodes) == 0:
                raise RuntimeError("No nodes in tree")
            pts_arr = [(n.x, n.y) for n in nodes]
            kdtree2 = cKDTree(pts_arr)
            _, nearest_idx = kdtree2.query([goal_node.x, goal_node.y], k=1)
            nearest_node = nodes[int(nearest_idx)]
            if not intersects_nfz(self.env, nearest_node.to_tuple(), goal_node.to_tuple()) and \
               is_segment_collision_free_3d(self.env, nearest_node.to_tuple(), goal_node.to_tuple(), clearance=clearance):
                path_nodes = []
                cur = nearest_node
                while cur is not None:
                    path_nodes.append(cur)
                    cur = cur.parent
                path_nodes = list(reversed(path_nodes))
                path_nodes.append(goal_node)
            else:
                raise RuntimeError("Planner failed to find a path within time")
        else:
            path_nodes = []
            cur = best_goal
            while cur is not None:
                path_nodes.append(cur)
                cur = cur.parent
            path_nodes = list(reversed(path_nodes))
            path_nodes.append(goal_node)

        # convert to lat/lon/alt
        path_ll = []
        for node in path_nodes:
            lon, lat = self.env.point_to_4326(node.x, node.y)
            path_ll.append([lat, lon, float(node.z)])

        # smoothing
        smoothed = self.smooth_path(path_ll) if len(path_ll) >= 4 else path_ll

        # save plan
        plan = {"raw_path": path_ll, "smoothed_path": smoothed}
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(PLAN_FILE, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)

        return plan

    def smooth_path(self, path_ll, s=0.0, k=3, num=200):
        """
        path_ll: list of [lat, lon, alt]
        returns list of [lat, lon, alt] smoothed
        """
        try:
            from scipy.interpolate import splprep, splev
        except Exception:
            return path_ll

        lats = [p[0] for p in path_ll]
        lons = [p[1] for p in path_ll]
        alts = [p[2] for p in path_ll]

        # convert to meters in 3857 for smoothing to avoid latitude distortion
        xs = []
        ys = []
        for lon, lat in zip(lons, lats):
            x, y = self.env.point_to_3857(lon, lat)
            xs.append(x)
            ys.append(y)
        xs = np.array(xs); ys = np.array(ys); zs = np.array(alts)

        try:
            tck, u = splprep([xs, ys, zs], s=s, k=min(k, len(xs)-1))
            u_new = np.linspace(0, 1, num)
            out = splev(u_new, tck)
            xs_s, ys_s, zs_s = out
            smoothed = []
            for x_s, y_s, z_s in zip(xs_s, ys_s, zs_s):
                lon_s, lat_s = self.env.point_to_4326(x_s, y_s)
                smoothed.append([lat_s, lon_s, float(z_s)])
            return smoothed
        except Exception:
            return path_ll
