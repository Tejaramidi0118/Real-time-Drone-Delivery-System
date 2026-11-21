"""
Phase 5: Collision-Free Path Planning
2D RRT planner matching old working code exactly.
"""
import math
import random
import logging
from shapely.geometry import LineString, Point

from core.environment_index import EnvironmentIndex

logger = logging.getLogger(__name__)


class RRTPlanner:
    """
    2D RRT planner for collision-free paths.
    Matches old working code logic exactly.
    """
    
    def __init__(self, env_index: EnvironmentIndex, 
                 max_iterations: int = 10000,
                 step_size: float = 10.0,
                 goal_bias: float = 0.2,
                 min_altitude: float = 10.0):
        self.env = env_index
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.min_altitude = min_altitude
    
    def _distance_2d(self, a: tuple, b: tuple) -> float:
        """Euclidean distance in 2D"""
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    def _steer(self, from_point: tuple, to_point: tuple) -> tuple:
        """Steer from from_point towards to_point by step_size"""
        d = self._distance_2d(from_point, to_point)
        if d <= self.step_size:
            return to_point
        
        ratio = self.step_size / d
        return (
            from_point[0] + (to_point[0] - from_point[0]) * ratio,
            from_point[1] + (to_point[1] - from_point[1]) * ratio
        )
    
    def _is_collision_free(self, p1: tuple, p2: tuple, goal: tuple = None) -> bool:
        """
        Simple collision check - matches old code EXACTLY.
        Only checks if segment intersects NFZ, not if nodes are in NFZ.
        """
        line = LineString([p1, p2])
        
        # Check NFZ - simple intersection check (matches old code exactly)
        # Old code: for zone in no_fly_zones: if line.intersects(zone): ...
        if self.env.nfz is not None and not self.env.nfz.empty:
            # Use .values to get geometry array like old code
            for nfz_geom in self.env.nfz.geometry.values:
                if nfz_geom.intersects(line):
                    # Special case: if goal is provided and segment ends at goal, allow it
                    # Old code: if goal and Point(goal).within(zone) and p2 == goal: continue
                    if goal is not None and p2 == goal:
                        goal_point = Point(goal)
                        if nfz_geom.contains(goal_point):
                            continue  # Allow segment to goal even if goal is in NFZ
                    return False
        
        # Don't check buildings for now - old code doesn't check buildings in is_collision_free
        # Buildings are checked separately in 3D collision checking
        
        return True
    
    def plan_segment(self, start_lat: float, start_lon: float,
                     goal_lat: float, goal_lon: float) -> list:
        """
        Plan a collision-free path from start to goal.
        Matches old code's RRT* logic exactly.
        """
        # Convert to EPSG:3857
        start_3857 = self.env.point_to_3857(start_lon, start_lat)
        goal_3857 = self.env.point_to_3857(goal_lon, goal_lat)
        
        # CRITICAL: If start or goal are in NFZ, find nearest safe point
        # This prevents RRT from being "trapped" if takeoff/landing is in NFZ buffer
        try:
            in_start_nfz, _ = self.env.check_point_in_nfz(start_3857)
            if in_start_nfz:
                logger.warning(f"Start point ({start_lat}, {start_lon}) is in NFZ, adjusting...")
                start_3857 = self.env.find_nearest_safe_point(start_3857)
                start_lon, start_lat = self.env.point_to_4326(start_3857[0], start_3857[1])
                logger.info(f"Adjusted start to: ({start_lat:.6f}, {start_lon:.6f})")
            
            in_goal_nfz, _ = self.env.check_point_in_nfz(goal_3857)
            if in_goal_nfz:
                logger.warning(f"Goal point ({goal_lat}, {goal_lon}) is in NFZ, adjusting...")
                goal_3857 = self.env.find_nearest_safe_point(goal_3857)
                goal_lon, goal_lat = self.env.point_to_4326(goal_3857[0], goal_3857[1])
                logger.info(f"Adjusted goal to: ({goal_lat:.6f}, {goal_lon:.6f})")
        except Exception as e:
            logger.error(f"Error adjusting points: {e}")
            # Continue with original points if adjustment fails

        logger.info(f"Planning path from ({start_lat:.6f}, {start_lon:.6f}) to ({goal_lat:.6f}, {goal_lon:.6f})")
        
        # Get bounds - use base buffer approach from old code
        # Old code: base_buffer = base_gdf.to_crs(epsg=3857).buffer(5000).to_crs(epsg=4326)
        # Then: bounds = buffer_utm.total_bounds
        base_lon, base_lat = 79.548670, 18.006020
        base_3857 = self.env.point_to_3857(base_lon, base_lat)
        buffer_m = 5000  # 5km buffer like old code
        
        # Start with base buffer
        min_x = base_3857[0] - buffer_m
        min_y = base_3857[1] - buffer_m
        max_x = base_3857[0] + buffer_m
        max_y = base_3857[1] + buffer_m
        
        # Expand bounds to include start and goal (old code does this)
        min_x = min(min_x, start_3857[0], goal_3857[0])
        min_y = min(min_y, start_3857[1], goal_3857[1])
        max_x = max(max_x, start_3857[0], goal_3857[0])
        max_y = max(max_y, start_3857[1], goal_3857[1])
        
        bounds = (min_x, min_y, max_x, max_y)
        logger.debug(f"Planning bounds: min=({min_x:.0f}, {min_y:.0f}), max=({max_x:.0f}, {max_y:.0f})")
        
        # RRT tree - matches old code
        tree = {start_3857: None}
        costs = {start_3857: 0.0}
        
        for iteration in range(self.max_iterations):
            # Sample point (goal bias) - matches old code exactly
            if random.random() < self.goal_bias:
                random_point = goal_3857
            else:
                random_point = (
                    random.uniform(bounds[0], bounds[2]),
                    random.uniform(bounds[1], bounds[3])
                )
            
            # Extend tree towards random point
            new_node = self._extend_tree(random_point, tree, costs, bounds, goal_3857)
            
            # Check if we can reach goal from new node (matches old code)
            if new_node and self._distance_2d(new_node, goal_3857) < self.step_size:
                if self._is_collision_free(new_node, goal_3857, goal=goal_3857):
                    tree[goal_3857] = new_node
                    costs[goal_3857] = costs[new_node] + self._distance_2d(new_node, goal_3857)
                    # Reconstruct path
                    path = self._reconstruct_path(tree, start_3857, goal_3857)
                    logger.info(f"Found path with {len(path)} points after {iteration + 1} iterations")
                    return path
        
        # Failed to find path - provide detailed error
        logger.error(f"RRT failed after {self.max_iterations} iterations")
        logger.error(f"  Start: ({start_lat:.6f}, {start_lon:.6f})")
        logger.error(f"  Goal: ({goal_lat:.6f}, {goal_lon:.6f})")
        logger.error(f"  Bounds: ({bounds[0]:.0f}, {bounds[1]:.0f}) to ({bounds[2]:.0f}, {bounds[3]:.0f})")
        logger.error(f"  Tree size: {len(tree)} nodes")
        logger.error(f"  NFZ count: {len(self.env.nfz) if self.env.nfz is not None and not self.env.nfz.empty else 0}")
        
        raise RuntimeError(
            f"RRT planner failed to find path after {self.max_iterations} iterations.\n"
            f"Start: ({start_lat:.6f}, {start_lon:.6f}), Goal: ({goal_lat:.6f}, {goal_lon:.6f})\n"
            f"Tree expanded to {len(tree)} nodes. Area may be too constrained by NFZ zones."
        )
    
    def _extend_tree(self, random_point: tuple, tree: dict, costs: dict, 
                     bounds: tuple, goal: tuple) -> tuple:
        """Extend tree towards random point - matches old code exactly"""
        # Find nearest node (matches old code)
        nearest_node = min(tree.keys(), key=lambda n: self._distance_2d(n, random_point))
        
        # Steer towards random point (matches old code's extend_tree logic)
        direction = (
            random_point[0] - nearest_node[0],
            random_point[1] - nearest_node[1]
        )
        length = math.hypot(direction[0], direction[1])
        
        if length == 0:
            return None
        
        # Create new node by stepping towards random point
        new_node = (
            nearest_node[0] + self.step_size * direction[0] / length,
            nearest_node[1] + self.step_size * direction[1] / length
        )
        
        # Check bounds (matches old code's is_within_bounds)
        if not (bounds[0] <= new_node[0] <= bounds[2] and bounds[1] <= new_node[1] <= bounds[3]):
            return None
        
        # Check collision for segment - CRITICAL: matches old code's is_collision_free
        # Old code: if not is_collision_free(nearest_node, new_node, self.no_fly_zones): return None
        if not self._is_collision_free(nearest_node, new_node, goal=None):
            return None
        
        # Add node to tree (matches old code)
        tree[new_node] = nearest_node
        costs[new_node] = costs[nearest_node] + self._distance_2d(nearest_node, new_node)
        
        # Rewire tree (matches old code's rewire_tree)
        self._rewire_tree(new_node, tree, costs, goal)
        
        return new_node
    
    def _rewire_tree(self, new_node: tuple, tree: dict, costs: dict, goal: tuple):
        """Rewire tree - simplified version matching old code"""
        radius = 15.0  # Neighbor radius (like old code)
        
        for node in tree.keys():
            if node == new_node:
                continue
            
            if self._distance_2d(node, new_node) < radius:
                if self._is_collision_free(new_node, node, goal=None):
                    new_cost = costs[new_node] + self._distance_2d(new_node, node)
                    if new_cost < costs[node]:
                        tree[node] = new_node
                        costs[node] = new_cost
    
    def _reconstruct_path(self, tree: dict, start: tuple, goal: tuple) -> list:
        """Reconstruct path from tree"""
        path_3857 = []
        node = goal
        
        while node is not None:
            path_3857.append(node)
            node = tree.get(node)
        
        path_3857.reverse()
        
        # Convert to lat/lon
        path = []
        for x, y in path_3857:
            lon, lat = self.env.point_to_4326(x, y)
            path.append([lat, lon])
        
        logger.info(f"Found path with {len(path)} points")
        return path
    
    def plan_route(self, route_points: list) -> list:
        """
        Plan collision-free path for entire route.
        Plans one segment at a time.
        """
        if len(route_points) < 2:
            return route_points
        
        self.delivery_waypoints = []
        full_path = []
        
        for i in range(len(route_points) - 1):
            start = route_points[i]
            goal = route_points[i + 1]
            
            logger.info(f"Planning segment {i+1}: ({start[0]:.6f}, {start[1]:.6f}) -> ({goal[0]:.6f}, {goal[1]:.6f})")
            
            segment_path = self.plan_segment(start[0], start[1], goal[0], goal[1])
            
            # Merge segments
            if i == 0:
                full_path.extend(segment_path)
            else:
                full_path.extend(segment_path[1:])  # Skip duplicate point
        
            if i < len(route_points) - 2:
                self.delivery_waypoints.append(len(full_path) - 1)

        logger.info(f"Planned complete route with {len(full_path)} points")
        return full_path
