"""
Environment Index for Path Planning
Loads buildings (with height) and NFZ for collision checking.
"""
import os
import json
import logging
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.prepared import prep
from pyproj import Transformer

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BUILDINGS_FILE = os.path.join(DATA_DIR, "buildings.geojson")
NFZ_FILE = os.path.join(DATA_DIR, "nfz.geojson")


class EnvironmentIndex:
    """
    Spatial index for collision checking.
    All geometries stored in EPSG:3857 (meters) for accurate distance calculations.
    """
    
    def __init__(self):
        # Coordinate transformers
        self._to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self._to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        
        # Buildings (with height_m)
        self.buildings = None
        self._buildings_prepared = []
        self._buildings_sindex = None
        
        # No-fly zones
        self.nfz = None
        self._nfz_prepared = []
        self._nfz_sindex = None
        
        self.load()
    
    def load(self):
        """Load buildings and NFZ from GeoJSON files"""
        # Load buildings
        if os.path.exists(BUILDINGS_FILE):
            gdf = gpd.read_file(BUILDINGS_FILE)
            if not gdf.empty:
                # CRITICAL: Only buildings with height_m are valid
                if "height_m" not in gdf.columns:
                    logger.warning("Buildings file missing height_m column")
                    gdf = gpd.GeoDataFrame(columns=["geometry", "height_m"], crs="EPSG:4326")
                else:
                    # Filter out null heights (shouldn't exist per Phase 2, but safety check)
                    gdf = gdf[gdf["height_m"].notna()]
                    gdf = gdf[gdf["height_m"] > 0]
                
                if not gdf.empty:
                    # Convert to Web Mercator for metric operations
                    gdf = gdf.to_crs(epsg=3857)
                    self.buildings = gdf.reset_index(drop=True)
                    self._buildings_prepared = [prep(geom) for geom in self.buildings.geometry]
                    self._buildings_sindex = self.buildings.sindex
                    logger.info(f"Loaded {len(self.buildings)} buildings with heights")
                else:
                    self.buildings = gpd.GeoDataFrame(columns=["geometry", "height_m"], crs="EPSG:3857")
                    logger.warning("No buildings with valid heights")
            else:
                self.buildings = gpd.GeoDataFrame(columns=["geometry", "height_m"], crs="EPSG:3857")
        else:
            self.buildings = gpd.GeoDataFrame(columns=["geometry", "height_m"], crs="EPSG:3857")
            logger.warning(f"Buildings file not found: {BUILDINGS_FILE}")
        
        # Load NFZ
        if os.path.exists(NFZ_FILE):
            gdf = gpd.read_file(NFZ_FILE)
            if not gdf.empty:
                gdf = gdf.to_crs(epsg=3857)
                self.nfz = gdf.reset_index(drop=True)
                self._nfz_prepared = [prep(geom) for geom in self.nfz.geometry]
                self._nfz_sindex = self.nfz.sindex
                logger.info(f"Loaded {len(self.nfz)} NFZ features")
                # Log NFZ types
                if "nfz_type" in self.nfz.columns:
                    nfz_types = self.nfz["nfz_type"].value_counts().to_dict()
                    logger.info(f"NFZ breakdown: {nfz_types}")
            else:
                self.nfz = None
                self._nfz_prepared = []
                self._nfz_sindex = None
                logger.warning(f"NFZ file exists but is empty: {NFZ_FILE}")
                logger.warning("WARNING: No-fly zones not loaded! Path planning may not avoid hospitals/schools.")
        else:
            self.nfz = None
            self._nfz_prepared = []
            self._nfz_sindex = None
            logger.warning(f"NFZ file not found: {NFZ_FILE}")
            logger.warning("WARNING: No-fly zones not loaded! Build environment first to extract NFZ.")
    
    def point_to_3857(self, lon: float, lat: float) -> tuple:
        """Convert lat/lon to Web Mercator (meters)"""
        x, y = self._to_3857.transform(lon, lat)
        return (x, y)
    
    def point_to_4326(self, x: float, y: float) -> tuple:
        """Convert Web Mercator to lat/lon"""
        lon, lat = self._to_4326.transform(x, y)
        return (lon, lat)
    
    def check_segment_collision_2d(self, start_3857: tuple, end_3857: tuple, 
                                   min_altitude: float = 0.0, goal_3857: tuple = None) -> tuple:
        """
        DEPRECATED: Use planner's _is_collision_free instead.
        Kept for backward compatibility.
        """
        """
        Check if a 2D segment collides with buildings or NFZ.
        Simplified approach matching old code.
        
        Args:
            start_3857: (x, y) in EPSG:3857
            end_3857: (x, y) in EPSG:3857
            min_altitude: Minimum flight altitude in meters (for height checking)
            goal_3857: Optional goal point - if segment ends at goal and goal is in NFZ, allow it
        
        Returns:
            (collision: bool, reason: str)
            If collision=True, reason explains why (building height, NFZ, etc.)
        """
        segment = LineString([start_3857, end_3857])
        
        # Check NFZ FIRST (2D, no height consideration) - STRICT: no flying over NFZ
        # But allow if segment ends at goal and goal is in NFZ (like old code)
        if self.nfz is not None and not self.nfz.empty and self._nfz_sindex is not None:
            try:
                candidates = list(self._nfz_sindex.intersection(segment.bounds))
                for idx in candidates:
                    if idx < len(self._nfz_prepared):
                        # Use prepared geometry for fast intersection check
                        if self._nfz_prepared[idx].intersects(segment):
                            # Special case: if goal is provided and segment ends at goal, allow it
                            # (This handles delivery points that are in NFZ but were adjusted)
                            if goal_3857 is not None:
                                goal_point = Point(goal_3857)
                                if self._nfz_prepared[idx].contains(goal_point) and end_3857 == goal_3857:
                                    continue  # Allow segment to goal even if goal is in NFZ
                            
                            nfz_type = self.nfz.iloc[idx].get("nfz_type", "unknown")
                            return (True, f"Segment intersects no-fly zone: {nfz_type} (index {idx})")
            except Exception as e:
                logger.error(f"Error checking NFZ intersection: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # If error, assume collision for safety
                return (True, f"Error checking NFZ: {e}")
        
        # Check buildings (2D footprint, but verify height is known)
        if self.buildings is not None and not self.buildings.empty:
            candidates = list(self._buildings_sindex.intersection(segment.bounds))
            for idx in candidates:
                building_geom = self.buildings.iloc[idx].geometry
                if building_geom.intersects(segment):
                    # Get building height
                    height_m = self.buildings.iloc[idx]["height_m"]
                    
                    # STRICT MODE: If building height is unknown/null, FAIL
                    if height_m is None or height_m <= 0:
                        return (True, f"Segment intersects building with unknown height (index {idx})")
                    
                    # Check if altitude is sufficient (if min_altitude provided)
                    if min_altitude > 0 and min_altitude <= height_m:
                        return (True, f"Segment at altitude {min_altitude}m intersects building {idx} (height: {height_m}m)")
        
        return (False, "")
    
    def check_point_in_nfz(self, point_3857: tuple) -> tuple:
        """
        Check if a point is inside any NFZ.
        
        Args:
            point_3857: (x, y) in EPSG:3857
        
        Returns:
            (in_nfz: bool, nfz_type: str or None)
        """
        if self.nfz is None or self.nfz.empty:
            return (False, None)
        
        if self._nfz_sindex is None:
            return (False, None)
        
        point = Point(point_3857[0], point_3857[1])
        try:
            # Use point bounds for spatial index query (more reliable)
            point_bounds = (point_3857[0], point_3857[1], point_3857[0], point_3857[1])
            candidates = list(self._nfz_sindex.intersection(point_bounds))
            for idx in candidates:
                if idx < len(self._nfz_prepared):
                    # Use prepared geometry for fast contains check
                    if self._nfz_prepared[idx].contains(point):
                        nfz_type = self.nfz.iloc[idx].get("nfz_type", "unknown")
                        return (True, nfz_type)
        except Exception as e:
            logger.error(f"Error checking point in NFZ: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return (False, None)
        
        return (False, None)
    
    def find_nearest_safe_point(self, point_3857: tuple, max_search_radius: float = 500.0) -> tuple:
        """
        Find nearest point outside NFZ from given point.
        Uses spiral search pattern (matches old code logic).
        
        Args:
            point_3857: (x, y) in EPSG:3857
            max_search_radius: Maximum search radius in meters
        
        Returns:
            (x, y) in EPSG:3857
        """
        import math
        
        if self.nfz is None or self.nfz.empty:
            return point_3857
        
        # Check if original point is safe
        in_nfz, _ = self.check_point_in_nfz(point_3857)
        if not in_nfz:
            return point_3857
        
        # Spiral search for safe point (like old code)
        search_radius = 50  # Start with 50m like old code
        angle_step = 10  # 10 degree steps like old code
        max_retries = 5  # Try up to 5 radius increases
        
        for retry in range(max_retries):
            radius = search_radius * (retry + 1)
            if radius > max_search_radius:
                break
                
            for angle in range(0, 360, angle_step):
                dx = radius * math.cos(math.radians(angle))
                dy = radius * math.sin(math.radians(angle))
                test_point = (point_3857[0] + dx, point_3857[1] + dy)
                
                in_nfz, _ = self.check_point_in_nfz(test_point)
                if not in_nfz:
                    logger.info(f"Found safe point at distance {radius:.1f}m from NFZ")
                    return test_point
        
        # If no safe point found, raise error
        logger.error(f"Could not find safe point within {max_search_radius}m of {point_3857}")
        raise RuntimeError(
            f"Cannot find safe point outside NFZ within {max_search_radius}m. "
            f"Point is in a heavily restricted area."
        )
    
    def get_bounds_3857(self) -> tuple:
        """Get bounding box in EPSG:3857"""
        if self.buildings is not None and not self.buildings.empty:
            bounds = self.buildings.total_bounds
            return (bounds[0], bounds[1], bounds[2], bounds[3])  # minx, miny, maxx, maxy
        
        # Default bounds around Hanamkonda
        base_lon, base_lat = 79.548670, 18.006020
        x, y = self.point_to_3857(base_lon, base_lat)
        return (x - 5000, y - 5000, x + 5000, y + 5000)  # 10km box

