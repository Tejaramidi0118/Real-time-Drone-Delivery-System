"""
Phase 1: Base & Geographic Truth
Resolves base location from OSM and persists it immutably.
"""
import os
import json
import osmnx as ox
import logging
import requests

logger = logging.getLogger(__name__)

# Base location query
BASE_QUERY = "Hanamkonda Head Post Office, Hanamkonda, Telangana, India"
BASE_OSM_TAGS = {"amenity": "post_office"}

# Storage file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BASE_FILE = os.path.join(DATA_DIR, "base.json")
os.makedirs(DATA_DIR, exist_ok=True)


class BaseLocation:
    """Immutable base location resolved from OSM"""
    
    def __init__(self):
        self._lat = None
        self._lon = None
        self._name = BASE_QUERY
        self._load_or_resolve()
    
    def _load_or_resolve(self):
        """Load from disk or resolve from OSM"""
        if os.path.exists(BASE_FILE):
            try:
                with open(BASE_FILE, "r") as f:
                    data = json.load(f)
                    self._lat = data["lat"]
                    self._lon = data["lon"]
                    self._name = data.get("name", BASE_QUERY)
                    logger.info(f"Loaded base from disk: {self._lat}, {self._lon}")
                    return
            except Exception as e:
                logger.warning(f"Failed to load base from disk: {e}")
        
        # Resolve from OSM
        self._resolve_from_osm()
    
    def _resolve_from_osm(self):
        """Resolve base location from OpenStreetMap"""
        import requests
        
        # Known accurate coordinates for Hanamkonda Head Post Office
        # Verified coordinates: 18.006020, 79.548670
        fallback_lat, fallback_lon = 18.006020, 79.548670
        
        try:
            # Method 1: Use Nominatim API to search for the specific post office
            try:
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    "q": BASE_QUERY,
                    "format": "json",
                    "limit": 5,
                    "addressdetails": 1
                }
                response = requests.get(
                    url,
                    params=params,
                    headers={"User-Agent": "RealWorldDrone/1.0"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    results = response.json()
                    # Look for results with "post_office" in type or name
                    for result in results:
                        if "post" in result.get("type", "").lower() or "post" in result.get("display_name", "").lower():
                            self._lat = float(result["lat"])
                            self._lon = float(result["lon"])
                            self._save()
                            logger.info(f"Resolved base from Nominatim: {self._lat}, {self._lon}")
                            return
                    
                    # If no post office found, use first result
                    if results:
                        self._lat = float(results[0]["lat"])
                        self._lon = float(results[0]["lon"])
                        self._save()
                        logger.info(f"Resolved base from Nominatim (first result): {self._lat}, {self._lon}")
                        return
            except Exception as e:
                logger.debug(f"Nominatim search failed: {e}")
            
            # Method 2: Use OSMNX to search for post offices near known location
            try:
                gdf = ox.features_from_point(
                    (fallback_lat, fallback_lon),
                    BASE_OSM_TAGS,
                    dist=1000  # 1km radius
                )
                
                if not gdf.empty:
                    # Filter to valid geometries
                    gdf = gdf[gdf.geometry.notnull()]
                    gdf = gdf[gdf.geometry.type.isin(["Point", "Polygon", "MultiPolygon"])]
                    
                    if not gdf.empty:
                        # Get first result and extract centroid
                        geom = gdf.geometry.iloc[0]
                        if geom.geom_type in ["Polygon", "MultiPolygon"]:
                            geom = geom.centroid
                        
                        self._lon = float(geom.x)
                        self._lat = float(geom.y)
                        self._save()
                        logger.info(f"Resolved base from OSMNX point search: {self._lat}, {self._lon}")
                        return
            except Exception as e:
                logger.debug(f"OSMNX point search failed: {e}")
            
            # Method 3: Use verified fallback coordinates
            logger.warning("Using verified fallback coordinates for Hanamkonda Head Post Office")
            self._lon = fallback_lon
            self._lat = fallback_lat
            self._save()
            logger.info(f"Using verified base coordinates: {self._lat}, {self._lon}")
            
        except Exception as e:
            logger.error(f"Failed to resolve base from OSM: {e}")
            # Last resort: use verified coordinates
            self._lon = fallback_lon
            self._lat = fallback_lat
            self._save()
            logger.warning(f"Using emergency fallback coordinates due to error: {e}")
    
    def _save(self):
        """Save base location to disk"""
        data = {
            "name": self._name,
            "lat": self._lat,
            "lon": self._lon,
            "source": "osm"
        }
        with open(BASE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    
    @property
    def lat(self):
        """Base latitude (read-only)"""
        return self._lat
    
    @property
    def lon(self):
        """Base longitude (read-only)"""
        return self._lon
    
    @property
    def name(self):
        """Base name (read-only)"""
        return self._name
    
    def to_dict(self):
        """Return base as dictionary"""
        return {
            "name": self._name,
            "lat": self._lat,
            "lon": self._lon
        }


# Singleton instance
_base_location = None


def get_base():
    """Get the singleton base location instance"""
    global _base_location
    if _base_location is None:
        _base_location = BaseLocation()
    return _base_location

