"""
Phase 3: Delivery Input Contract
Accepts ordered delivery points, validates them, and stores them.
"""
import math
import logging

logger = logging.getLogger(__name__)


def validate_delivery_point(lat: float, lon: float, bounds: dict = None) -> tuple:
    """
    Validate a delivery point.
    Returns (lat, lon) if valid, raises ValueError if invalid.
    
    Args:
        lat: Latitude
        lon: Longitude
        bounds: Optional bounding box {"min_lat": ..., "max_lat": ..., "min_lon": ..., "max_lon": ...}
    """
    # Check numeric
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")
    
    # Check valid range
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude out of range: {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude out of range: {lon}")
    
    # Check NaN/Inf
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError(f"Coordinates must be finite: lat={lat}, lon={lon}")
    
    # Check bounds if provided
    if bounds:
        if lat < bounds.get("min_lat", -90) or lat > bounds.get("max_lat", 90):
            raise ValueError(f"Latitude {lat} outside bounds")
        if lon < bounds.get("min_lon", -180) or lon > bounds.get("max_lon", 180):
            raise ValueError(f"Longitude {lon} outside bounds")
    
    return (lat, lon)


def validate_delivery_list(points: list, base_lat: float, base_lon: float, env_index=None) -> list:
    """
    Validate and normalize a list of delivery points.
    Preserves order of placement.
    
    Args:
        points: List of dicts with "lat" and "lon" keys
        base_lat: Base latitude (for bounds calculation)
        base_lon: Base longitude (for bounds calculation)
    
    Returns:
        List of validated (lat, lon) tuples in order
    
    Raises:
        ValueError if validation fails
    """
    if not isinstance(points, list):
        raise ValueError("Points must be a list")
    
    if len(points) == 0:
        raise ValueError("Empty delivery list")
    
    # Define reasonable bounds (within ~10km of base)
    bounds = {
        "min_lat": base_lat - 0.1,  # ~11km
        "max_lat": base_lat + 0.1,
        "min_lon": base_lon - 0.1,
        "max_lon": base_lon + 0.1,
    }
    
    validated = []
    for i, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError(f"Point {i} must be a dict with 'lat' and 'lon' keys")
        
        lat = point.get("lat")
        lon = point.get("lon")
        
        if lat is None or lon is None:
            raise ValueError(f"Point {i} missing 'lat' or 'lon'")
        
        validated_point = validate_delivery_point(lat, lon, bounds)
        
        # If env_index provided, check if point is in NFZ and adjust
        if env_index is not None:
            point_3857 = env_index.point_to_3857(validated_point[1], validated_point[0])
            in_nfz, nfz_type = env_index.check_point_in_nfz(point_3857)
            if in_nfz:
                logger.warning(f"Delivery point {i+1} is in NFZ ({nfz_type}), finding safe alternative")
                try:
                    safe_point_3857 = env_index.find_nearest_safe_point(point_3857)
                    safe_lon, safe_lat = env_index.point_to_4326(safe_point_3857[0], safe_point_3857[1])
                    validated_point = (safe_lat, safe_lon)
                    logger.info(f"Adjusted delivery point {i+1} to safe location: {safe_lat:.6f}, {safe_lon:.6f}")
                except RuntimeError as e:
                    # If no safe point found, fail explicitly
                    raise ValueError(f"Delivery point {i+1} is in NFZ and cannot be adjusted to safe location: {e}")
        
        validated.append(validated_point)
    
    logger.info(f"Validated {len(validated)} delivery points")
    return validated


class DeliveryManager:
    """Manages delivery points in order"""
    
    def __init__(self):
        self._deliveries = []  # List of (lat, lon) tuples
    
    def set_deliveries(self, points: list, base_lat: float, base_lon: float, env_index=None):
        """
        Set delivery points (replaces existing).
        Validates and stores in order.
        If env_index provided, adjusts points in NFZ to nearest safe location.
        """
        self._deliveries = validate_delivery_list(points, base_lat, base_lon, env_index)
        logger.info(f"Set {len(self._deliveries)} delivery points")
    
    def get_deliveries(self) -> list:
        """Get delivery points as list of dicts"""
        return [{"lat": lat, "lon": lon, "order": i + 1} 
                for i, (lat, lon) in enumerate(self._deliveries)]
    
    def get_deliveries_as_tuples(self) -> list:
        """Get delivery points as list of (lat, lon) tuples"""
        return self._deliveries.copy()
    
    def clear(self):
        """Clear all delivery points"""
        self._deliveries = []
        logger.info("Cleared delivery points")
    
    def count(self) -> int:
        """Get number of delivery points"""
        return len(self._deliveries)


# Singleton instance
_delivery_manager = None


def get_delivery_manager() -> DeliveryManager:
    """Get the singleton delivery manager"""
    global _delivery_manager
    if _delivery_manager is None:
        _delivery_manager = DeliveryManager()
    return _delivery_manager

