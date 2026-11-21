"""
Phase 4: Route Sequencing
Deterministic algorithm to order delivery points.
NOT path planning - just decides the order.
"""
import math
import logging

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points in kilometers.
    Uses Haversine formula.
    """
    R = 6371.0  # Earth radius in km
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def compute_route_sequence(base_lat: float, base_lon: float, deliveries: list) -> list:
    """Visit the nearest outstanding stop from each position, then return to base."""
    base = (base_lat, base_lon)
    route = [base]
    remaining = list(deliveries)
    while remaining:
        current = route[-1]
        index = min(range(len(remaining)), key=lambda i: haversine_distance(*current, *remaining[i]))
        route.append(remaining.pop(index))
    if deliveries:
        route.append(base)
    return route


def get_route_segments(route: list) -> list:
    """
    Convert route into segments for path planning.
    Returns list of (start, end) tuples.
    """
    if len(route) < 2:
        return []
    
    segments = []
    for i in range(len(route) - 1):
        segments.append((route[i], route[i + 1]))
    
    return segments

