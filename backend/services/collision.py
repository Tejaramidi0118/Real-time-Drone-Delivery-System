# backend/services/collision.py
import math
from shapely.geometry import Point, LineString
import numpy as np

def linear_interp(a, b, t):
    return a + (b - a) * t

def segment_length(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def is_segment_collision_free_3d(env_index, p1, p2, clearance=5.0, sample_step=2.0):
    """
    Check if 3D segment between p1 and p2 is free of collisions with building volumes.
    p1, p2: tuples (x, y, z) in EPSG:3857 meters (x,y) and meters altitude (z).
    clearance: minimal clearance above building roof (meters).
    sample_step: horizontal sampling step in meters.
    Returns True if free, False if collides.
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    # horizontal distance
    dist = segment_length(x1, y1, x2, y2)
    if dist == 0:
        steps = 1
    else:
        steps = max(1, int(math.ceil(dist / sample_step)))
    for i in range(steps + 1):
        t = i / steps
        xi = linear_interp(x1, x2, t)
        yi = linear_interp(y1, y2, t)
        zi = linear_interp(z1, z2, t)
        # query nearby buildings using 2D sindex
        cand = env_index.query_buildings_by_point(xi, yi, buffer_m=0.0)
        if not cand:
            continue
        # test each candidate building exact intersection
        pt = Point(xi, yi)
        for idx in cand:
            building = env_index.buildings.iloc[idx]
            geom = building.geometry
            # quick bbox check done by sindex
            if not geom.contains(pt):
                continue
            roof_height = float(building["height_m"]) if "height_m" in building else 12.0
            # if our altitude zi is <= roof_height + clearance -> collision
            if zi <= (roof_height + clearance):
                return False
    return True

def intersects_nfz(env_index, p1, p2):
    """
    p1, p2: (x, y, z) in EPSG:3857
    Returns True if segment intersects any NFZ polygon (2D).
    """
    if env_index.nfz is None or env_index.nfz_sindex is None:
        return False

    line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    candidates = list(env_index.nfz_sindex.intersection(line.bounds))

    for idx in candidates:
        if env_index.nfz_geoms[idx].intersects(line):
            return True
    return False
