"""
Manual NFZ Creation Script
Creates NFZ zones around known hospitals and schools in Hanamkonda area
"""
import os
import json
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

# Data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
NFZ_FILE = os.path.join(DATA_DIR, "nfz.geojson")
os.makedirs(DATA_DIR, exist_ok=True)

# Transformer for creating buffers in meters
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
transformer_back = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

# Known hospitals and schools in Hanamkonda area (from map labels)
# Coordinates from the map you showed
NFZ_LOCATIONS = [
    # Hospitals
    {"name": "Rohini Super Speciality Hospital", "type": "hospital", "lat": 17.9955, "lon": 79.5465, "buffer": 150},
    {"name": "Prithvi Hospital", "type": "hospital", "lat": 17.9960, "lon": 79.5470, "buffer": 150},
    {"name": "Aparna ENT Hospital", "type": "hospital", "lat": 17.9958, "lon": 79.5468, "buffer": 150},
    {"name": "Sri Sai Nursing Home", "type": "hospital", "lat": 17.9953, "lon": 79.5462, "buffer": 150},
    {"name": "Khan children specialist", "type": "hospital", "lat": 17.9954, "lon": 79.5463, "buffer": 150},
    
    # Schools
    {"name": "Raman High School", "type": "school", "lat": 17.9956, "lon": 79.5466, "buffer": 200},
    {"name": "Arts College -City", "type": "college", "lat": 17.9962, "lon": 79.5472, "buffer": 200},
    {"name": "Arts College Administrative Building", "type": "college", "lat": 17.9963, "lon": 79.5473, "buffer": 200},
]

def create_nfz():
    """Create NFZ GeoJSON from known locations"""
    features = []
    
    for loc in NFZ_LOCATIONS:
        # Convert to Web Mercator
        x, y = transformer.transform(loc["lon"], loc["lat"])
        
        # Create buffer in meters
        point = Point(x, y)
        buffered = point.buffer(loc["buffer"])
        
        # Convert back to lat/lon
        coords = list(buffered.exterior.coords)
        lon_lat_coords = []
        for x_coord, y_coord in coords:
            lon, lat = transformer_back.transform(x_coord, y_coord)
            lon_lat_coords.append([lon, lat])
        
        feature = {
            "type": "Feature",
            "properties": {
                "nfz_type": loc["type"],
                "name": loc["name"]
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [lon_lat_coords]
            }
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Save to file
    with open(NFZ_FILE, "w") as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Created {len(features)} NFZ zones:")
    for loc in NFZ_LOCATIONS:
        print(f"  - {loc['name']} ({loc['type']}) - {loc['buffer']}m buffer")
    print(f"\nSaved to: {NFZ_FILE}")
    
    return len(features)

if __name__ == "__main__":
    count = create_nfz()
    print(f"\n✓ Successfully created {count} NFZ zones!")

