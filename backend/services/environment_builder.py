import json
from services.buildings_loader import load_buildings
from services.nfz_loader import load_no_fly_zones
from config.base import BASE_LOCATION

DATA_DIR = "data"

def build_environment(place_name):
    buildings = load_buildings(place_name)
    nfz = load_no_fly_zones(place_name)

    buildings.to_file(f"{DATA_DIR}/buildings.geojson", driver="GeoJSON")
    nfz.to_file(f"{DATA_DIR}/no_fly_zones.geojson", driver="GeoJSON")

    meta = {
        "place": place_name,
        "base": BASE_LOCATION,
        "building_count": len(buildings),
        "nfz_count": len(nfz)
    }

    with open(f"{DATA_DIR}/environment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta
