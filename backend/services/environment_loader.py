import os
import json
from services.osm_buildings import build_building_geojson
from services.nfz_builder import build_nfz


DATA_DIR = "data"


def build_environment(place_name: str):
    os.makedirs(DATA_DIR, exist_ok=True)

    buildings_path = os.path.join(DATA_DIR, "buildings.geojson")
    nfz_path = os.path.join(DATA_DIR, "nfz.geojson")

    build_building_geojson(place_name, buildings_path)
    build_nfz(place_name, nfz_path)

    metadata = {
        "place": place_name,
        "building_file": buildings_path,
        "nfz_file": nfz_path
    }

    with open(os.path.join(DATA_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
