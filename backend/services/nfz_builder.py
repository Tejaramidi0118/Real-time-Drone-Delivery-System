import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


NFZ_TAGS = {
    "hospital": {"amenity": "hospital"},
    "school": {"amenity": "school"},
    "college": {"amenity": "college"},
    "military": {"landuse": "military"},
    "airport": {"aeroway": "aerodrome"}
}

BUFFER_RADIUS = {
    "hospital": 150,
    "school": 200,
    "college": 200,
    "military": 500,
    "airport": 1000
}


def build_nfz(place_name: str, output_path: str):
    nfz_frames = []

    for nfz_type, tags in NFZ_TAGS.items():
        try:
            gdf = ox.features_from_place(place_name, tags)
        except Exception:
            # 🔒 ABSOLUTE GUARANTEE: never crash here
            continue

        if gdf is None or gdf.empty:
            continue

        gdf = gdf[gdf.geometry.notnull()]
        if gdf.empty:
            continue

        gdf = gdf.to_crs(epsg=3857)

        gdf["geometry"] = gdf.geometry.apply(
            lambda geom: geom.buffer(BUFFER_RADIUS[nfz_type])
            if isinstance(geom, Point)
            else geom.buffer(BUFFER_RADIUS[nfz_type])
        )

        gdf = gdf.to_crs(epsg=4326)
        gdf["nfz_type"] = nfz_type
        gdf["severity"] = "hard"

        nfz_frames.append(gdf[["geometry", "nfz_type", "severity"]])

    # 🔑 CRITICAL: FILE IS WRITTEN UNCONDITIONALLY
    if nfz_frames:
        nfz = gpd.GeoDataFrame(
            pd.concat(nfz_frames, ignore_index=True),
            crs="EPSG:4326"
        )
    else:
        nfz = gpd.GeoDataFrame(
            columns=["geometry", "nfz_type", "severity"],
            crs="EPSG:4326"
        )

    nfz.to_file(output_path, driver="GeoJSON")
