import osmnx as ox
import geopandas as gpd

DEFAULT_HEIGHT = 12.0
FLOOR_HEIGHT = 3.0


def fetch_buildings(place_name: str) -> gpd.GeoDataFrame:
    tags = {"building": True}
    gdf = ox.features_from_place(place_name, tags)

    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf.reset_index(drop=True)
    return gdf


def assign_heights(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    heights = []

    for _, row in gdf.iterrows():
        if "height" in row and row["height"]:
            try:
                heights.append(float(row["height"]))
                continue
            except:
                pass

        if "building:levels" in row and row["building:levels"]:
            try:
                heights.append(float(row["building:levels"]) * FLOOR_HEIGHT)
                continue
            except:
                pass

        heights.append(DEFAULT_HEIGHT)

    gdf["height_m"] = heights
    return gdf


def build_building_geojson(place_name: str, output_path: str):
    gdf = fetch_buildings(place_name)
    gdf = assign_heights(gdf)

    gdf = gdf[["geometry", "height_m"]]
    gdf.to_file(output_path, driver="GeoJSON")
