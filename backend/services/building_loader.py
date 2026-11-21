import osmnx as ox
import geopandas as gpd

def load_buildings(place_name):
    tags = {"building": True}
    gdf = ox.features_from_place(place_name, tags)

    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

    def extract_height(row):
        if "height" in row and row["height"]:
            try:
                return float(str(row["height"]).replace("m", ""))
            except:
                return None
        if "building:levels" in row and row["building:levels"]:
            try:
                return float(row["building:levels"]) * 3.0
            except:
                return None
        return None

    gdf["height_m"] = gdf.apply(extract_height, axis=1)

    return gdf[["geometry", "height_m"]]
