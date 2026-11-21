# backend/services/environment_index.py
import os
import json
import geopandas as gpd
from shapely.prepared import prep
from shapely.geometry import Point
from pyproj import Transformer

DATA_DIR = "data"
BUILDINGS_FILE = os.path.join(DATA_DIR, "buildings.geojson")
NFZ_FILE = os.path.join(DATA_DIR, "nfz.geojson")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")


class EnvironmentIndex:
    def __init__(self):
        # transformer lat/lon (EPSG:4326) -> WebMercator (EPSG:3857) (meters)
        self._to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self._to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        self.buildings = None
        self._prepared = []
        self._sindex = None

        # NFZ attributes
        self.nfz = None
        self.nfz_geoms = []
        self.nfz_sindex = None

        self.load()

    def load(self):
        # --- Buildings ---
        if not os.path.exists(BUILDINGS_FILE):
            raise FileNotFoundError(f"Buildings file not found: {BUILDINGS_FILE}")
        gdf = gpd.read_file(BUILDINGS_FILE)
        # Ensure geometry & height_m exist
        if "height_m" not in gdf.columns:
            gdf["height_m"] = 12.0
        # Convert to Web Mercator (meters) for metric checks
        gdf = gdf.to_crs(epsg=3857)
        # simplify heavy polygons slightly for speed (preserve topology; tolerance small)
        gdf["geometry"] = gdf.geometry.simplify(tolerance=0.5, preserve_topology=True)
        self.buildings = gdf.reset_index(drop=True)
        # prepare geometries and sindex
        self._prepared = [prep(geom) for geom in self.buildings.geometry]
        self._sindex = self.buildings.sindex

        # --- NFZs ---
        if os.path.exists(NFZ_FILE):
            nfz_gdf = gpd.read_file(NFZ_FILE)
            if not nfz_gdf.empty:
                self.nfz = nfz_gdf.to_crs(epsg=3857)
                self.nfz_geoms = [prep(g) for g in self.nfz.geometry]
                self.nfz_sindex = self.nfz.sindex
            else:
                self.nfz = None
                self.nfz_geoms = []
                self.nfz_sindex = None
        else:
            self.nfz = None
            self.nfz_geoms = []
            self.nfz_sindex = None

    def point_to_3857(self, lon, lat):
        # returns x, y in meters
        x, y = self._to_3857.transform(lon, lat)
        return x, y

    def point_to_4326(self, x, y):
        lon, lat = self._to_4326.transform(x, y)
        return lon, lat

    def query_buildings_by_point(self, x, y, buffer_m=0.0):
        """
        Query candidate building indices whose bbox intersects the buffer, returns indices list.
        x,y are in EPSG:3857 (meters).
        """
        pt = Point(x, y)
        if buffer_m > 0:
            geom = pt.buffer(buffer_m)
        else:
            geom = pt
        candidate_idx = list(self._sindex.intersection(geom.bounds))
        return candidate_idx

    def get_building_props(self, idx):
        """
        return dict with geometry (in EPSG:3857) and height_m for building at index.
        """
        row = self.buildings.iloc[idx]
        return {"geometry": row.geometry, "height_m": float(row["height_m"]) if "height_m" in row else 12.0}
