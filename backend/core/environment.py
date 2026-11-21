"""
Phase 2: Real Environment Extraction
Extracts buildings (with height tag only) and no-fly zones from OSM.
Strict rule: Only buildings with explicit height tags are included.
"""
import os
import json
import re
import logging
import osmnx as ox
import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BUILDINGS_FILE = os.path.join(DATA_DIR, "buildings.geojson")
NFZ_FILE = os.path.join(DATA_DIR, "nfz.geojson")
os.makedirs(DATA_DIR, exist_ok=True)


def parse_height(value):
    """
    Parse height value from OSM tag.
    Returns float or None. Only accepts numeric values.
    """
    if value is None:
        return None
    
    try:
        # Extract numeric value (handles "12 m", "12.5", etc.)
        match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", str(value))
        if match:
            return float(match.group(1))
    except Exception:
        pass
    
    return None


def extract_buildings(place_name: str) -> gpd.GeoDataFrame:
    """
    Extract buildings from OSM with STRICT height requirement.
    Only includes buildings with explicit 'height' tag.
    Returns empty GeoDataFrame if no buildings with height found.
    """
    logger.info(f"Extracting buildings from OSM for: {place_name}")
    
    try:
        # Fetch all buildings
        gdf = ox.features_from_place(place_name, {"building": True})
        
        if gdf.empty:
            logger.warning("No buildings found in OSM")
            return gpd.GeoDataFrame(columns=["geometry", "height_m"], crs="EPSG:4326")
        
        # Filter to valid geometries
        gdf = gdf[gdf.geometry.notnull()]
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        
        if gdf.empty:
            logger.warning("No valid building geometries found")
            return gpd.GeoDataFrame(columns=["geometry", "height_m"], crs="EPSG:4326")
        
        # Extract and parse heights
        heights = []
        valid_indices = []
        
        for idx, row in gdf.iterrows():
            height = None
            
            # Check for 'height' tag (ONLY)
            if "height" in row and row["height"]:
                height = parse_height(row["height"])
            
            # Only include if we have a valid height
            if height is not None and height > 0:
                heights.append(height)
                valid_indices.append(idx)
            else:
                # Explicitly skip buildings without height
                pass
        
        if not valid_indices:
            logger.warning("No buildings with height tags found")
            return gpd.GeoDataFrame(columns=["geometry", "height_m"], crs="EPSG:4326")
        
        # Create filtered GeoDataFrame
        filtered_gdf = gdf.loc[valid_indices].copy()
        filtered_gdf["height_m"] = heights
        filtered_gdf = filtered_gdf[["geometry", "height_m"]]
        filtered_gdf = filtered_gdf.reset_index(drop=True)
        
        logger.info(f"Extracted {len(filtered_gdf)} buildings with height tags")
        return filtered_gdf
        
    except Exception as e:
        logger.error(f"Error extracting buildings: {e}")
        return gpd.GeoDataFrame(columns=["geometry", "height_m"], crs="EPSG:4326")


def extract_no_fly_zones(place_name: str) -> gpd.GeoDataFrame:
    """
    Extract no-fly zones from OSM:
    - Hospitals
    - Schools
    - Military areas
    - Aerodromes
    """
    logger.info(f"Extracting no-fly zones from OSM for: {place_name}")
    
    nfz_frames = []
    
    # Define NFZ types and their OSM tags
    nfz_configs = [
        {"type": "hospital", "tags": {"amenity": "hospital"}},
        {"type": "school", "tags": {"amenity": "school"}},
        {"type": "college", "tags": {"amenity": "college"}},
        {"type": "military", "tags": {"landuse": "military", "military": True}},
        {"type": "aerodrome", "tags": {"aeroway": "aerodrome"}},
    ]
    
    for config in nfz_configs:
        nfz_type = config["type"]
        tags = config["tags"]
        
        try:
            gdf = ox.features_from_place(place_name, tags)
            
            if gdf.empty:
                logger.debug(f"No {nfz_type} found")
                continue
            
            # Filter to valid geometries
            gdf = gdf[gdf.geometry.notnull()]
            gdf = gdf[gdf.geometry.type.isin(["Point", "Polygon", "MultiPolygon"])]
            
            if gdf.empty:
                continue
            
            # Convert points to small polygons, buffer polygons
            gdf = gdf.to_crs(epsg=3857)  # Meters
            
            # Use proper buffer sizes based on NFZ type (from old code)
            buffer_sizes = {
                "hospital": 50,
                "school": 50,
                "college": 50,
                "military": 100,
                "aerodrome": 200
            }
            buffer_size = buffer_sizes.get(nfz_type, 50)  # Default 50m
            
            def buffer_geom(geom):
                if geom.geom_type == "Point":
                    return geom.buffer(buffer_size)
                else:
                    return geom.buffer(buffer_size)
            
            gdf["geometry"] = gdf.geometry.apply(buffer_geom)
            gdf = gdf.to_crs(epsg=4326)
            
            gdf["nfz_type"] = nfz_type
            nfz_frames.append(gdf[["geometry", "nfz_type"]])
            
            logger.info(f"Extracted {len(gdf)} {nfz_type} zones")
            
        except Exception as e:
            logger.warning(f"Error extracting {nfz_type}: {e}")
            continue
    
    # Combine all NFZ
    if nfz_frames:
        nfz = gpd.GeoDataFrame(
            pd.concat(nfz_frames, ignore_index=True),
            crs="EPSG:4326"
        )
    else:
        nfz = gpd.GeoDataFrame(columns=["geometry", "nfz_type"], crs="EPSG:4326")
    
    logger.info(f"Total NFZ features: {len(nfz)}")
    return nfz


def build_environment(place_name: str):
    """
    Build environment data from OSM and save to GeoJSON files.
    Always writes files, even if empty.
    """
    logger.info(f"Building environment for: {place_name}")
    
    # Extract buildings (strict height requirement)
    buildings = extract_buildings(place_name)
    
    # Extract no-fly zones
    nfz = extract_no_fly_zones(place_name)
    
    # Save to files (always write, even if empty)
    if not buildings.empty:
        buildings.to_file(BUILDINGS_FILE, driver="GeoJSON")
        logger.info(f"Saved {len(buildings)} buildings to {BUILDINGS_FILE}")
    else:
        # Write empty GeoJSON
        empty_collection = {"type": "FeatureCollection", "features": []}
        with open(BUILDINGS_FILE, "w") as f:
            json.dump(empty_collection, f, indent=2)
        logger.warning(f"Saved empty buildings file: {BUILDINGS_FILE}")
    
    if not nfz.empty:
        nfz.to_file(NFZ_FILE, driver="GeoJSON")
        logger.info(f"Saved {len(nfz)} NFZ features to {NFZ_FILE}")
    else:
        # Write empty GeoJSON
        empty_collection = {"type": "FeatureCollection", "features": []}
        with open(NFZ_FILE, "w") as f:
            json.dump(empty_collection, f, indent=2)
        logger.warning(f"Saved empty NFZ file: {NFZ_FILE}")
    
    # Save metadata
    metadata = {
        "place": place_name,
        "buildings_count": len(buildings),
        "nfz_count": len(nfz),
        "buildings_file": BUILDINGS_FILE,
        "nfz_file": NFZ_FILE
    }
    
    metadata_file = os.path.join(DATA_DIR, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Environment built. Buildings: {len(buildings)}, NFZ: {len(nfz)}")
    return metadata

