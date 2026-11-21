"""
Test script to verify NFZ loading and checking
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.environment_index import EnvironmentIndex
from shapely.geometry import Point

print("Testing NFZ loading and checking...")

try:
    env = EnvironmentIndex()
    
    print(f"\nNFZ Status:")
    if env.nfz is None:
        print("  ❌ NFZ is None")
    elif env.nfz.empty:
        print("  ❌ NFZ is empty")
    else:
        print(f"  ✓ NFZ loaded: {len(env.nfz)} features")
        if "nfz_type" in env.nfz.columns:
            types = env.nfz["nfz_type"].value_counts().to_dict()
            print(f"  Types: {types}")
    
    print(f"\nSpatial Index:")
    if env._nfz_sindex is None:
        print("  ❌ NFZ spatial index is None")
    else:
        print(f"  ✓ NFZ spatial index exists")
    
    # Test a point that should be in NFZ (near Rohini Hospital)
    test_lat, test_lon = 17.9955, 79.5465
    test_point_3857 = env.point_to_3857(test_lon, test_lat)
    
    print(f"\nTesting point at {test_lat}, {test_lon} (near Rohini Hospital):")
    in_nfz, nfz_type = env.check_point_in_nfz(test_point_3857)
    print(f"  In NFZ: {in_nfz}, Type: {nfz_type}")
    
    # Test a point that should be safe
    safe_lat, safe_lon = 17.9950, 79.5450
    safe_point_3857 = env.point_to_3857(safe_lon, safe_lat)
    
    print(f"\nTesting point at {safe_lat}, {safe_lon} (should be safe):")
    in_nfz, nfz_type = env.check_point_in_nfz(safe_point_3857)
    print(f"  In NFZ: {in_nfz}, Type: {nfz_type}")
    
    # Test segment intersection
    print(f"\nTesting segment from safe point to NFZ point:")
    collision, reason = env.check_segment_collision_2d(safe_point_3857, test_point_3857)
    print(f"  Collision: {collision}, Reason: {reason}")
    
    print("\n✓ NFZ testing complete")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

