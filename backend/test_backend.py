"""
Quick test script to verify backend setup
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing backend setup...")

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    from flask import Flask
    print("   ✓ Flask imported")
except ImportError as e:
    print(f"   ✗ Flask import failed: {e}")
    sys.exit(1)

try:
    from flask_cors import CORS
    print("   ✓ flask_cors imported")
except ImportError as e:
    print(f"   ✗ flask_cors import failed: {e}")
    print("   Run: pip install flask-cors")
    sys.exit(1)

try:
    import osmnx as ox
    print("   ✓ osmnx imported")
except ImportError as e:
    print(f"   ✗ osmnx import failed: {e}")
    print("   Run: pip install osmnx")
    sys.exit(1)

# Test 2: Check base resolution
print("\n2. Testing base resolution...")
try:
    from core.base import get_base
    base = get_base()
    print(f"   ✓ Base resolved: {base.lat}, {base.lon}")
except Exception as e:
    print(f"   ✗ Base resolution failed: {e}")
    print("   This might be due to OSM API rate limiting or network issues")
    sys.exit(1)

# Test 3: Check Flask app
print("\n3. Testing Flask app...")
try:
    from app import app
    print("   ✓ Flask app created")
    
    # Test health endpoint
    with app.test_client() as client:
        response = client.get('/health')
        if response.status_code == 200:
            print("   ✓ Health endpoint works")
        else:
            print(f"   ✗ Health endpoint returned {response.status_code}")
            
        response = client.get('/base')
        if response.status_code == 200:
            data = response.get_json()
            print(f"   ✓ Base endpoint works: {data}")
        else:
            print(f"   ✗ Base endpoint returned {response.status_code}")
            
except Exception as e:
    print(f"   ✗ Flask app test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed! Backend is ready.")
print("\nTo start the server, run:")
print("  python app.py")

