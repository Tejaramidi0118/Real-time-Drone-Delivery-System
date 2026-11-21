from flask import Blueprint, jsonify
import os

from services.environment_loader import build_environment

env_bp = Blueprint("environment", __name__)


@env_bp.route("/environment/build", methods=["POST"])
def build_env():
    place = "Ashoka Colony, Hanamkonda, Telangana, India"
    build_environment(place)
    return jsonify({"status": "environment built"})


@env_bp.route("/environment/data/<filename>", methods=["GET"])
def get_environment_data(filename):
    path = os.path.join("data", filename)
    if not os.path.exists(path):
        return jsonify({"error": "file not found"}), 404
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
