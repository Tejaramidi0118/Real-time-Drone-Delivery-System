# backend/api/planner.py
from flask import Blueprint, request, jsonify
from services.environment_index import EnvironmentIndex
from services.planner import RRTStarPlanner
from pathlib import Path

planner_bp = Blueprint("planner", __name__)

# singleton environment index (lazy)
_env_index = None

def get_env_index():
    global _env_index
    if _env_index is None:
        _env_index = EnvironmentIndex()
    return _env_index

@planner_bp.route("/plan", methods=["POST"])
def plan_route():
    """
    Body (JSON):
    {
      "start": [lat, lon, alt_m],
      "goal": [lat, lon, alt_m],
      "clearance": 10.0,
      "max_time": 8.0
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    start = data.get("start")
    goal = data.get("goal")
    if not start or not goal:
        return jsonify({"error": "start and goal required"}), 400
    clearance = float(data.get("clearance", 10.0))
    max_time = float(data.get("max_time", 8.0))

    env = get_env_index()
    planner = RRTStarPlanner(env, max_iter=10000, step_size=15.0, neighbor_radius=40.0, goal_sample_rate=0.08)
    try:
        plan = planner.plan(start, goal, clearance=clearance, max_time=max_time)
        return jsonify({"status":"ok", "plan": plan})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@planner_bp.route("/plan/last", methods=["GET"])
def get_last_plan():
    p = Path("data/last_plan.json")
    if not p.exists():
        return jsonify({"error":"no plan found"}), 404
    return p.read_text()
