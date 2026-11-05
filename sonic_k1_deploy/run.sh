#!/usr/bin/env bash
# One-command launcher on the robot: loads ROS 2 + the Booster interface if rclpy is not importable yet (e.g. over ssh),
# then starts the menu app. Extra args go to app.py (--dry-run, --onnx other.onnx, ...).
# BOOSTER_ROS_SETUP: extra setup.bash to source (default: the BoosterRos2Interface install).
if ! python3 -c "import rclpy" 2>/dev/null; then
  source /opt/ros/humble/setup.bash
  S=${BOOSTER_ROS_SETUP:-/opt/booster/BoosterRos2Interface/install/setup.bash}
  [ -f "$S" ] && source "$S"
fi
cd "$(dirname "$0")" && exec python3 app.py "$@"
