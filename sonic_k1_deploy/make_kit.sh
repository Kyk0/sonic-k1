#!/usr/bin/env bash
# Build a copy-anywhere robot folder: app code + policy/ + assets/k1.xml (+ meshes) + a stand clip + the sample motions.
#   bash sonic_k1_deploy/make_kit.sh <out_dir> [weights_dir]
# weights_dir: a folder with onnx/policy_g1.onnx and k1_plant.json (default: hf download Kyry10/sonic-k1 --local-dir sonic-k1-weights)
set -euo pipefail
OUT=${1:?usage: make_kit.sh <out_dir> [weights_dir]}
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(dirname "$HERE")"
W=${2:-sonic-k1-weights}
if [ ! -f "$W/onnx/policy_g1.onnx" ]; then
  hf download Kyry10/sonic-k1 onnx/policy_g1.onnx k1_plant.json --local-dir "$W"
fi
mkdir -p "$OUT"/{policy,assets,motions,runs}
cp "$HERE"/{app.py,robot.py,sonic_k1.py,make_motion.py,run.sh,README.md} "$OUT"/
cp "$W/onnx/policy_g1.onnx" "$W/k1_plant.json" "$OUT/policy/"
M="$REPO/gear_sonic/data/assets/robot_description/mjcf"
cp "$M/k1.xml" "$OUT/assets/"; cp -r "$M/meshes" "$OUT/assets/"
cp "$REPO"/examples/k1/motions/*.npz "$OUT/motions/" 2>/dev/null || true
( cd "$OUT" && python3 make_motion.py --stand 10 --name 00_stand_10s )
echo "kit ready: $OUT  ->  scp -r $OUT booster@<robot-ip>:~/"
