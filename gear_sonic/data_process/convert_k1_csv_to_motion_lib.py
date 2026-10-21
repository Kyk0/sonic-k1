"""Convert K1 joint-space CSV motions to motion_lib PKL clips for SONIC-K1 (training or eval_k1_mujoco.py).

CSV format, one row per frame, no header:
    root_x, root_y, root_z, root_qx, root_qy, root_qz, root_qw, <22 joint angles in MuJoCo (k1.xml) order>
MuJoCo joint order: aahead_yaw, aahead_pitch, aaleft_shoulder_pitch, left_shoulder_roll, left_elbow_pitch,
left_elbow_yaw, aaright_shoulder_pitch, right_shoulder_roll, right_elbow_pitch, right_elbow_yaw, left_hip_pitch,
left_hip_roll, left_hip_yaw, left_knee_pitch, left_ankle_pitch, left_ankle_roll, right_hip_pitch, right_hip_roll,
right_hip_yaw, right_knee_pitch, right_ankle_pitch, right_ankle_roll (radians; root in meters, z up).

Usage:
    python gear_sonic/data_process/convert_k1_csv_to_motion_lib.py --input <csv_dir> --output <motion_lib_dir> --fps 30
"""

import argparse
import glob
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_k1_npz_to_motion_lib as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="directory of .csv files")
    ap.add_argument("--output", required=True, help="output motion_lib directory (one .pkl per clip)")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of the CSV files")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(args.input, "*.csv"))):
        c = np.loadtxt(f, delimiter=",")
        q = np.concatenate([c[:, :3], c[:, [6, 3, 4, 5]], c[:, 7:]], axis=1)  # quat xyzw -> wxyz
        assert q.shape[1] == 29, f"{f}: expected 29 columns (3 + 4 + 22), got {c.shape[1]}"
        name = os.path.splitext(os.path.basename(f))[0]
        joblib.dump({name: C.convert_clip(q, args.fps)}, os.path.join(args.output, f"{name}.pkl"), compress=True)
        print(f"{name:32s} {len(q) / args.fps:6.1f} s  root z0 {q[0, 2]:.3f} m")


if __name__ == "__main__":
    main()
