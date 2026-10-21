#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Convert UMR K1 retarget npz files to motion_lib PKL format for SONIC training.

Input: UMR retarget npz (data/k1_retarget/full/k1_official/*.npz), qpos [T, 29] =
root pos(3) + root quat wxyz(4) + 22 joints in MuJoCo/k1.xml order, fps~=60.
Output: one PKL per clip (joblib dict {name: entry}), matching the format
documented in new_embodiments.md Step 5 (root_trans_offset, pose_aa, dof,
root_rot, fps). fps is kept at 60 (project decision -- SONIC resamples to 50Hz
internally; do not resample here).

pose_aa uses the MJCF body tree order (23 bodies for K1: trunk + 22 actuated
links; the 2 empty camera-mount bodies were removed from k1.xml so the MJCF
body list matches K1_MUJOCO_TO_ISAACLAB_BODY), which
is what gear_sonic.utils.motion_lib.torch_humanoid_batch.Humanoid_Batch parses
from k1.xml at FK time -- NOT the IsaacLab 23-body order used elsewhere
(gear_sonic/envs/manager_env/robots/k1.py), which drops the 2 camera bodies.

Usage:
    python gear_sonic/data_process/convert_k1_npz_to_motion_lib.py \
        --input ../../data/k1_retarget/full/k1_official \
        --output ../../data/sonic_motionlib/k1_sample_1k \
        --clip-list ../../configs/train_sample_1k.txt \
        --num_workers 16
"""

import argparse
import os

import joblib
import numpy as np
from scipy.spatial import transform

NUM_DOF = 22

# k1.xml MuJoCo DOF order (see tools/k1_inspect_mujoco.py / k1.py comments).
MUJOCO_JOINT_ORDER = [
    "aahead_yaw_joint",
    "aahead_pitch_joint",
    "aaleft_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "aaright_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
assert len(MUJOCO_JOINT_ORDER) == NUM_DOF

# k1.xml MJCF body tree order (trunk + 24 children, depth-first), as parsed by
# Humanoid_Batch.from_mjcf / dumped by tools/k1_inspect_mujoco.py. The two
# camera-mount bodies have no joint of their own (fixed to their parent).
MJCF_BODY_ORDER = [
    "trunk",
    "aahead_yaw_link",
    "aahead_pitch_link",
    "aaleft_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "aaright_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_pitch_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_pitch_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]
NUM_BODIES = len(MJCF_BODY_ORDER)

# Rotation axis per DOF (verified against k1.xml jnt_axis via tools/k1_inspect_mujoco.py):
# *_pitch_joint -> Y, *_roll_joint -> X, *_yaw_joint -> Z.
_AXIS_BY_SUFFIX = {"pitch": [0, 1, 0], "roll": [1, 0, 0], "yaw": [0, 0, 1]}
DOF_AXIS = np.array(
    [_AXIS_BY_SUFFIX[name.split("_")[-2]] for name in MUJOCO_JOINT_ORDER],
    dtype=np.float32,
)

# body index (in MJCF_BODY_ORDER) -> dof index, or None for the root / fixed camera bodies.
BODY_TO_DOF = [None] * NUM_BODIES
for _dof_idx, _joint_name in enumerate(MUJOCO_JOINT_ORDER):
    _link_name = _joint_name.replace("_joint", "_link")
    BODY_TO_DOF[MJCF_BODY_ORDER.index(_link_name)] = _dof_idx


def convert_clip(qpos: np.ndarray, fps: int) -> dict:
    """qpos: (T, 29) root_pos(3) + root_quat_wxyz(4) + 22 dof (MuJoCo order)."""
    root_trans_offset = qpos[:, 0:3].astype(np.float32)
    root_quat_wxyz = qpos[:, 3:7]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]].astype(np.float32)
    dof = qpos[:, 7 : 7 + NUM_DOF].astype(np.float32)

    T = qpos.shape[0]
    pose_aa = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_quat_xyzw).as_rotvec()
    for body_idx, dof_idx in enumerate(BODY_TO_DOF):
        if dof_idx is not None:
            pose_aa[:, body_idx, :] = DOF_AXIS[dof_idx] * dof[:, dof_idx, None]

    return {
        "root_trans_offset": root_trans_offset,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_quat_xyzw,
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),
        "fps": int(round(fps)),
    }


def process_one(args_tuple):
    npz_path, out_path = args_tuple
    if os.path.exists(out_path):
        return "skip", None
    try:
        d = np.load(npz_path, allow_pickle=True)
        joint_names = list(d["robot_joint_names"])
        if joint_names != MUJOCO_JOINT_ORDER:
            return "fail", f"{os.path.basename(npz_path)}: joint order mismatch"
        qpos = d["qpos"]
        fps = float(np.asarray(d["fps"]).reshape(-1)[0])
        name = os.path.splitext(os.path.basename(npz_path))[0]
        entry = convert_clip(qpos, fps)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        joblib.dump({name: entry}, out_path, compress=True)
        return "ok", None
    except Exception as e:  # noqa: BLE001
        return "fail", f"{os.path.basename(npz_path)}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Convert K1 UMR npz to motion_lib PKLs")
    parser.add_argument("--input", required=True, help="Dir of k1_official *.npz files")
    parser.add_argument("--output", required=True, help="Output dir for individual PKLs")
    parser.add_argument(
        "--clip-list",
        default=None,
        help="Optional text file of clip base names (one per line, no extension) to convert",
    )
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    if args.clip_list:
        with open(args.clip_list) as f:
            wanted = {line.strip() for line in f if line.strip()}
        npz_files = [
            os.path.join(args.input, n + ".npz")
            for n in wanted
            if os.path.exists(os.path.join(args.input, n + ".npz"))
        ]
        missing = len(wanted) - len(npz_files)
        if missing:
            print(f"WARNING: {missing} clips in --clip-list not found under {args.input}")
    else:
        npz_files = [
            os.path.join(args.input, f) for f in sorted(os.listdir(args.input)) if f.endswith(".npz")
        ]

    jobs = [
        (p, os.path.join(args.output, os.path.splitext(os.path.basename(p))[0] + ".pkl"))
        for p in npz_files
    ]
    print(f"Converting {len(jobs)} clips -> {args.output} ({args.num_workers} workers)")

    import multiprocessing

    ok = skip = fail = 0
    failures = []
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for status, msg in pool.imap_unordered(process_one, jobs):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                failures.append(msg)

    print(f"Done: {ok} converted, {skip} skipped (existing), {fail} failed")
    for msg in failures[:20]:
        print("  FAIL:", msg)


if __name__ == "__main__":
    main()
