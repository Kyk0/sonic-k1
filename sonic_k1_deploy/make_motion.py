#!/usr/bin/env python3
"""Put a K1 clip into motions/ in the app's format, with the checks and fixes a robot test needs.

    python3 make_motion.py clip.npz                          # UMR retarget / any npz with qpos [T, 29] + fps
    python3 make_motion.py rollout.npz --key ref_qpos        # eval_k1_mujoco --render-dir output
    python3 make_motion.py clip.csv --fps 30                 # root xyz, quat xyzw, 22 joints per row
    python3 make_motion.py clip.npz --start 2 --end 8 --lead-in 1.5 --lead-out 1.0 --speed 0.8 --name wave_slow
    python3 make_motion.py --stand 10                        # stand still in Booster's prepare pose (first test)

Output motions/<name>.npz: qpos [T, 29] (root xyz, root quat wxyz, 22 joints in MuJoCo/Booster serial order), fps,
joint_names, source. The app also reads .npz/.csv dropped in motions/ directly; this script just adds the fixes:
  --lead-in S   blend from standing still (Booster prepare pose, upright) to the first frame over S s, after 0.5 s still --
                for clips that do not start from standing (the robot is soft-started to the first frame)
  --lead-out S  blend the last frame back to standing over S s, then 0.5 s still -- ends calm before DAMPING
  --speed F     play at F x speed (0.8 = slower); a gentler first try, not the trained motion
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sonic_k1 import Plant, load_qpos, quat_to_rpy, resolve, slerp, yaw_quat  # noqa: E402

MAX_JOINT_SPEED = 25.0      # rad/s, the retarget QA threshold used for every eval set


def standing(plant, root_xyz, yaw):
    q = np.zeros(29)
    q[:3], q[3:7], q[7:] = root_xyz, yaw_quat(yaw), plant.to_mj(plant.stand)
    return q


def blend(qa, qb, n):
    """n frames from qa to qb (exclusive of qa, inclusive of qb), cosine eased; root quat slerped."""
    out = []
    for i in range(1, n + 1):
        w = 0.5 - 0.5 * math.cos(math.pi * i / n)
        q = qa + (qb - qa) * w
        q[3:7] = slerp(qa[3:7], qb[3:7], w)
        out.append(q)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", help="input .npz / .csv")
    ap.add_argument("--stand", type=float, default=None, help="make a stand-still clip of this many seconds instead")
    ap.add_argument("--key", default=None, help="npz key of the [T, 29] qpos (default qpos, then ref_qpos)")
    ap.add_argument("--fps", type=float, default=None, help="source frame rate (overrides the file)")
    ap.add_argument("--start", type=float, default=0.0, help="trim: start second")
    ap.add_argument("--end", type=float, default=None, help="trim: end second")
    ap.add_argument("--lead-in", type=float, default=0.0)
    ap.add_argument("--lead-out", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--name", default=None, help="output name (default: source file name)")
    ap.add_argument("--out-dir", default=str(HERE / "motions"))
    ap.add_argument("--force", action="store_true", help="write even if the checks fail")
    ap.add_argument("--mjcf", default=None)
    ap.add_argument("--plant", default=None)
    a = ap.parse_args()

    mjcf, _, plant_json = resolve(a.mjcf, None, a.plant)
    if mjcf is None or plant_json is None:
        sys.exit("need k1.xml and k1_plant.json (--mjcf / --plant, or the kit's assets/ and policy/)")
    P = Plant(plant_json, mjcf)

    if a.stand is not None:
        fps = 50.0
        q = np.repeat(standing(P, [0, 0, 0.57], 0.0)[None], max(2, int(a.stand * fps)), 0)
        name, source = a.name or f"stand_{a.stand:g}s", f"stand {a.stand:g} s"
    else:
        if not a.src:
            ap.error("give a source clip or --stand SECONDS")
        q, fps = load_qpos(a.src, a.key)
        fps = a.fps or fps
        if a.src.endswith(".npz"):
            z = np.load(a.src, allow_pickle=True)
            if "robot_joint_names" in z.files or "joint_names" in z.files:
                names = [str(x) for x in z["robot_joint_names" if "robot_joint_names" in z.files else "joint_names"]]
                if names != P.mj:
                    if sorted(names) != sorted(P.mj):
                        sys.exit(f"joint names do not match K1: {names}")
                    q = np.concatenate([q[:, :7], q[:, 7:][:, [names.index(j) for j in P.mj]]], axis=1)
                    print("reordered joints to MuJoCo/serial order")
        i0 = int(round(a.start * fps))
        i1 = len(q) if a.end is None else int(round(a.end * fps))
        q = q[i0:i1].copy()
        name, source = a.name or Path(a.src).stem, str(Path(a.src).name)
        if len(q) < 2:
            sys.exit("clip is empty after trimming")
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)

    if a.lead_in > 0:
        q0 = q[0]
        st = standing(P, q0[:3], quat_to_rpy(q0[3:7])[2])
        q = np.concatenate([np.repeat(st[None], int(0.5 * fps), 0), blend(st, q0, max(1, int(a.lead_in * fps))), q[1:]])
    if a.lead_out > 0:
        qe = q[-1]
        st = standing(P, qe[:3], quat_to_rpy(qe[3:7])[2])
        q = np.concatenate([q, blend(qe, st, max(1, int(a.lead_out * fps))), np.repeat(st[None], int(0.5 * fps), 0)])
    fps_out = fps * a.speed

    # checks (joints in IsaacLab order for the plant constants)
    jp = q[:, 7:][:, P.m2i]
    jv = np.gradient(jp, 1.0 / fps_out, axis=0)
    peak = float(np.max(np.abs(jv)))
    start_dev = float(np.max(np.abs(jp[0] - P.default)[P.legs]))
    out_lo, out_hi = jp < P.lo - 0.02, jp > P.hi + 0.02
    bad_range = sorted({P.il[j] for j in np.where((out_lo | out_hi).any(0))[0]})
    tilt0 = np.degrees(np.max(np.abs(quat_to_rpy(q[0, 3:7])[:2])))
    print(f"{name}: {len(q) / fps_out:.1f} s at {fps_out:g} fps | legs start {start_dev:.2f} rad from standing, tilt {tilt0:.0f} deg"
          f" | peak joint speed {peak:.1f} rad/s")
    problems = []
    if peak > MAX_JOINT_SPEED:
        problems.append(f"joint speed {peak:.1f} > {MAX_JOINT_SPEED} rad/s (retarget glitch?)")
    if bad_range:
        problems.append(f"outside k1.xml ranges: {', '.join(bad_range)} (the app clamps targets to the ranges)")
    if start_dev > 0.35:
        print("  note: does not start from standing; the soft start goes to the first frame -- consider --lead-in 1.5")
    for p in problems:
        print(f"  ! {p}")
    if problems and not a.force:
        sys.exit("not written (use --force to write anyway)")
    out = Path(a.out_dir) / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, qpos=q, fps=np.array([fps_out]), joint_names=np.array(P.mj), source=np.array(source))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
