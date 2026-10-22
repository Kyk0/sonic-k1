#!/usr/bin/env python3
"""Play SONIC-K1 on a K1 motion clip in MuJoCo: live viewer, or a recorded video to share.

Only needs numpy, mujoco and onnxruntime (plus imageio[ffmpeg] for --record). No Isaac Lab, no GPU.

  python gear_sonic/scripts/play_k1.py                                   # default clip, live viewer
  python gear_sonic/scripts/play_k1.py --motion examples/k1/motions/zombie_walk.npz --record zombie.mp4
  (macOS: the live viewer needs `mjpython` instead of `python`; --record works with plain python)

Same simulation settings as the released MuJoCo evaluation (gear_sonic/scripts/eval_k1_mujoco.py): 5 ms physics, 50 Hz policy,
explicit PD with the training gains and effort limits, and a start in the clip's first frame. Observations are built by
the same code as the deploy script (deploy_k1_booster.py).

Your own motion: any .npz with `qpos` [T, 29] (root xyz, root quat wxyz, 22 joints in MuJoCo k1.xml order) and `fps`,
e.g. a UMR retarget of a human clip (see docs/k1/data.md).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_k1_booster import MJCF, POLICY_DT, Plant, Policy, Reference, quat_to_mat, yaw_of  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SIM_DT = 0.005
DECIMATION = int(round(POLICY_DT / SIM_DT))
DEFAULT_MOTION = REPO / "examples/k1/motions/penguin_walk.npz"
DEFAULT_CKPT = REPO / "checkpoints/sonic-k1"


def build_sim(plant: Plant, ref: Reference, qpos0: np.ndarray, fps: float):
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(MJCF))
    m.opt.timestep = SIM_DT
    m.dof_armature[6:] = plant.armature[plant.i2m]        # training armature
    m.actuator_ctrllimited[:] = 0                          # torques are clipped below, to the training effort limits
    m.actuator_forcelimited[:] = 0
    m.jnt_actfrclimited[:] = 0
    d = mujoco.MjData(m)
    # start in the clip's first frame, with its joint and root velocities
    d.qpos[:] = qpos0[0]
    d.qvel[6:] = ref.jv[0][plant.i2m]
    if len(qpos0) > 1:
        d.qvel[:3] = (qpos0[1, :3] - qpos0[0, :3]) * fps
    mujoco.mj_forward(m, d)
    return mujoco, m, d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", default=str(DEFAULT_MOTION), help="K1 clip (.npz with qpos [T,29] + fps)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                    help="folder with onnx/policy_g1.onnx and k1_plant.json (hf download Kyry10/sonic-k1)")
    ap.add_argument("--record", default=None, help="write an mp4 instead of opening the viewer")
    ap.add_argument("--size", default="1280x720", help="video size for --record, WxH")
    ap.add_argument("--loop", action="store_true", help="viewer: replay the clip until the window is closed")
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    onnx_path, plant_path = ck / "onnx/policy_g1.onnx", ck / "k1_plant.json"
    for p in (onnx_path, plant_path):
        if not p.exists():
            sys.exit(f"missing {p}\n  download the weights first:  hf download Kyry10/sonic-k1 --local-dir {ck}")
    plant = Plant(str(plant_path))
    z = np.load(args.motion, allow_pickle=True)
    qpos0 = np.asarray(z["qpos"], dtype=np.float64)
    fps = float(np.ravel(z["fps"])[0])
    ref = Reference(args.motion, plant)
    policy = Policy(str(onnx_path), plant)
    kp, kd, effort = plant.kp[plant.i2m], plant.kd[plant.i2m], plant.effort[plant.i2m]

    def episode(mujoco, m, d, on_step):
        policy.reset()
        ref.align_heading(yaw_of(d.qpos[3:7]))
        fell_at = None
        for k in range(ref.n - 1):
            rq = d.qpos[3:7].copy()
            q_il, qd_il = d.qpos[7:][plant.m2i], d.qvel[6:][plant.m2i]
            gravity = quat_to_mat(rq).T @ np.array([0.0, 0.0, -1.0])
            prop = policy.proprio(d.qvel[3:6].copy(), q_il, qd_il, gravity)
            cmd, ori = ref.command(k, rq)
            target = policy.act(prop, cmd, ori)[0][plant.i2m]
            for _ in range(DECIMATION):
                tau = kp * (target - d.qpos[7:]) - kd * d.qvel[6:]
                d.ctrl[:] = np.clip(tau, -effort, effort)
                mujoco.mj_step(m, d)
            if fell_at is None and d.qpos[2] < 0.25:
                fell_at = (k + 1) * POLICY_DT
            if on_step(k) is False:
                break
        return fell_at

    clip_s = (ref.n - 1) * POLICY_DT
    name = Path(args.motion).stem
    if args.record:
        import imageio.v2 as imageio

        mujoco, m, d = build_sim(plant, ref, qpos0, fps)
        w, h = (int(x) for x in args.size.lower().split("x"))
        m.vis.global_.offwidth, m.vis.global_.offheight = max(w, 640), max(h, 480)
        renderer = mujoco.Renderer(m, height=h, width=w)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        cam.distance, cam.elevation, cam.azimuth = 1.7, -10.0, 140.0
        writer = imageio.get_writer(args.record, fps=round(1 / POLICY_DT), codec="libx264", quality=8, macro_block_size=8)

        def on_step(_k):
            renderer.update_scene(d, camera=cam)
            writer.append_data(renderer.render())

        fell_at = episode(mujoco, m, d, on_step)
        writer.close()
        print(f"[play_k1] {name}: {clip_s:.1f} s, " + (f"fell at {fell_at:.1f} s" if fell_at else "no fall") + f" -> {args.record}")
        return

    import mujoco.viewer

    while True:
        mujoco, m, d = build_sim(plant, ref, qpos0, fps)
        with mujoco.viewer.launch_passive(m, d) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
            viewer.cam.distance, viewer.cam.elevation = 1.7, -10.0
            t0 = time.perf_counter()

            def on_step(k):
                if not viewer.is_running():
                    return False
                viewer.sync()
                lag = t0 + (k + 1) * POLICY_DT - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)

            fell_at = episode(mujoco, m, d, on_step)
            print(f"[play_k1] {name}: {clip_s:.1f} s, " + (f"fell at {fell_at:.1f} s" if fell_at else "no fall"))
            if not args.loop or not viewer.is_running():
                break


if __name__ == "__main__":
    main()
