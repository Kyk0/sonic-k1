#!/usr/bin/env python3
"""Play SONIC-K1 on K1 motion clips in MuJoCo: live viewer, or a recorded video to share.

Only needs numpy, mujoco and onnxruntime (plus imageio[ffmpeg] for --record). No Isaac Lab, no GPU.

  python gear_sonic/scripts/play_k1.py                                   # the demo: all sample motions, back to back
  python gear_sonic/scripts/play_k1.py --record demo.mp4                 # the same demo as a video to share
  python gear_sonic/scripts/play_k1.py --motion examples/k1/motions/scissors_jump.npz --record jump.mp4   # one clip
  (macOS: the live viewer needs `mjpython` instead of `python`; --record works with plain python)

Same simulation settings as the released MuJoCo evaluation (gear_sonic/scripts/eval_k1_mujoco.py): 5 ms physics, 50 Hz policy,
explicit PD with the training gains and effort limits, and a start in each clip's first frame. Observations are built by
the same code as the deploy script (deploy_k1_booster.py).

Your own motion: any .npz with `qpos` [T, 29] (root xyz, root quat wxyz, 22 joints in MuJoCo k1.xml order) and `fps`,
e.g. a UMR retarget of a human clip (see docs/k1/data.md). --motion also takes a folder of them.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_k1_booster import MJCF, POLICY_DT, Plant, Policy, Reference, quat_to_mat, yaw_of  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SIM_DT = 0.005
DECIMATION = int(round(POLICY_DT / SIM_DT))
SAMPLES = REPO / "examples/k1/motions"
DEMO_ORDER = ["shadow_boxing", "scissors_jump", "boxing_dodge", "mountain_climber", "high_knees", "forward_lunge",
              "squat", "crouch_walk", "injured_leg_jog", "exaggerated_walk"]
DEFAULT_CKPT = REPO / "checkpoints/sonic-k1"


def motion_list(arg):
    """A clip, or every .npz in a folder (the samples in demo order first)."""
    p = Path(arg) if arg else SAMPLES
    if p.is_file():
        return [p]
    files = sorted(p.glob("*.npz"))
    rank = {n: i for i, n in enumerate(DEMO_ORDER)}
    return sorted(files, key=lambda f: (rank.get(f.stem, len(rank)), f.stem))


def make_model():
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(MJCF))
    m.opt.timestep = SIM_DT
    return mujoco, m


def reset(mujoco, m, d, plant: Plant, ref: Reference, qpos0: np.ndarray, fps: float):
    """Start in the clip's first frame, with its joint and root velocities."""
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos0[0]
    d.qvel[6:] = ref.jv[0][plant.i2m]
    if len(qpos0) > 1:
        d.qvel[:3] = (qpos0[1, :3] - qpos0[0, :3]) * fps
    mujoco.mj_forward(m, d)


def caption(img, text):
    """Clip name in the corner of a recorded frame (Pillow ships with imageio)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return img
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    size = max(18, im.height // 24)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    pad = size // 2
    w = dr.textlength(text, font=font)
    dr.rounded_rectangle([pad, pad, pad * 3 + w, pad * 3 + size], radius=pad, fill=(255, 255, 255))
    dr.text((pad * 2, pad * 2 - size // 8), text, font=font, fill=(25, 25, 30))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", default=None, help="a K1 clip (.npz with qpos [T,29] + fps) or a folder of them "
                                                   "(default: all sample motions, the demo)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                    help="folder with onnx/policy_g1.onnx and k1_plant.json (hf download Kyry10/sonic-k1)")
    ap.add_argument("--record", default=None, help="write an mp4 instead of opening the viewer")
    ap.add_argument("--size", default="1280x720", help="video size for --record, WxH")
    ap.add_argument("--loop", action="store_true", help="viewer: start over after the last clip, until the window is closed")
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    onnx_path, plant_path = ck / "onnx/policy_g1.onnx", ck / "k1_plant.json"
    for p in (onnx_path, plant_path):
        if not p.exists():
            sys.exit(f"missing {p}\n  download the weights first:  hf download Kyry10/sonic-k1 --local-dir {ck}")
    motions = motion_list(args.motion)
    if not motions:
        sys.exit(f"no .npz motions in {args.motion or SAMPLES}")
    plant = Plant(str(plant_path))
    policy = Policy(str(onnx_path), plant)
    kp, kd, effort = plant.kp[plant.i2m], plant.kd[plant.i2m], plant.effort[plant.i2m]
    mujoco, m = make_model()
    m.dof_armature[6:] = plant.armature[plant.i2m]        # training armature
    m.actuator_ctrllimited[:] = 0                          # torques are clipped below, to the training effort limits
    m.actuator_forcelimited[:] = 0
    m.jnt_actfrclimited[:] = 0
    d = mujoco.MjData(m)

    def load(path):
        z = np.load(path, allow_pickle=True)
        return Reference(str(path), plant), np.asarray(z["qpos"], dtype=np.float64), float(np.ravel(z["fps"])[0])

    def episode(ref, on_step):
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
                return fell_at, False
        return fell_at, True

    def report(path, ref, fell_at, extra=""):
        clip_s = (ref.n - 1) * POLICY_DT
        print(f"[play_k1] {path.stem}: {clip_s:.1f} s, " + (f"fell at {fell_at:.1f} s" if fell_at else "no fall") + extra,
              flush=True)

    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    if args.record:
        import imageio.v2 as imageio

        w, h = (int(x) for x in args.size.lower().split("x"))
        m.vis.global_.offwidth, m.vis.global_.offheight = max(w, 640), max(h, 480)
        renderer = mujoco.Renderer(m, height=h, width=w)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = trunk
        cam.distance, cam.elevation, cam.azimuth = 1.7, -10.0, 140.0
        writer = imageio.get_writer(args.record, fps=round(1 / POLICY_DT), codec="libx264", quality=8, macro_block_size=8)
        for path in motions:
            ref, qpos0, fps = load(path)
            reset(mujoco, m, d, plant, ref, qpos0, fps)
            label = path.stem.replace("_", " ") if len(motions) > 1 else None

            def on_step(_k):
                renderer.update_scene(d, camera=cam)
                img = renderer.render()
                writer.append_data(caption(img, label) if label else img)

            fell_at, _ = episode(ref, on_step)
            report(path, ref, fell_at)
        writer.close()
        print(f"[play_k1] {len(motions)} clip(s) -> {args.record}")
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = trunk
        viewer.cam.distance, viewer.cam.elevation = 1.7, -10.0
        while viewer.is_running():
            for path in motions:
                ref, qpos0, fps = load(path)
                with viewer.lock():
                    reset(mujoco, m, d, plant, ref, qpos0, fps)
                try:                                       # clip name in the window (newer MuJoCo viewers)
                    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, path.stem.replace("_", " "), ""))
                except Exception:  # noqa: BLE001
                    pass
                t0 = time.perf_counter()

                def on_step(k):
                    if not viewer.is_running():
                        return False
                    viewer.sync()
                    lag = t0 + (k + 1) * POLICY_DT - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)

                fell_at, finished = episode(ref, on_step)
                if not finished:
                    return
                report(path, ref, fell_at)
            if not args.loop:
                break


if __name__ == "__main__":
    main()
