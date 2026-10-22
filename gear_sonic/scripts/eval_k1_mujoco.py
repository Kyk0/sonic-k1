#!/usr/bin/env python3
"""Headless batch MuJoCo sim-to-sim evaluator for SONIC (universal-token) Booster K1 policies.

Adapted from eval_x2_mujoco.py of the AgiBot X2 port (github.com/meetsitaram/GR00T-WholeBodyControl-X2,
Apache-2.0) for the Booster K1, without a viewer and with many clips in parallel worker processes.
The policy runs either from a .pt checkpoint (PyTorch) or from the exported ONNX files (--onnx), and
--parity-onnx checks ONNX against PyTorch on every control step of every rollout.

Pipeline per clip (policy 50 Hz, physics dt 0.005 s, decimation 4, same as training):
  * reference: gear_sonic's own MotionLibRobot on CPU (60 fps pkl -> 50 Hz, FK on k1.xml), i.e. the
    exact arrays IsaacLab feeds the policy (dof pos/vel in IsaacLab order, root pose/vel, body pos);
  * RSI at frame 0: joint pos/vel + root pos/quat/lin vel/ang vel from the reference (no noise);
  * observations: policy `actor_obs` (base_ang_vel, joint_pos_rel, joint_vel, last_action,
    gravity_dir; 10-step history each, oldest first, history primed with the first sample) and
    the g1-encoder tokenizer input (command_multi_future_nonflat + motion_anchor_ori_b_mf_nonflat);
  * actor: g1 encoder MLP -> FSQ (vector_quantize_pytorch, levels [32]*32) -> g1_dyn decoder MLP;
  * action: clip +-20, target = default_joint_pos + action_scale * action, explicit PD torque
    tau = kp (q* - q) - kd qd clamped to the training effort limits, recomputed every physics step.
    Gains/armature/effort/defaults are read from gear_sonic/envs/manager_env/robots/k1.py.

Failure ("paper" rule, the rollout stops): |root height - ref| > 0.25 m or root orientation error
> 1 rad. The "official" im_eval extra checks (ankle/elbow body height error > 0.25 m, ankle position
error > 0.2 m w.r.t. the heading-aligned reference) are recorded but don't stop the rollout.

Usage (from the repo root; config.yaml must sit next to the checkpoint):
  python gear_sonic/scripts/eval_k1_mujoco.py --checkpoint <ckpt_dir>/sonic_k1_it50000_actor.pt \
      --motion-dir <motion_lib_dir> --out <out_dir> --clips all --workers 16 --max-steps 500
  # deployment path (ONNX instead of PyTorch):           add --onnx <dir with policy_g1.onnx>
  # ONNX vs PyTorch action parity on every step:          add --parity-onnx <dir with the ONNX files>
  # write per-clip qpos npz for rendering:                add --render-dir <dir>
<motion_lib_dir> holds motion_lib .pkl clips, e.g. from convert_k1_npz_to_motion_lib.py or
convert_k1_csv_to_motion_lib.py.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]          # gear_sonic/scripts/ -> repo root
MJCF = REPO / "gear_sonic/data/assets/robot_description/mjcf/k1.xml"
K1_PY = REPO / "gear_sonic/envs/manager_env/robots/k1.py"

SIM_DT = 0.005
DECIMATION = 4
CONTROL_DT = SIM_DT * DECIMATION
HIST = 10
ACTION_CLIP = 20.0
PAPER_HEIGHT_THR = 0.25        # eval.yaml anchor_pos threshold (non-adaptive)
PAPER_ORI_THR = 1.0            # eval.yaml anchor_ori_full: angle^2 > 1.0
EE_HEIGHT_THR = 0.25           # eval.yaml ee_body_pos (z only)
FOOT_POS_THR = 0.2             # training foot_pos_xyz (xyz, heading-aligned reference)
EE_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link", "left_elbow_yaw_link", "right_elbow_yaw_link"]
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
# training push_robot event (push_by_setting_velocity, every 4-6 s): max |delta| of root lin (x, y, z) / ang (roll, pitch, yaw) vel
PUSH_LIN = np.array([0.35, 0.35, 0.14])
PUSH_ANG = np.array([0.36, 0.36, 0.55])
PUSH_INTERVAL_S = (4.0, 6.0)
# torque caps of Booster's own K1 deploy config (booster_deploy K1_CFG.effort_limit), MuJoCo / serial joint order
BOOSTER_DEPLOY_EFFORT = [6, 6, 14, 14, 14, 14, 14, 14, 14, 14, 30, 35, 20, 40, 20, 20, 30, 35, 20, 40, 20, 20]


# --------------------------------------------------------------------------------------------
# K1 plant (from k1.py, parsed with ast because the module itself imports isaaclab)
# --------------------------------------------------------------------------------------------
def load_k1_constants(path: Path = K1_PY) -> dict:
    """Evaluate the constant/mapping assignments of k1.py (no isaaclab import)."""
    tree = ast.parse(path.read_text())
    ns: dict = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
        elif isinstance(node, ast.Assign):
            names = [n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            if names and names[0].startswith(("K1_ISAAC", "K1_MUJOCO", "ARMATURE", "EFFORT", "STIFFNESS", "DAMPING")):
                exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    # init_state joint_pos dict literal (default joint positions)
    default = None
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "joint_pos" and isinstance(node.value, ast.Dict):
            default = ast.literal_eval(node.value)
    ns["INIT_JOINT_POS"] = default or {}
    return ns


# (joint regex, gain suffix, actuator spec suffix) exactly as grouped in K1_CFG.actuators
_K1_GROUPS = [
    (r".*_hip_pitch_joint", "HIP_PITCH", "E6408"),
    (r".*_hip_roll_joint", "HIP_ROLL", "E4315"),
    (r".*_hip_yaw_joint", "HIP_YAW", "E4310"),
    (r".*_knee_pitch_joint", "KNEE", "E6416"),
    (r".*_ankle_pitch_joint|.*_ankle_roll_joint", "ANKLE", "ANKLE"),
    (r".*shoulder_pitch_joint|.*_shoulder_roll_joint|.*_elbow_pitch_joint|.*_elbow_yaw_joint", "ARM", "R14"),
    (r".*head_yaw_joint|.*head_pitch_joint", "HEAD", "HT4438"),
]


class K1Plant:
    """Joint names, orderings, PD gains, effort limits, action scale and defaults (IsaacLab order)."""

    def __init__(self):
        c = load_k1_constants()
        m = c["K1_ISAACLAB_TO_MUJOCO_MAPPING"]
        self.il_bodies = list(m["isaaclab_joints"])                    # 23 bodies, trunk first
        self.il_joints = [b.replace("_link", "_joint") for b in self.il_bodies[1:]]
        self.i2m = np.array(m["isaaclab_to_mujoco_dof"])                # q_mj = q_il[i2m]
        self.m2i = np.array(m["mujoco_to_isaaclab_dof"])                # q_il = q_mj[m2i]
        self.mapping = m
        n = len(self.il_joints)
        self.kp, self.kd, self.armature, self.effort = (np.zeros(n) for _ in range(4))
        for i, jn in enumerate(self.il_joints):
            for rx, g, a in _K1_GROUPS:
                if re.fullmatch(rx, jn):
                    self.kp[i], self.kd[i] = c[f"STIFFNESS_{g}"], c[f"DAMPING_{g}"]
                    self.armature[i], self.effort[i] = c[f"ARMATURE_{a}"], c[f"EFFORT_{a}"]
                    break
            else:
                raise KeyError(f"no actuator group for {jn}")
        self.action_scale = 0.25 * self.effort / self.kp               # K1_ACTION_SCALE
        self.default_pos = np.array([float(c["INIT_JOINT_POS"].get(j, 0.0)) for j in self.il_joints])
        self.mj_joints = [self.il_joints[k] for k in self.i2m]


# --------------------------------------------------------------------------------------------
# Small quaternion helpers (wxyz, IsaacLab conventions)
# --------------------------------------------------------------------------------------------
def quat_conj(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_apply(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u, w = q[..., 1:], q[..., :1]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_apply_inv(q, v):
    return quat_apply(quat_conj(q), v)


def quat_to_mat(q):
    w, x, y, z = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def quat_angle(a, b):
    """IsaacLab quat_error_magnitude: rotation angle of a^-1 b."""
    d = quat_mul(quat_conj(a), b)
    return 2.0 * np.arctan2(np.linalg.norm(d[..., 1:], axis=-1), np.abs(d[..., 0]))


def heading_q(q):
    q = np.array(q, dtype=np.float64)
    q[..., 1:3] = 0.0
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# --------------------------------------------------------------------------------------------
# Run config / reference motions
# --------------------------------------------------------------------------------------------
def load_run_config(checkpoint: str):
    from omegaconf import OmegaConf

    path = Path(checkpoint).resolve().parent / "config.yaml"
    return OmegaConf.load(path)


def load_reference_clips(cfg, plant: K1Plant, motion_dir: str, num_clips: int | None) -> list[dict]:
    """Load clips with gear_sonic's MotionLibRobot (CPU, eval mode = no augmentation)."""
    import easydict
    from omegaconf import OmegaConf

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot

    cmd = cfg.manager_env.commands.motion
    mcfg = OmegaConf.to_container(cmd.motion_lib_cfg, resolve=False)
    body_names = list(cmd.body_names)
    m = plant.mapping
    mcfg.update(
        motion_file=str(Path(motion_dir).resolve()),
        mujoco_to_isaaclab_dof=m["mujoco_to_isaaclab_dof"], mujoco_to_isaaclab_body=m["mujoco_to_isaaclab_body"],
        isaaclab_to_mujoco_dof=m["isaaclab_to_mujoco_dof"], isaaclab_to_mujoco_body=m["isaaclab_to_mujoco_body"],
        body_indexes_data=[plant.il_bodies.index(n) for n in body_names],
        adaptive_sampling={"enable": False}, multi_thread=False, override_num_motions_to_load=None,
    )
    mcfg["asset"]["assetRoot"] = str(REPO / mcfg["asset"]["assetRoot"])
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        ml = MotionLibRobot(easydict.EasyDict(mcfg), 1, "cpu")
        n_all = ml._num_unique_motions  # noqa: SLF001
        n = n_all if num_clips is None else min(num_clips, n_all)
        ml.load_motions(random_sample=False, num_motions_to_load=n, start_idx=0, is_evaluation=True)
    finally:
        os.chdir(cwd)
    keys = ml._motion_data_keys  # noqa: SLF001
    cur = ml._curr_motion_ids  # noqa: SLF001
    clips = []
    for i in range(n):
        s, T = int(ml.length_starts[i]), int(ml._motion_num_frames[i])  # noqa: SLF001
        sl = slice(s, s + T)
        f = lambda x: x[sl].double().numpy().copy()  # noqa: E731
        clips.append({
            "index": i, "name": str(keys[int(cur[i])]), "T": T,
            "dof_pos": f(ml.dof_pos), "dof_vel": f(ml.dof_vel),                 # (T,22) IsaacLab order
            "body_pos": f(ml.body_pos_w), "body_quat": f(ml.body_quat_w),        # (T,12,.) tracked bodies
            "body_lin_vel": f(ml.body_lin_vel_w), "body_ang_vel": f(ml.body_ang_vel_w),
            "body_pos_full": f(ml.body_pos_w_full),                              # (T,23,3) IsaacLab order
            "body_names": body_names,
        })
    return clips


# --------------------------------------------------------------------------------------------
# Actor (g1 encoder + FSQ + g1_dyn decoder)
# --------------------------------------------------------------------------------------------
def extract_actor_state(checkpoint: str, encoder: str = "g1") -> dict:
    """Pull encoder/decoder weights out of a full training checkpoint (tolerant unpickling)."""
    import pickle
    import types

    import torch

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            pass

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:  # noqa: BLE001
                return _Stub

    tol = types.ModuleType("_tolerant_pickle")
    for a in dir(pickle):
        setattr(tol, a, getattr(pickle, a))
    tol.Unpickler = _Unpickler
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, pickle_module=tol)
    sd = ck.get("policy_state_dict") or ck.get("actor_model_state_dict")
    enc, dec = f"actor_module.encoders.{encoder}.module.", "actor_module.decoders.g1_dyn.module."
    out = {}
    for k, v in sd.items():
        if k.startswith(enc):
            out["encoder." + k[len(enc):]] = v.float().clone()
        elif k.startswith(dec):
            out["decoder." + k[len(dec):]] = v.float().clone()
    return out


def _mlp_from_state(sd: dict, prefix: str):
    import torch.nn as nn

    idx = sorted({int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix)})
    layers = []
    for n, i in enumerate(idx):
        w = sd[f"{prefix}{i}.weight"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(sd[f"{prefix}{i}.bias"])
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class UniversalTokenActor:
    """Inference path of UniversalTokenModule for one encoder (g1) + g1_dyn decoder."""

    def __init__(self, state: dict, max_num_tokens: int = 2, fsq_levels: int = 32, num_fsq_levels: int = 32):
        import torch
        from vector_quantize_pytorch import FSQ

        self.torch = torch
        self.encoder = _mlp_from_state(state, "encoder.").eval()
        self.decoder = _mlp_from_state(state, "decoder.").eval()
        self.quantizer = FSQ(levels=[fsq_levels] * num_fsq_levels).eval()  # what the module instantiates
        self.max_num_tokens, self.token_dim = max_num_tokens, num_fsq_levels

    def __call__(self, proprio: np.ndarray, tokenizer: np.ndarray, return_all: bool = False):
        torch = self.torch
        with torch.no_grad():
            p = torch.as_tensor(proprio, dtype=torch.float32).reshape(-1, proprio.shape[-1])
            t = torch.as_tensor(tokenizer, dtype=torch.float32).reshape(p.shape[0], -1)
            latent = self.encoder(t).view(-1, self.max_num_tokens, self.token_dim)
            tokens, _ = self.quantizer(latent)
            act = self.decoder(torch.cat([tokens.reshape(p.shape[0], -1), p], dim=-1))
        if return_all:
            return act.numpy(), latent.numpy(), tokens.numpy()
        return act.numpy()


class OnnxActor:
    """Same call as UniversalTokenActor, run with onnxruntime on the exported files.

    mode "fused": policy_g1.onnx, input [command_multi_future (10x44) | motion_anchor_ori_b_mf (10x6) |
                  actor_obs (720)] -> action (22)
    mode "split": model_encoder.onnx, input [mode_id (1) | encoder_index slot (3) | tokenizer features in
                  module order (1607)] -> tokens (64); model_decoder.onnx, input [tokens (64) | actor_obs (720)]
                  -> action (22). mode_id 0 = g1 (robot motion), 1 = teleop, 2 = smpl (untrained).
    """

    # encoder input after [mode_id | encoder_index(3)]: (feature, flat size); only the g1 rows are filled here
    ENC_LAYOUT = [("command_multi_future_nonflat", 440), ("motion_anchor_ori_b", 6),
                  ("motion_anchor_ori_b_mf_nonflat", 60), ("command_multi_future_lower_body", 240),
                  ("vr_3point_local_target", 9), ("vr_3point_local_orn_target", 12),
                  ("smpl_joints_multi_future_local_nonflat", 720), ("smpl_root_ori_b_multi_future", 60),
                  ("joint_pos_multi_future_wrist_for_smpl", 60)]

    def __init__(self, onnx_dir: str, mode: str = "fused", num_future: int = 10):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        S = lambda f: ort.InferenceSession(str(Path(onnx_dir) / f), sess_options=so,  # noqa: E731
                                           providers=["CPUExecutionProvider"])
        self.mode, self.F = mode, num_future
        if mode == "fused":
            self.policy = S("policy_g1.onnx")
        else:
            self.enc, self.dec = S("model_encoder.onnx"), S("model_decoder.onnx")

    def __call__(self, proprio: np.ndarray, tokenizer: np.ndarray):
        B = proprio.shape[0]
        tok = np.asarray(tokenizer, dtype=np.float32).reshape(B, self.F, -1)     # per frame [cmd 44 | ori 6]
        cmd, ori = tok[..., :44].reshape(B, -1), tok[..., 44:].reshape(B, -1)
        prop = np.asarray(proprio, dtype=np.float32).reshape(B, -1)
        if self.mode == "fused":
            x = np.concatenate([cmd, ori, prop], axis=-1)
            return self.policy.run(None, {self.policy.get_inputs()[0].name: x})[0]
        feats = {"command_multi_future_nonflat": cmd, "motion_anchor_ori_b_mf_nonflat": ori}
        head = np.zeros((B, 4), dtype=np.float32)
        head[:, 1] = 1.0                                   # mode_id 0 (g1); one-hot slot is not read by the encoders
        x = np.concatenate([head] + [feats.get(k, np.zeros((B, n), np.float32)) for k, n in self.ENC_LAYOUT], -1)
        tokens = self.enc.run(None, {self.enc.get_inputs()[0].name: x})[0]
        return self.dec.run(None, {self.dec.get_inputs()[0].name: np.concatenate([tokens, prop], -1)})[0]


# --------------------------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------------------------
class ProprioHistory:
    """`actor_obs`: terms in PolicyCfg attribute order, each a 10-step history (oldest first).

    IsaacLab CircularBuffer semantics: the first sample after a reset fills every slot.
    """

    TERMS = ("base_ang_vel", "joint_pos", "joint_vel", "actions", "gravity_dir")

    def __init__(self):
        self.buf = None

    def reset(self):
        self.buf = None

    def append(self, **terms):
        new = [np.asarray(terms[k], dtype=np.float32) for k in self.TERMS]
        if self.buf is None:
            self.buf = [np.repeat(x[None], HIST, axis=0) for x in new]
        else:
            self.buf = [np.concatenate([b[1:], x[None]], axis=0) for b, x in zip(self.buf, new)]

    def flat(self) -> np.ndarray:
        return np.concatenate([b.reshape(-1) for b in self.buf]).astype(np.float32)


def g1_tokenizer_obs(clip: dict, step: int, robot_root_quat: np.ndarray, num_future: int = 10,
                     frame_skips: int = 5, noise=None) -> tuple[np.ndarray, dict]:
    """g1 encoder input (num_future x 50 -> 500), IsaacLab layout.

    command_multi_future = cat(joint_pos_mf.flat (F*22), joint_vel_mf.flat (F*22)).reshape(F, 44)
    motion_anchor_ori_b_mf = mat(quat_inv(robot_root) * ref_root)[..., :2] (row-major 6D), (F, 6)
    """
    T = clip["T"]
    idx = np.minimum(step + np.arange(num_future) * frame_skips, T - 1)
    jp, jv = clip["dof_pos"][idx], clip["dof_vel"][idx]
    cmd = np.concatenate([jp.reshape(-1), jv.reshape(-1)]).reshape(num_future, -1)
    ref_q = clip["body_quat"][idx, 0]
    rel = quat_mul(np.repeat(quat_conj(robot_root_quat)[None], num_future, 0), ref_q)
    ori = quat_to_mat(rel)[..., :2].reshape(num_future, 6)
    if noise is not None:
        ori = ori + noise("motion_anchor_ori_b_mf_nonflat", ori.shape)
    tok = np.concatenate([cmd, ori], axis=-1).reshape(-1).astype(np.float32)
    return tok, {"command_multi_future_nonflat": cmd, "motion_anchor_ori_b_mf_nonflat": ori}


class ObsNoise:
    """Optional additive uniform obs noise, ranges read from the run config (enable_corruption)."""

    def __init__(self, cfg, seed: int, scale: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.scale = scale
        self.ranges = {}
        for group in ("policy", "tokenizer"):
            for name, term in cfg.manager_env.observations[group].items():
                if hasattr(term, "get") and term.get("noise") is not None:
                    self.ranges[name] = (float(term.noise.n_min), float(term.noise.n_max))

    def __call__(self, name, shape):
        lo, hi = self.ranges.get(name, (0.0, 0.0))
        return self.scale * self.rng.uniform(lo, hi, size=shape) if hi > lo else np.zeros(shape)


# --------------------------------------------------------------------------------------------
# MuJoCo simulation
# --------------------------------------------------------------------------------------------
class K1MujocoSim:
    def __init__(self, plant: K1Plant, sim_dt: float = SIM_DT, kp_scale: float = 1.0, kd_scale: float = 1.0,
                 contact_timeconst: float | None = None, friction: float | None = None, payload_kg: float = 0.0,
                 strength: float = 1.0, latency_s: float = 0.0, booster_effort_caps: bool = False):
        import mujoco

        self.mj = mujoco
        self.plant = plant
        m = mujoco.MjModel.from_xml_path(str(MJCF))
        jn = [m.joint(i + 1).name for i in range(m.njnt - 1)]
        assert jn == plant.mj_joints, f"MJCF joint order != k1.py mapping:\n{jn}\n{plant.mj_joints}"
        assert [m.actuator(i).trnid[0] for i in range(m.nu)] == list(range(1, m.njnt)), "actuator order"
        m.opt.timestep = sim_dt
        # training plant: armature from k1.py (MJCF head armature is 0.002, Isaac 0.001); torque limits
        # are applied in the PD loop with the training effort limits (MJCF hip_roll forcerange 43 vs 76)
        m.dof_armature[6:] = plant.armature[plant.i2m]
        m.actuator_ctrllimited[:] = 0
        m.actuator_forcelimited[:] = 0
        m.jnt_actfrclimited[:] = 0
        # k1.xml sets geom solref timeconst 0.001 (stiff); MuJoCo clamps it to 2*dt (0.01 s at dt 5 ms).
        # At dt 1 ms the retargeted RSI poses (forearm inside the hip cylinders by up to ~4 cm) get kicked out.
        if contact_timeconst is not None:
            m.geom_solref[:, 0] = contact_timeconst
        if friction is not None:            # sliding friction of every geom (MuJoCo takes the max of a contact pair)
            m.geom_friction[:, 0] = friction
        if payload_kg:                      # point mass at the trunk COM
            b = m.body("trunk").id
            m.body_mass[b] += payload_kg
            m.body_inertia[b] += payload_kg * 0.05 ** 2
        self.strength = strength
        self.effort_mj = (np.minimum(plant.effort[plant.i2m], BOOSTER_DEPLOY_EFFORT) if booster_effort_caps
                          else plant.effort[plant.i2m])
        self.latency_steps = int(round(latency_s / sim_dt))
        self.targets = []                   # (physics step issued, target) for action latency
        self.step_count = 0
        self.m, self.d = m, mujoco.MjData(m)
        self.decimation = max(1, int(round(CONTROL_DT / sim_dt)))
        self.kp = plant.kp * kp_scale
        self.kd = plant.kd * kd_scale
        self.il_body_ids = np.array([m.body(b).id for b in plant.il_bodies])
        self.jnt_range_il = m.jnt_range[1:][plant.m2i]

    def body_ids(self, names):
        return np.array([self.m.body(n).id for n in names])

    # state accessors (IsaacLab order for joints) ------------------------------------------
    def joint_pos(self):
        return self.d.qpos[7:][self.plant.m2i].copy()

    def joint_vel(self):
        return self.d.qvel[6:][self.plant.m2i].copy()

    def root_quat(self):
        return self.d.qpos[3:7].copy()

    def root_pos(self):
        return self.d.qpos[:3].copy()

    def root_ang_vel_b(self):
        return self.d.qvel[3:6].copy()          # free-joint angular velocity is body-frame

    def set_state(self, root_pos, root_quat, jpos_il, root_lin_vel_w=None, root_ang_vel_w=None, jvel_il=None):
        d = self.d
        d.qpos[:3], d.qpos[3:7] = root_pos, root_quat / np.linalg.norm(root_quat)
        d.qpos[7:] = np.asarray(jpos_il)[self.plant.i2m]
        d.qvel[:] = 0.0
        if root_lin_vel_w is not None:
            d.qvel[:3] = root_lin_vel_w
        if root_ang_vel_w is not None:
            d.qvel[3:6] = quat_apply_inv(root_quat, root_ang_vel_w)
        if jvel_il is not None:
            d.qvel[6:] = np.asarray(jvel_il)[self.plant.i2m]
        d.qacc_warmstart[:] = 0.0
        d.ctrl[:] = 0.0
        self.targets, self.step_count = [], 0
        self.mj.mj_forward(self.m, d)

    def step_pd(self, target_il):
        """One policy step: `decimation` physics steps of explicit PD (MuJoCo actuator order)."""
        p, d = self.plant, self.d
        self.targets.append((self.step_count, np.asarray(target_il)[p.i2m]))
        kp, kd, eff = self.kp[p.i2m], self.kd[p.i2m], self.effort_mj * self.strength
        for _ in range(self.decimation):
            # latency: newest target issued at least `latency_steps` physics steps ago (else the oldest one)
            due = [t for i, t in self.targets if i <= self.step_count - self.latency_steps]
            tgt = due[-1] if due else self.targets[0][1]
            tau = self.strength * (kp * (tgt - d.qpos[7:]) - kd * d.qvel[6:])
            d.ctrl[:] = np.clip(tau, -eff, eff)
            self.mj.mj_step(self.m, d)
            self.step_count += 1
        if len(self.targets) > 64:
            self.targets = self.targets[-64:]

    def push(self, rng, scale):
        """Training-style push: add a random root velocity change (world lin, body ang), scaled."""
        self.d.qvel[:3] += rng.uniform(-1, 1, 3) * PUSH_LIN * scale
        self.d.qvel[3:6] += rng.uniform(-1, 1, 3) * PUSH_ANG * scale

    def qpos29(self):
        return self.d.qpos.copy()


# --------------------------------------------------------------------------------------------
# One clip rollout
# --------------------------------------------------------------------------------------------
class Evaluator:
    def __init__(self, actor_state: dict, cfg_dict: dict, opts: dict):
        from omegaconf import OmegaConf

        self.cfg = OmegaConf.create(cfg_dict)
        self.opts = opts
        self.plant = K1Plant()
        self.sim = K1MujocoSim(self.plant, sim_dt=opts.get("sim_dt", SIM_DT),
                               kp_scale=opts.get("kp_scale", 1.0), kd_scale=opts.get("kd_scale", 1.0),
                               contact_timeconst=opts.get("contact_timeconst"), friction=opts.get("friction"),
                               payload_kg=opts.get("payload_kg", 0.0), strength=opts.get("strength", 1.0),
                               latency_s=opts.get("latency_ms", 0.0) / 1000.0,
                               booster_effort_caps=opts.get("booster_effort_caps", False))
        bb = self.cfg.algo.config.actor.backbone
        mc = self.cfg.manager_env.commands.motion
        self.num_future = int(mc.num_future_frames)
        if opts.get("onnx"):
            self.actor = OnnxActor(opts["onnx"], "fused", self.num_future)
        else:
            self.actor = UniversalTokenActor(actor_state, max_num_tokens=int(bb.max_num_tokens),
                                             fsq_levels=int(bb.fsq_level_list), num_fsq_levels=int(bb.num_fsq_levels))
        self.parity = ({m: OnnxActor(opts["parity_onnx"], m, self.num_future) for m in ("fused", "split")}
                       if opts.get("parity_onnx") else None)
        self.frame_skips = int(float(mc.dt_future_ref_frames) // (1.0 / float(mc.motion_lib_cfg.target_fps)))
        self.body_names = list(mc.body_names)
        self.tracked_ids = self.sim.body_ids(self.body_names)
        self.ee_idx = [self.body_names.index(b) for b in EE_BODIES]
        self.foot_idx = [self.body_names.index(b) for b in FOOT_BODIES]

    def reset(self, clip, frame=0):
        jp = np.clip(clip["dof_pos"][frame], self.sim.jnt_range_il[:, 0], self.sim.jnt_range_il[:, 1])
        self.sim.set_state(clip["body_pos"][frame, 0], clip["body_quat"][frame, 0], jp,
                           clip["body_lin_vel"][frame, 0], clip["body_ang_vel"][frame, 0], clip["dof_vel"][frame])

    def observe(self, clip, step, hist: ProprioHistory, last_action, noise=None):
        s, p = self.sim, self.plant
        q = s.root_quat()
        nz = noise if noise is not None else (lambda name, shape: 0.0)
        hist.append(
            base_ang_vel=s.root_ang_vel_b() + nz("base_ang_vel", 3),
            joint_pos=s.joint_pos() - p.default_pos + nz("joint_pos", 22),
            joint_vel=s.joint_vel() + nz("joint_vel", 22),
            actions=last_action,
            gravity_dir=quat_apply_inv(q, np.array([0.0, 0.0, -1.0])) + nz("gravity_dir", 3),
        )
        tok, _ = g1_tokenizer_obs(clip, step, q, self.num_future, self.frame_skips, noise)
        return hist.flat(), tok

    def run_clip(self, clip, max_steps=500, seed=0):
        p, s = self.plant, self.sim
        noise = ObsNoise(self.cfg, seed, self.opts.get("obs_noise_scale", 1.0)) if self.opts.get("obs_noise") else None
        push_scale = self.opts.get("push_scale", 0.0)
        push_rng = np.random.default_rng(10_000 + seed)
        interval = tuple(self.opts.get("push_interval") or PUSH_INTERVAL_S)
        next_push = push_rng.uniform(*interval) if push_scale > 0 else None
        pushes = []                                      # |delta v_lin| (m/s) of every applied push
        n_steps = min(max_steps, clip["T"] - 1)
        self.reset(clip, 0)
        hist, last_a = ProprioHistory(), np.zeros(22, dtype=np.float32)
        rec = {k: [] for k in ("qpos", "ref_qpos", "body", "ref_body", "jpos", "ref_jpos", "action")}
        par = {m: 0.0 for m in (self.parity or {})}
        checks = {"anchor_pos": None, "anchor_ori_full": None, "ee_body_pos": None, "foot_pos_xyz": None}
        result, t0 = "time_out", time.time()
        for n in range(n_steps):
            prop, tok = self.observe(clip, n, hist, last_a, noise)
            raw = np.asarray(self.actor(prop[None], tok[None])[0], dtype=np.float32)
            for m, act in (self.parity or {}).items():   # raw (pre-clip) action difference, same inputs
                par[m] = max(par[m], float(np.max(np.abs(act(prop[None], tok[None])[0] - raw))))
            a = np.clip(raw, -ACTION_CLIP, ACTION_CLIP).astype(np.float32)
            if next_push is not None and n * CONTROL_DT >= next_push:
                v0 = s.d.qvel[:3].copy()
                s.push(push_rng, push_scale)
                pushes.append(float(np.linalg.norm(s.d.qvel[:3] - v0)))
                next_push += push_rng.uniform(*interval)
            s.step_pd(p.default_pos + p.action_scale * a)
            last_a = a
            k = n + 1                                   # reference frame matching the post-step state
            body = s.d.xpos[self.tracked_ids].copy()
            ref_body = clip["body_pos"][k]
            rq, rp = s.root_quat(), s.root_pos()
            ref_q, ref_p = clip["body_quat"][k, 0], clip["body_pos"][k, 0]
            rec["qpos"].append(s.qpos29())
            rec["ref_qpos"].append(np.concatenate([ref_p, ref_q, clip["dof_pos"][k][p.i2m]]))
            rec["body"].append(body)
            rec["ref_body"].append(ref_body)
            rec["jpos"].append(s.joint_pos())
            rec["ref_jpos"].append(clip["dof_pos"][k])
            rec["action"].append(a)
            # ---- termination checks (eval.yaml + training foot_pos_xyz) ----
            fired = {
                "anchor_pos": abs(ref_p[2] - rp[2]) > PAPER_HEIGHT_THR,
                "anchor_ori_full": quat_angle(ref_q, rq) ** 2 > PAPER_ORI_THR,
                "ee_body_pos": bool(np.any(np.abs(ref_body[self.ee_idx, 2] - body[self.ee_idx, 2]) > EE_HEIGHT_THR)),
            }
            dq = heading_q(quat_mul(rq, quat_conj(ref_q)))
            rel = quat_apply(dq[None], ref_body[self.foot_idx] - ref_p[None])
            rel_pos = np.concatenate([rp[:2], ref_p[2:]])[None] + rel
            fired["foot_pos_xyz"] = bool(np.any(np.linalg.norm(rel_pos - body[self.foot_idx], axis=-1) > FOOT_POS_THR))
            for name, f in fired.items():
                if f and checks[name] is None:
                    checks[name] = k
            if fired["anchor_pos"] or fired["anchor_ori_full"]:
                result = "anchor_pos" if fired["anchor_pos"] else "anchor_ori_full"
                break
        out = {k: np.asarray(v) for k, v in rec.items()}
        out.update(result=result, checks=checks, n_steps=n_steps, wall=time.time() - t0, parity=par, pushes=pushes)
        return out


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------
def _body_metrics(pred, gt):
    """SONIC/PHC compute_metrics_lite (smpl_sim) on (T,J,3) arrays -> per-clip means."""
    from smpl_sim.smpllib.smpl_eval import compute_metrics_lite

    if len(pred) < 3:
        return {k: float("nan") for k in ("mpjpe_g", "mpjpe_l", "mpjpe_pa", "accel_dist", "vel_dist")}
    m = compute_metrics_lite([pred], [gt], concatenate=False, use_tqdm=False)
    return {k: float(np.mean(v[0])) for k, v in m.items()}


def clip_metrics(r: dict, dt: float = CONTROL_DT) -> dict:
    n = len(r["jpos"])
    off = r["checks"]
    official = [v for v in off.values() if v is not None]
    off_step = min(official) if official else None
    res = {
        "steps": n, "n_target": r["n_steps"],
        "success": r["result"] == "time_out",
        "survival": n / max(1, r["n_steps"]) if r["result"] != "time_out" else 1.0,
        "result": r["result"],
        "official_success": off_step is None,
        "official_fail_step": off_step,
        "official_survival": (off_step - 1) / max(1, r["n_steps"]) if off_step else 1.0,
        "first_fire": {k: v for k, v in off.items()},
    }
    res.update(_body_metrics(r["body"], r["ref_body"]))
    # PHC-style rule: fail if the mean (global) body-position deviation ever exceeds 0.5 m
    dev = np.linalg.norm(r["body"] - r["ref_body"], axis=-1).mean(-1) if n else np.zeros(1)
    res["phc_success"] = bool(r["result"] == "time_out" and dev.max() <= 0.5)
    res["max_mean_body_dev_m"] = float(dev.max())
    if off_step is not None:     # same metrics truncated where the official im_eval would stop
        res.update({f"{k}_official": v for k, v in _body_metrics(r["body"][:off_step], r["ref_body"][:off_step]).items()})
    else:
        res.update({f"{k}_official": res[k] for k in ("mpjpe_g", "mpjpe_l", "mpjpe_pa", "accel_dist", "vel_dist")})
    q, qr, a = r["jpos"], r["ref_jpos"], r["action"]
    res["joint_err_rad"] = float(np.mean(np.abs(q - qr)))
    if n >= 4:
        acc, acc_r = np.diff(q, 2, axis=0) / dt**2, np.diff(qr, 2, axis=0) / dt**2
        jerk, jerk_r = np.diff(q, 3, axis=0) / dt**3, np.diff(qr, 3, axis=0) / dt**3
        res.update(joint_acc=float(np.mean(np.abs(acc))), ref_joint_acc=float(np.mean(np.abs(acc_r))),
                   joint_jerk=float(np.mean(np.abs(jerk))), ref_joint_jerk=float(np.mean(np.abs(jerk_r))))
    if n >= 2:
        res["action_rate"] = float(np.mean(np.abs(np.diff(a, axis=0))))
        res["action_rate_l2"] = float(np.mean(np.sum(np.diff(a, axis=0) ** 2, axis=-1)))
    for m, v in r.get("parity", {}).items():
        res[f"parity_max_abs_{m}"] = v
    if r.get("pushes") is not None:
        res["n_pushes"] = len(r["pushes"])
        res["push_dv_mean"] = float(np.mean(r["pushes"])) if r["pushes"] else 0.0
    return res


def save_rollout_npz(path: Path, r: dict, name: str):
    np.savez_compressed(path, qpos=r["qpos"].astype(np.float32), ref_qpos=r["ref_qpos"].astype(np.float32),
                        fps=1.0 / CONTROL_DT, motion=name, result=str(r["result"]),
                        actions=r["action"].astype(np.float32),
                        official_result=str(min((v, k) for k, v in r["checks"].items() if v is not None)[1]
                                            if any(v is not None for v in r["checks"].values()) else "time_out"))


# --------------------------------------------------------------------------------------------
# Workers / main
# --------------------------------------------------------------------------------------------
_W = {}


def _worker_init(actor_path, cfg_dict, opts):
    import torch

    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W["ev"] = Evaluator(torch.load(actor_path, weights_only=False), cfg_dict, opts)
    _W["opts"] = opts


def _worker_run(clip):
    ev, opts = _W["ev"], _W["opts"]
    r = ev.run_clip(clip, max_steps=opts["max_steps"], seed=clip["index"])
    met = clip_metrics(r)
    met.update(motion=clip["name"], index=clip["index"], wall=r["wall"])
    if opts.get("render_dir"):
        save_rollout_npz(Path(opts["render_dir"]) / f"{clip['index']:03d}_{clip['name'].replace('/', '_')}.npz",
                         r, clip["name"])
    return met


AGG_KEYS = ["n_pushes", "push_dv_mean", "survival", "official_survival", "mpjpe_g", "mpjpe_l", "mpjpe_pa", "accel_dist", "vel_dist",
            "mpjpe_g_official", "mpjpe_l_official", "accel_dist_official", "vel_dist_official",
            "joint_err_rad", "joint_acc", "ref_joint_acc", "joint_jerk", "ref_joint_jerk", "action_rate",
            "action_rate_l2"]


def aggregate(per_clip: list[dict]) -> dict:
    def mean(rows, k):
        v = [r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
        return float(np.mean(v)) if v else float("nan")

    succ = [r for r in per_clip if r["success"]]
    osucc = [r for r in per_clip if r["official_success"]]
    agg = {"num_clips": len(per_clip),
           "success_rate_paper": len(succ) / max(1, len(per_clip)),
           "success_rate_official": len(osucc) / max(1, len(per_clip)),
           "success_rate_phc": sum(r.get("phc_success", False) for r in per_clip) / max(1, len(per_clip)),
           "fail_reasons": {k: sum(r["result"] == k for r in per_clip) for k in ("anchor_pos", "anchor_ori_full")},
           "official_first_fire": {k: sum(r["first_fire"][k] is not None and r["first_fire"][k] == r["official_fail_step"]
                                          for r in per_clip) for k in per_clip[0]["first_fire"]} if per_clip else {}}
    agg["all"] = {k: mean(per_clip, k) for k in AGG_KEYS}
    agg["success_paper"] = {k: mean(succ, k) for k in AGG_KEYS}
    agg["success_official"] = {k: mean(osucc, k) for k in AGG_KEYS}
    return agg


def print_table(agg: dict, title: str):
    a = agg["all"]
    print(f"\n=== {title} ({agg['num_clips']} clips) ===")
    print(f"success paper-style  : {100 * agg['success_rate_paper']:.1f}%   survival {100 * a['survival']:.1f}%")
    print(f"success PHC-style    : {100 * agg['success_rate_phc']:.1f}%   (paper-style and mean body deviation <= 0.5 m)")
    print(f"success official-like: {100 * agg['success_rate_official']:.1f}%   survival {100 * a['official_survival']:.1f}%"
          f"   first check fired: {agg['official_first_fire']}")
    rows = [("MPJPE-G (mm)", "mpjpe_g"), ("MPJPE-L (mm)", "mpjpe_l"), ("PA-MPJPE (mm)", "mpjpe_pa"),
            ("accel err (mm/fr^2)", "accel_dist"), ("vel err (mm/fr)", "vel_dist"),
            ("joint err (deg)", "joint_err_rad"), ("joint |acc| (rad/s^2)", "joint_acc"),
            ("  ref joint |acc|", "ref_joint_acc"), ("joint |jerk| (rad/s^3)", "joint_jerk"),
            ("  ref joint |jerk|", "ref_joint_jerk"), ("action rate |da|", "action_rate")]
    print(f"{'metric':26s} {'all':>10s} {'succ(paper)':>12s} {'succ(official)':>15s}")
    for label, k in rows:
        f = (180 / math.pi) if k == "joint_err_rad" else 1.0
        print(f"{label:26s} {a[k] * f:10.2f} {agg['success_paper'][k] * f:12.2f} {agg['success_official'][k] * f:15.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="last.pt (config.yaml must sit next to it)")
    ap.add_argument("--motion-dir", required=True, help="motion_lib per-clip pkl directory")
    ap.add_argument("--out", required=True, help="output dir for summary.json / per_clip.csv")
    ap.add_argument("--render-dir", default=None, help="write per-clip qpos npz here (render queue format)")
    ap.add_argument("--clips", default="all", help="N (first N clips in motion-lib order) or 'all'")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--sim-dt", type=float, default=SIM_DT)
    ap.add_argument("--kp-scale", type=float, default=1.0, help="deployment PD stiffness multiplier (default 1 = training)")
    ap.add_argument("--kd-scale", type=float, default=1.0)
    ap.add_argument("--contact-timeconst", type=float, default=None,
                    help="override geom solref timeconst (k1.xml: 0.001, effectively 2*sim_dt)")
    ap.add_argument("--obs-noise", action="store_true", help="add the training/eval obs noise (Isaac eval keeps it on)")
    ap.add_argument("--obs-noise-scale", type=float, default=1.0, help="multiplier on the obs noise ranges (with --obs-noise)")
    ap.add_argument("--push-scale", type=float, default=0.0,
                    help="training-style velocity pushes every 4-6 s, ranges x this factor (1 = training range)")
    ap.add_argument("--push-interval", type=float, nargs=2, default=None, metavar=("MIN_S", "MAX_S"),
                    help="seconds between pushes (default: training 4-6 s)")
    ap.add_argument("--friction", type=float, default=None, help="sliding friction for all geoms (default: MJCF)")
    ap.add_argument("--payload-kg", type=float, default=0.0, help="extra mass on the trunk")
    ap.add_argument("--latency-ms", type=float, default=0.0, help="action latency (physics-step resolution)")
    ap.add_argument("--strength", type=float, default=1.0, help="motor strength factor on torque and effort limits")
    ap.add_argument("--booster-effort-caps", action="store_true",
                    help="clamp torques to Booster's deploy limits (legs 30/35/20/40/20 N m instead of 68/76/38/112/38)")
    ap.add_argument("--name", default=None, help="run label for the printed table")
    ap.add_argument("--onnx", default=None, help="run the policy from <dir>/policy_g1.onnx instead of the .pt")
    ap.add_argument("--parity-onnx", default=None,
                    help="<dir> with policy_g1.onnx + model_encoder.onnx + model_decoder.onnx: report max |a_onnx - a_pt|")
    args = ap.parse_args()

    import multiprocessing as mp

    import torch
    from omegaconf import OmegaConf

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.render_dir:
        Path(args.render_dir).mkdir(parents=True, exist_ok=True)
    cfg = load_run_config(args.checkpoint)
    plant = K1Plant()
    clips = load_reference_clips(cfg, plant, args.motion_dir, None if args.clips == "all" else int(args.clips))
    actor_path = out / "actor_g1.pt"
    torch.save(extract_actor_state(args.checkpoint), actor_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    opts = dict(max_steps=args.max_steps, sim_dt=args.sim_dt, kp_scale=args.kp_scale, kd_scale=args.kd_scale,
                obs_noise=args.obs_noise, render_dir=args.render_dir, contact_timeconst=args.contact_timeconst,
                onnx=args.onnx, parity_onnx=args.parity_onnx, obs_noise_scale=args.obs_noise_scale,
                push_scale=args.push_scale, push_interval=args.push_interval, friction=args.friction, payload_kg=args.payload_kg,
                latency_ms=args.latency_ms, strength=args.strength, booster_effort_caps=args.booster_effort_caps)
    print(f"[eval_k1_mujoco] {len(clips)} clips loaded in {time.time() - t0:.1f}s; {args.workers} workers", flush=True)

    ctx = mp.get_context("spawn")
    per_clip = []
    with ctx.Pool(args.workers, initializer=_worker_init, initargs=(str(actor_path), cfg_dict, opts)) as pool:
        for i, met in enumerate(pool.imap_unordered(_worker_run, clips)):
            per_clip.append(met)
            if (i + 1) % 20 == 0 or i + 1 == len(clips):
                ok = sum(r["success"] for r in per_clip)
                print(f"  {i + 1}/{len(clips)} done, paper success {ok}/{i + 1}  ({time.time() - t0:.0f}s)", flush=True)
    per_clip.sort(key=lambda r: r["index"])
    agg = aggregate(per_clip)
    summary = {"checkpoint": str(Path(args.checkpoint).resolve()), "motion_dir": str(Path(args.motion_dir).resolve()),
               "options": opts, "encoder": "g1", "aggregate": agg, "per_clip": per_clip,
               "wall_s": time.time() - t0}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: None))
    with open(out / "per_clip.csv", "w") as f:
        cols = ["index", "motion", "success", "result", "steps", "official_success", "official_fail_step",
                "mpjpe_g", "mpjpe_l", "joint_err_rad", "action_rate"]
        f.write(",".join(cols) + "\n")
        for r in per_clip:
            f.write(",".join(str(r.get(c)) for c in cols) + "\n")
    print_table(agg, args.name or out.name)
    for m in ("fused", "split"):
        v = [r[f"parity_max_abs_{m}"] for r in per_clip if f"parity_max_abs_{m}" in r]
        if v:
            agg[f"parity_max_abs_{m}"] = float(max(v))
            print(f"ONNX ({m}) vs PyTorch: max |delta a| = {max(v):.3e} over {len(v)} clips")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: None))
    print(f"\n[eval_k1_mujoco] wrote {out / 'summary.json'} ({time.time() - t0:.0f}s)")
    actor_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
