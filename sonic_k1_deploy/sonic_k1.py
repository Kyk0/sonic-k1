"""SONIC-K1 policy core: robot constants, reference motion, observation and policy_g1.onnx. No robot I/O here.

Identical to training and gear_sonic/scripts/eval_k1_mujoco.py:
  obs    : actor_obs 720 = 10-step history (oldest first, term-major) of
           [base_ang_vel 3 | joint_pos - default 22 | joint_vel 22 | last_action 22 | gravity_dir 3]   (IsaacLab order)
           g1 command 500 = [future joint_pos 10x22 | future joint_vel 10x22 | anchor ori 10x6], frames t, t+0.1 .. t+0.9 s
  policy : policy_g1.onnx [1, 1220] -> action [1, 22]; clip +-20; q* = default + action_scale * action, 50 Hz
Booster serial motor order == MuJoCo k1.xml joint order; the maps to IsaacLab order come from k1_plant.json.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

POLICY_DT = 0.02
HIST = 10
NUM_FUTURE, FRAME_SKIP = 10, 5
ACTION_CLIP = 20.0
# joint velocity limits (rad/s) by group, from the K1 motor specs
VEL_LIMIT = {"head": 7.85, "arm": 33.51, "leg": 14.66, "ankle": 17.59}
# Booster "prepare" (stiff stand-up) gains and the Booster deploy torque caps, MuJoCo/serial order (booster_deploy K1_CFG)
PREPARE_KP = [40, 40, 40, 50, 20, 20, 40, 50, 20, 20, 350, 350, 180, 350, 250, 250, 350, 350, 180, 350, 250, 250]
PREPARE_KD = [1.5, 1.5, 0.5, 1.5, 0.2, 0.2, 0.5, 1.5, 0.2, 0.2, 7.5, 7.5, 3, 5.5, 5, 5, 7.5, 7.5, 3, 5.5, 5, 5]
# Booster's prepare pose: the training default with knees 0.105 / ankles -0.10 instead of on the knee limit
PREPARE_POS = [0, 0, 0, -1.3, 0, 0, 0, 1.3, 0, 0, 0, 0, 0, 0.105, -0.10, 0, 0, 0, 0, 0.105, -0.10, 0]
BOOSTER_DEPLOY_EFFORT = [6, 6, 14, 14, 14, 14, 14, 14, 14, 14, 30, 35, 20, 40, 20, 20, 30, 35, 20, 40, 20, 20]


# ------------------------------------------------------------------------------------------- math (wxyz quaternions)
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rpy_to_quat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def quat_to_rpy(q):
    w, x, y, z = q
    r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    p = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return np.array([r, p, yaw_of(q)])


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


# ------------------------------------------------------------------------------------------- files
HERE = Path(__file__).resolve().parent


def resolve(mjcf=None, onnx=None, plant=None):
    """Explicit paths first, then the kit layout (policy/, assets/), then the repo's k1.xml."""
    def first(*cands):
        return next((Path(c) for c in cands if c and Path(c).exists()), None)
    return (first(mjcf, HERE / "assets/k1.xml", HERE.parent / "gear_sonic/data/assets/robot_description/mjcf/k1.xml"),
            first(onnx, HERE / "policy/policy_g1.onnx"), first(plant, HERE / "policy/k1_plant.json"))


# ------------------------------------------------------------------------------------------- robot constants
class Plant:
    """Per-joint constants from k1_plant.json (shipped with the weights) + k1.xml joint ranges."""

    def __init__(self, plant_json, mjcf):
        P = json.load(open(plant_json))
        self.il = P["joint_order_isaaclab"]
        self.mj = P["joint_order_mujoco"]
        self.i2m = np.array(P["isaaclab_to_mujoco_dof"])      # x_mj = x_il[i2m]
        self.m2i = np.array(P["mujoco_to_isaaclab_dof"])      # x_il = x_mj[m2i]
        pj = {r["joint"]: r for r in P["per_joint"]}
        g = lambda k: np.array([pj[j][k] for j in self.il], dtype=np.float64)  # noqa: E731
        self.kp, self.kd, self.effort, self.scale, self.default = g("kp"), g("kd"), g("effort"), g("action_scale"), g("default")
        self.armature = g("armature")
        rng = {}
        for j in ET.parse(mjcf).getroot().iter("joint"):
            if j.get("range"):
                rng[j.get("name")] = [float(x) for x in j.get("range").split()]
        self.lo = np.array([rng[j][0] for j in self.il])
        self.hi = np.array([rng[j][1] for j in self.il])
        grp = lambda j: "head" if "head" in j else "arm" if ("shoulder" in j or "elbow" in j) else "ankle" if "ankle" in j else "leg"  # noqa: E731
        self.vmax = np.array([VEL_LIMIT[grp(j)] for j in self.il])
        self.prep_kp = np.array(PREPARE_KP, dtype=np.float64)[self.m2i]
        self.prep_kd = np.array(PREPARE_KD, dtype=np.float64)[self.m2i]
        self.booster_effort = np.array(BOOSTER_DEPLOY_EFFORT, dtype=np.float64)[self.m2i]
        self.stand = np.array(PREPARE_POS, dtype=np.float64)[self.m2i]
        self.legs = np.array([i for i, j in enumerate(self.il) if any(k in j for k in ("hip", "knee", "ankle"))])

    def to_mj(self, x_il):
        return np.asarray(x_il)[self.i2m]

    def to_il(self, x_mj):
        return np.asarray(x_mj)[self.m2i]


# ------------------------------------------------------------------------------------------- reference motion
def load_qpos(path, key=None):
    """K1 clip -> (qpos [T, 29] = root xyz, root quat wxyz, 22 joints in MuJoCo order; fps).
    .npz: key `qpos` (UMR retarget, make_motion.py) or `ref_qpos` (eval_k1_mujoco --render-dir) + `fps`.
    .csv: root xyz, root quat xyzw, 22 joints per row, 30 fps unless named in a `<name>.fps` sidecar."""
    path = str(path)
    if path.endswith(".csv"):
        c = np.loadtxt(path, delimiter=",")
        q = np.concatenate([c[:, :3], c[:, [6, 3, 4, 5]], c[:, 7:]], axis=1)   # quat xyzw -> wxyz
        side = Path(path).with_suffix(".fps")
        return q, float(side.read_text().strip()) if side.exists() else 30.0
    z = np.load(path, allow_pickle=True)
    for k in [key] if key else ["qpos", "ref_qpos"]:
        if k in z.files:
            q = np.asarray(z[k], dtype=np.float64)
            break
    else:
        raise KeyError(f"{path}: no qpos / ref_qpos array (keys: {z.files})")
    if q.ndim != 2 or q.shape[1] != 29:
        raise ValueError(f"{path}: expected [T, 29] qpos, got {q.shape}")
    return q, float(np.ravel(z["fps"])[0]) if "fps" in z.files else 50.0


class Reference:
    """K1 joint-space clip resampled to 50 Hz; joints in IsaacLab order, finite-difference velocities."""

    def __init__(self, path, plant: Plant):
        q, src_fps = load_qpos(path)
        n = max(2, int(len(q) / src_fps / POLICY_DT))
        t = np.arange(n) * POLICY_DT * src_fps
        i0 = np.minimum(np.floor(t).astype(int), len(q) - 1)
        i1 = np.minimum(i0 + 1, len(q) - 1)
        a = (t - i0)[:, None]
        jp_mj = q[i0, 7:] * (1 - a) + q[i1, 7:] * a
        self.root_quat = np.stack([slerp(q[i, 3:7], q[j, 3:7], float(w)) for i, j, w in zip(i0, i1, a[:, 0])])
        self.jp = jp_mj[:, plant.m2i]
        self.jv = np.gradient(self.jp, POLICY_DT, axis=0)
        self.n = n
        self.seconds = n * POLICY_DT
        self.yaw_offset = 0.0
        # how far the first frame's legs are from standing still in the default pose (clips should start from standing;
        # arms may differ, the soft start moves them)
        self.start_dev = float(np.max(np.abs(self.jp[0] - plant.default)[plant.legs]))
        self.start_speed = float(np.max(np.abs(self.jv[:5][:, plant.legs])))
        self.start_tilt = float(np.max(np.abs(quat_to_rpy(self.root_quat[0])[:2])))
        self.peak_speed = float(np.max(np.abs(self.jv)))

    def align_heading(self, robot_yaw: float):
        self.yaw_offset = robot_yaw - yaw_of(self.root_quat[0])

    def ref_quat(self, k):
        return quat_mul(yaw_quat(self.yaw_offset), self.root_quat[min(k, self.n - 1)])

    def command(self, k, robot_quat):
        idx = np.minimum(k + np.arange(NUM_FUTURE) * FRAME_SKIP, self.n - 1)
        cmd = np.concatenate([self.jp[idx].reshape(-1), self.jv[idx].reshape(-1)])            # 440
        rq_inv = quat_conj(robot_quat)
        ori = np.concatenate([quat_to_mat(quat_mul(rq_inv, self.ref_quat(i)))[:, :2].reshape(-1) for i in idx])  # 60
        return cmd, ori


# ------------------------------------------------------------------------------------------- policy
class Policy:
    def __init__(self, onnx_path, plant: Plant, threads: int = 2):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.s = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.name = self.s.get_inputs()[0].name
        assert self.s.get_inputs()[0].shape[-1] == 1220, "expects policy_g1.onnx (input 1220)"
        self.p = plant
        self.reset()

    def reset(self):
        self.hist, self.last_action = None, np.zeros(22)

    def step(self, ref: Reference, k, rpy, gyro, q_il, qd_il):
        """One 50 Hz step -> (joint targets q* in IsaacLab order, raw action)."""
        rq = rpy_to_quat(*rpy)
        gravity = quat_to_mat(rq).T @ np.array([0.0, 0.0, -1.0])
        terms = [np.asarray(gyro, dtype=np.float64), q_il - self.p.default, qd_il, self.last_action, gravity]
        if self.hist is None:
            self.hist = [np.repeat(x[None], HIST, 0) for x in terms]
        else:
            self.hist = [np.concatenate([h[1:], x[None]], 0) for h, x in zip(self.hist, terms)]
        prop = np.concatenate([h.reshape(-1) for h in self.hist])
        cmd, ori = ref.command(k, rq)
        x = np.concatenate([cmd, ori, prop]).astype(np.float32)[None]
        a = np.clip(self.s.run(None, {self.name: x})[0][0].astype(np.float64), -ACTION_CLIP, ACTION_CLIP)
        self.last_action = a
        return self.p.default + self.p.scale * a, a
