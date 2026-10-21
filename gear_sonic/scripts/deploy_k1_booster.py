#!/usr/bin/env python3
"""SONIC-K1 deployment: run policy_g1.onnx on a Booster K1 (low-level SDK) or in a MuJoCo bench, with a safety layer.

UNTESTED ON HARDWARE. Read the test ladder (deploy_k1_README.md) before using --backend booster without --dry-run.

Pipeline (50 Hz policy, identical to training and gear_sonic/scripts/eval_k1_mujoco.py):
  state  : IMU rpy + gyro, 22 joint positions/velocities (Booster serial order == MuJoCo k1.xml order)
  obs    : actor_obs 720 = 10-step history (oldest first, term-major) of
           [base_ang_vel 3 | joint_pos - default 22 | joint_vel 22 | last_action 22 | gravity_dir 3]   (IsaacLab order)
           g1 command 500 = [future joint_pos 10x22 | future joint_vel 10x22 | anchor ori 10x6], frames t, t+0.1 .. t+0.9 s
  policy : policy_g1.onnx [1, 1220] -> action [1, 22]; clip +-20; q* = default + action_scale * action
  output : per-motor (q*, kp, kd) impedance command; the motor drivers run the PD loop
Reference motion: a K1 joint-space clip (.npz with qpos [T,29] = root xyz, root quat wxyz, 22 joints in MuJoCo order,
and fps; e.g. a UMR retarget or eval_k1_mujoco --render-dir ref_qpos via --ref-key), or a .csv as accepted by
convert_k1_csv_to_motion_lib.py. Its heading is aligned to the robot's yaw when the policy starts.

Safety layer
  * soft start: current pose -> the clip's first-frame joint pose (or the default pose, --soft-start-to default) over
    --soft-start s with the stiff "prepare" gains, then hold --hold s; this mirrors training, where every episode starts
    at the reference pose. Use clips that start from standing (legs near default, root still).
  * torque limit: q* is clipped so the predicted PD torque stays within --torque-frac x effort limit
  * position limit: q* clipped to the k1.xml joint ranges minus --pos-margin rad
  * velocity guard: e-stop if any |qd| > --vel-frac x joint velocity limit
  * tilt guard: e-stop if |roll| or |pitch| > --max-tilt rad
  * watchdog: e-stop if no fresh state for --watchdog-ms, or if the policy loop overruns 5 times in a row
  * e-stop: press 'e' (or space) + Enter, or Ctrl-C -> damping mode (kp = 0, kd = damping) for 2 s, then exit
  * --dry-run: read state, compute and log actions, never send motor commands (and never switch robot mode)
Every run writes a .npz log (time, state, obs, raw action, targets, flags).

Usage
  bench (no robot):  python gear_sonic/scripts/deploy_k1_booster.py --backend mujoco --onnx <dir>/policy_g1.onnx \
                         --plant <dir>/k1_plant.json --motion clip.npz --log run.npz
  robot, dry run:    ... --backend booster --dry-run --motion clip.npz --log dry.npz
  robot, live:       ... --backend booster --motion clip.npz --log live.npz      (only after the test ladder)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import signal
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

POLICY_DT = 0.02
HIST = 10
NUM_FUTURE, FRAME_SKIP = 10, 5
ACTION_CLIP = 20.0
REPO = Path(__file__).resolve().parents[2]
MJCF = REPO / "gear_sonic/data/assets/robot_description/mjcf/k1.xml"
# joint velocity limits (rad/s) by group, from the K1 motor specs (robots/k1.py)
VEL_LIMIT = {"head": 7.85, "arm": 33.51, "leg": 14.66, "ankle": 17.59}
# Booster "prepare" (stiff stand-up) gains and the Booster deploy torque caps, MuJoCo/serial order (booster_deploy K1_CFG)
PREPARE_KP = [40, 40, 40, 50, 20, 20, 40, 50, 20, 20, 350, 350, 180, 350, 250, 250, 350, 350, 180, 350, 250, 250]
PREPARE_KD = [1.5, 1.5, 0.5, 1.5, 0.2, 0.2, 0.5, 1.5, 0.2, 0.2, 7.5, 7.5, 3, 5.5, 5, 5, 7.5, 7.5, 3, 5.5, 5, 5]
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


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


# ------------------------------------------------------------------------------------------- robot constants
class Plant:
    """Per-joint constants from k1_plant.json (shipped with the weights) + k1.xml joint ranges."""

    def __init__(self, plant_json: str, mjcf: Path = MJCF):
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


# ------------------------------------------------------------------------------------------- reference motion
class Reference:
    """K1 joint-space clip resampled to 50 Hz; joints in IsaacLab order, finite-difference velocities."""

    def __init__(self, path: str, plant: Plant, key: str = "qpos", fps: float | None = None):
        if path.endswith(".csv"):
            c = np.loadtxt(path, delimiter=",")
            q = np.concatenate([c[:, :3], c[:, [6, 3, 4, 5]], c[:, 7:]], axis=1)   # quat xyzw -> wxyz
            src_fps = fps or 30.0
        else:
            z = np.load(path, allow_pickle=True)
            q = np.asarray(z[key], dtype=np.float64)
            src_fps = fps or float(np.ravel(z["fps"])[0])
        assert q.shape[1] == 29, f"expected [T, 29] qpos, got {q.shape}"
        T = len(q) / src_fps
        n = int(T / POLICY_DT)
        t = np.arange(n) * POLICY_DT * src_fps
        i0 = np.minimum(np.floor(t).astype(int), len(q) - 1)
        i1 = np.minimum(i0 + 1, len(q) - 1)
        a = (t - i0)[:, None]
        jp_mj = q[i0, 7:] * (1 - a) + q[i1, 7:] * a
        self.root_quat = np.stack([slerp(q[i, 3:7], q[j, 3:7], float(w)) for i, j, w in zip(i0, i1, a[:, 0])])
        self.jp = jp_mj[:, plant.m2i]
        self.jv = np.gradient(self.jp, POLICY_DT, axis=0)
        self.n = n
        self.yaw_offset = 0.0

    def align_heading(self, robot_yaw: float):
        self.yaw_offset = robot_yaw - yaw_of(self.root_quat[0])

    def ref_quat(self, k):
        qz = np.array([math.cos(self.yaw_offset / 2), 0, 0, math.sin(self.yaw_offset / 2)])
        return quat_mul(qz, self.root_quat[min(k, self.n - 1)])

    def command(self, k, robot_quat):
        idx = np.minimum(k + np.arange(NUM_FUTURE) * FRAME_SKIP, self.n - 1)
        cmd = np.concatenate([self.jp[idx].reshape(-1), self.jv[idx].reshape(-1)])            # 440
        rq_inv = quat_conj(robot_quat)
        ori = np.concatenate([quat_to_mat(quat_mul(rq_inv, self.ref_quat(i)))[:, :2].reshape(-1) for i in idx])  # 60
        return cmd, ori


# ------------------------------------------------------------------------------------------- policy
class Policy:
    def __init__(self, onnx_path: str, plant: Plant):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.s = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.name = self.s.get_inputs()[0].name
        assert self.s.get_inputs()[0].shape[-1] == 1220, "expects policy_g1.onnx (input 1220)"
        self.p = plant
        self.hist = None
        self.last_action = np.zeros(22)

    def reset(self):
        self.hist, self.last_action = None, np.zeros(22)

    def proprio(self, gyro, q_il, qd_il, gravity):
        terms = [np.asarray(gyro), q_il - self.p.default, qd_il, self.last_action, gravity]
        if self.hist is None:
            self.hist = [np.repeat(x[None], HIST, 0) for x in terms]
        else:
            self.hist = [np.concatenate([h[1:], x[None]], 0) for h, x in zip(self.hist, terms)]
        return np.concatenate([h.reshape(-1) for h in self.hist])

    def act(self, prop, cmd, ori):
        x = np.concatenate([cmd, ori, prop]).astype(np.float32)[None]
        a = np.clip(self.s.run(None, {self.name: x})[0][0].astype(np.float64), -ACTION_CLIP, ACTION_CLIP)
        self.last_action = a
        return self.p.default + self.p.scale * a, a


# ------------------------------------------------------------------------------------------- backends
class State:
    def __init__(self, rpy, gyro, q_mj, qd_mj, tau_mj, stamp):
        self.rpy, self.gyro, self.q_mj, self.qd_mj, self.tau_mj, self.stamp = rpy, gyro, q_mj, qd_mj, tau_mj, stamp


class BoosterBackend:
    """Booster low-level SDK: /low_state (LowState) in, joint_ctrl (LowCmd, serial order) out, B1LocoClient for modes."""

    def __init__(self, dry_run: bool):
        import rclpy
        from booster_interface.msg import LowCmd, LowState, MotorCmd
        from booster_robotics_sdk_python import B1LocoClient, RobotMode  # type: ignore
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self.rclpy, self.LowCmd, self.RobotMode, self.dry = rclpy, LowCmd, RobotMode, dry_run
        rclpy.init()
        self.node = rclpy.create_node("sonic_k1_deploy")
        self.state = None
        self.lock = threading.Lock()
        self.node.create_subscription(LowState, "/low_state", self._on_state,
                                      QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
        self.pub = self.node.create_publisher(LowCmd, "joint_ctrl",
                                              QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))
        self.cmd = LowCmd()
        self.cmd.cmd_type = LowCmd.CMD_TYPE_SERIAL
        self.cmd.motor_cmd.extend([MotorCmd() for _ in range(22)])
        self.client = B1LocoClient()
        self.client.Init()
        self.spin = threading.Thread(target=self._spin, daemon=True)
        self.spin.start()

    def _spin(self):
        while self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def _on_state(self, m):
        st = State(np.array(m.imu_state.rpy, float), np.array(m.imu_state.gyro, float),
                   np.array([x.q for x in m.motor_state_serial], float), np.array([x.dq for x in m.motor_state_serial], float),
                   np.array([x.tau_est for x in m.motor_state_serial], float), time.monotonic())
        with self.lock:
            self.state = st

    def read(self):
        with self.lock:
            return self.state

    def enter_custom_mode(self, q_mj, kp_mj, kd_mj):
        """As booster_deploy: wait for the joint_ctrl subscriber, send a hold-current-pose command, then switch mode."""
        if self.dry:
            return
        t0 = time.monotonic()
        while self.pub.get_subscription_count() == 0:
            if time.monotonic() - t0 > 10.0:
                raise EStop("no joint_ctrl subscriber")
            time.sleep(0.5)
        self.send(q_mj, kp_mj, kd_mj)
        time.sleep(0.1)
        self.client.ChangeMode(self.RobotMode.kCustom)

    def send(self, q_mj, kp_mj, kd_mj):
        if self.dry:
            return
        for i, mc in enumerate(self.cmd.motor_cmd):
            mc.q, mc.dq, mc.tau, mc.kp, mc.kd, mc.weight = float(q_mj[i]), 0.0, 0.0, float(kp_mj[i]), float(kd_mj[i]), 0.0
        self.pub.publish(self.cmd)

    def wait(self, t_next):
        while time.monotonic() < t_next:
            time.sleep(0.0005)

    def close(self):
        self.rclpy.shutdown()


class MujocoBackend:
    """Bench backend: k1.xml in MuJoCo; the motor-driver PD loop is emulated at 1 kHz (1 ms physics, 20 substeps per policy
    step) because the stiff Booster prepare gains are unstable with a 5 ms explicit PD. IMU read from the free joint."""

    def __init__(self, plant: Plant, ref: Reference, effort_mj=None):
        import mujoco

        self.mj, self.p = mujoco, plant
        self.m = mujoco.MjModel.from_xml_path(str(MJCF))
        self.m.opt.timestep = 0.001
        self.m.dof_armature[6:] = plant.armature[plant.i2m]       # training armature (k1.xml head value differs)
        self.m.actuator_ctrllimited[:] = 0
        self.m.actuator_forcelimited[:] = 0
        self.m.jnt_actfrclimited[:] = 0
        self.d = mujoco.MjData(self.m)
        # start like the real robot: standing in the default pose (the soft start then holds it)
        self.d.qpos[:3] = [0.0, 0.0, 0.56]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qpos[7:] = plant.default[plant.i2m]
        mujoco.mj_forward(self.m, self.d)
        self.target, self.kp, self.kd = self.d.qpos[7:].copy(), np.zeros(22), np.zeros(22)
        self.effort = effort_mj if effort_mj is not None else plant.effort[plant.i2m]
        self.t = 0.0

    def read(self):
        q = self.d.qpos[3:7]
        w, x, y, z = q
        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        return State(np.array([r, pch, yaw_of(q)]), self.d.qvel[3:6].copy(), self.d.qpos[7:].copy(), self.d.qvel[6:].copy(),
                     self.d.qfrc_actuator[6:].copy(), time.monotonic())

    def enter_custom_mode(self, q_mj, kp_mj, kd_mj):
        self.send(q_mj, kp_mj, kd_mj)

    def send(self, q_mj, kp_mj, kd_mj):
        self.target, self.kp, self.kd = np.asarray(q_mj, float), np.asarray(kp_mj, float), np.asarray(kd_mj, float)

    def wait(self, t_next):
        for _ in range(int(round(POLICY_DT / self.m.opt.timestep))):
            tau = self.kp * (self.target - self.d.qpos[7:]) - self.kd * self.d.qvel[6:]
            self.d.ctrl[:] = np.clip(tau, -self.effort, self.effort)
            self.mj.mj_step(self.m, self.d)
        self.t += POLICY_DT

    def close(self):
        pass


# ------------------------------------------------------------------------------------------- safety + main loop
class EStop(Exception):
    pass


def keyboard_watch(flag: threading.Event):
    if not sys.stdin.isatty():
        return
    for line in sys.stdin:
        if line.strip().lower() in ("e", "") or line.startswith(" "):
            flag.set()
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["booster", "mujoco"], required=True)
    ap.add_argument("--onnx", required=True, help="policy_g1.onnx")
    ap.add_argument("--plant", required=True, help="k1_plant.json")
    ap.add_argument("--motion", required=True, help="K1 clip (.npz with qpos [T,29] + fps, or .csv)")
    ap.add_argument("--ref-key", default="qpos", help="npz key of the reference qpos (e.g. ref_qpos for eval rollouts)")
    ap.add_argument("--fps", type=float, default=None, help="override the clip frame rate")
    ap.add_argument("--dry-run", action="store_true", help="never send motor commands")
    ap.add_argument("--log", default="deploy_log.npz")
    ap.add_argument("--soft-start", type=float, default=3.0)
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold the start pose before the policy starts")
    ap.add_argument("--soft-start-to", choices=["clip", "default"], default="clip")
    ap.add_argument("--gain-ramp", type=float, default=0.5, help="seconds to blend prepare gains into policy gains")
    ap.add_argument("--torque-frac", type=float, default=1.0, help="x effort limit (training clips at 1.0)")
    ap.add_argument("--booster-effort-caps", action="store_true", help="use Booster's deploy torque caps (much lower on legs)")
    ap.add_argument("--pos-margin", type=float, default=0.0, help="rad inside the k1.xml range (default pose has knees at the limit)")
    ap.add_argument("--vel-frac", type=float, default=0.9)
    ap.add_argument("--max-tilt", type=float, default=0.8)
    ap.add_argument("--watchdog-ms", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None, help="stop the clip after this many seconds")
    args = ap.parse_args()

    plant = Plant(args.plant)
    ref = Reference(args.motion, plant, args.ref_key, args.fps)
    policy = Policy(args.onnx, plant)
    effort_il = plant.booster_effort if args.booster_effort_caps else plant.effort
    be = BoosterBackend(args.dry_run) if args.backend == "booster" else MujocoBackend(plant, ref, effort_il[plant.i2m])
    estop = threading.Event()
    signal.signal(signal.SIGINT, lambda *a: estop.set())
    threading.Thread(target=keyboard_watch, args=(estop,), daemon=True).start()
    log = {k: [] for k in ("t", "phase", "rpy", "gyro", "q", "qd", "tau_est", "action", "target", "flags")}
    to_mj = lambda x_il: np.asarray(x_il)[plant.i2m]  # noqa: E731

    def get_state():
        st = be.read()
        if st is None or (args.backend == "booster" and time.monotonic() - st.stamp > args.watchdog_ms / 1000):
            raise EStop("watchdog: no fresh state")
        q, qd = st.q_mj[plant.m2i], st.qd_mj[plant.m2i]
        if np.any(np.abs(qd) > args.vel_frac * plant.vmax):
            raise EStop(f"velocity limit: {plant.il[int(np.argmax(np.abs(qd) / plant.vmax))]}")
        if abs(st.rpy[0]) > args.max_tilt or abs(st.rpy[1]) > args.max_tilt:
            raise EStop(f"tilt {st.rpy[:2]}")
        return st, q, qd

    def limit(target, q, qd, kp, kd):
        flags = 0
        t = np.clip(target, plant.lo + args.pos_margin, plant.hi - args.pos_margin)
        flags |= int(np.any(t != target))
        cap = args.torque_frac * effort_il
        tau = kp * (t - q) - kd * qd
        over = np.abs(tau) > cap
        if np.any(over):
            t = np.where(over, q + (np.sign(tau) * cap + kd * qd) / np.maximum(kp, 1e-6), t)
            flags |= 2
        return t, flags

    def record(phase, st, a, tgt, flags):
        log["t"].append(time.monotonic()); log["phase"].append(phase); log["rpy"].append(st.rpy); log["gyro"].append(st.gyro)
        log["q"].append(st.q_mj); log["qd"].append(st.qd_mj); log["tau_est"].append(st.tau_mj); log["action"].append(a)
        log["target"].append(to_mj(tgt)); log["flags"].append(flags)

    reason = "clip finished"
    try:
        t0 = time.monotonic()
        while be.read() is None:
            if time.monotonic() - t0 > 3.0:
                raise EStop("no state received")
            time.sleep(0.01)
        st, q, qd = get_state()
        be.enter_custom_mode(st.q_mj, to_mj(plant.prep_kp), to_mj(plant.prep_kd))
        # 1) soft start to the default pose with the stiff prepare gains, then hold
        q_start, n_soft, n_hold = q.copy(), int(args.soft_start / POLICY_DT), int(args.hold / POLICY_DT)
        q_goal = ref.jp[0].copy() if args.soft_start_to == "clip" else plant.default.copy()
        t_next = time.monotonic()
        for k in range(n_soft + n_hold):
            if estop.is_set():
                raise EStop("e-stop")
            a = min(1.0, (k + 1) / n_soft)
            tgt = q_start + (q_goal - q_start) * (0.5 - 0.5 * math.cos(math.pi * a))
            st, q, qd = get_state()
            tgt, fl = limit(tgt, q, qd, plant.prep_kp, plant.prep_kd)
            be.send(to_mj(tgt), to_mj(plant.prep_kp), to_mj(plant.prep_kd))
            record(0, st, np.zeros(22), tgt, fl)
            t_next += POLICY_DT
            be.wait(t_next)
        # 2) policy: align the clip heading to the robot, blend gains over --gain-ramp
        ref.align_heading(float(st.rpy[2]))
        policy.reset()
        n_ramp = max(1, int(args.gain_ramp / POLICY_DT))
        n_run = ref.n - 1 if args.max_seconds is None else min(ref.n - 1, int(args.max_seconds / POLICY_DT))
        overruns = 0
        for k in range(n_run):
            if estop.is_set():
                raise EStop("e-stop")
            t_step = time.monotonic()
            st, q, qd = get_state()
            rq = rpy_to_quat(*st.rpy)
            gravity = quat_to_mat(rq).T @ np.array([0.0, 0.0, -1.0])
            prop = policy.proprio(st.gyro, q, qd, gravity)
            cmd, ori = ref.command(k, rq)
            tgt, a = policy.act(prop, cmd, ori)
            w = min(1.0, (k + 1) / n_ramp)
            kp, kd = (1 - w) * plant.prep_kp + w * plant.kp, (1 - w) * plant.prep_kd + w * plant.kd
            tgt, fl = limit(tgt, q, qd, kp, kd)
            be.send(to_mj(tgt), to_mj(kp), to_mj(kd))
            record(1, st, a, tgt, fl)
            if args.backend == "booster":
                overruns = overruns + 1 if time.monotonic() - t_step > POLICY_DT else 0
                if overruns >= 5:
                    raise EStop("policy loop overrun x5")
            t_next += POLICY_DT
            be.wait(t_next)
    except EStop as e:
        reason = f"E-STOP: {e}"
    finally:
        # damping mode: hold no position, damp joints, for 2 s
        try:
            st = be.read()
            if st is not None:
                for _ in range(int(2.0 / POLICY_DT)):
                    be.send(st.q_mj, np.zeros(22), to_mj(plant.prep_kd))
                    be.wait(time.monotonic() + POLICY_DT)
        finally:
            be.close()
            np.savez_compressed(args.log, reason=reason, dry_run=args.dry_run, backend=args.backend,
                                **{k: np.asarray(v) for k, v in log.items()})
            n_flag = int(np.sum(np.asarray(log["flags"]) > 0)) if log["flags"] else 0
            print(f"[deploy_k1] {reason}; {len(log['t'])} steps logged to {args.log}; limited steps: {n_flag}")


if __name__ == "__main__":
    main()
