"""Robot backends with one interface: the real Booster K1 (wired like booster_deploy) and a MuJoCo stand-in.

    read()                         -> State (MuJoCo/serial joint order) or None
    enter_custom(q, kp, kd)        hold q with (kp, kd), switch to custom mode, allow commands
    send(q, kp, kd)                -> False once damped (nothing is sent after damp())
    damp(reason)                   stop commanding, kp = 0 command, robot DAMPING mode. Any thread, any time, idempotent.
    advance(t_next)                wait for the next control tick (sim: step the physics)
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from sonic_k1 import POLICY_DT, quat_to_rpy

DAMP_KD = 1.0   # kd of the kp = 0 command sent right before switching to the firmware's damping mode


class State:
    __slots__ = ("rpy", "gyro", "q", "dq", "tau", "stamp")

    def __init__(self, rpy, gyro, q, dq, tau, stamp):
        self.rpy, self.gyro, self.q, self.dq, self.tau, self.stamp = rpy, gyro, q, dq, tau, stamp


class BoosterK1:
    """Booster low-level interface, as booster_deploy does it:
    ChannelFactory.Init(0, net) -> B1LocoClient for modes; ROS 2 /low_state (LowState) in, joint_ctrl (LowCmd, serial order) out."""

    name = "robot"

    def __init__(self, net: str, dry_run: bool):
        import rclpy
        from booster_interface.msg import LowCmd, LowState, MotorCmd
        import booster_robotics_sdk_python as sdk  # type: ignore
        from booster_robotics_sdk_python import B1LocoClient, ChannelFactory, RobotMode  # type: ignore
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self.dry, self.RobotMode = dry_run, RobotMode
        self.fake = "fake_booster" in (getattr(sdk, "__file__", "") or "")
        ChannelFactory.Instance().Init(0, net)
        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("sonic_k1_deploy")
        self._state = None
        self._state_lock = threading.Lock()
        self.node.create_subscription(LowState, "/low_state", self._on_state,
                                      QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
        self.pub = self.node.create_publisher(LowCmd, "joint_ctrl",
                                              QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))
        self.cmd = LowCmd()
        self.cmd.cmd_type = LowCmd.CMD_TYPE_SERIAL
        self.cmd.motor_cmd.extend([MotorCmd() for _ in range(22)])
        self.client = B1LocoClient()
        self.client.Init()
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self.node)
        self._stop = threading.Event()
        threading.Thread(target=self._spin, daemon=True).start()
        self._cmd_lock = threading.RLock()     # publish + the active flag; damp() takes it, so nothing is sent after it
        self._mode_lock = threading.Lock()     # one ChangeMode RPC at a time
        self.active = False                    # True only between enter_custom() and damp()
        self.touched = False                   # we changed the robot's mode at least once
        self.mode = "unknown (not touched)"

    def _spin(self):
        while self.rclpy.ok() and not self._stop.is_set():
            try:
                self._exec.spin_once(timeout_sec=0.05)
            except Exception:
                if self._stop.is_set():
                    return

    def _on_state(self, m):
        st = State(np.array(m.imu_state.rpy, float), np.array(m.imu_state.gyro, float),
                   np.array([x.q for x in m.motor_state_serial], float), np.array([x.dq for x in m.motor_state_serial], float),
                   np.array([x.tau_est for x in m.motor_state_serial], float), time.monotonic())
        with self._state_lock:
            self._state = st

    def read(self):
        with self._state_lock:
            return self._state

    def _publish(self, q, kp, kd):
        for i, mc in enumerate(self.cmd.motor_cmd):
            mc.q, mc.dq, mc.tau, mc.kp, mc.kd, mc.weight = float(q[i]), 0.0, 0.0, float(kp[i]), float(kd[i]), 0.0
        self.pub.publish(self.cmd)

    def _change_mode(self, mode, label):
        with self._mode_lock:
            for attempt in range(3):
                try:
                    ret = self.client.ChangeMode(mode)
                except Exception as e:  # noqa: BLE001
                    ret = e
                if ret in (0, None):
                    self.mode = label
                    return True
                print(f"\n[robot] ChangeMode({label}) attempt {attempt + 1} returned {ret}", flush=True)
                time.sleep(0.05)
            self.mode = f"{label}?? (ChangeMode failed)"
            return False

    def enter_custom(self, q, kp, kd, cancelled):
        """As booster_deploy start_custom_mode_conditionally: wait for the joint_ctrl subscriber, publish a hold-current-pose
        command with the given gains, then switch to custom mode. `cancelled()` aborts the wait (a damp request)."""
        if self.dry:
            self.mode = "custom (dry run: not switched)"
            self.active = True
            return
        t0 = time.monotonic()
        while self.pub.get_subscription_count() == 0:
            if cancelled():
                raise RuntimeError("cancelled while waiting for the joint_ctrl subscriber")
            if time.monotonic() - t0 > 10.0:
                raise RuntimeError("no joint_ctrl subscriber after 10 s (is the robot's low-level service up, firmware >= 1.4?)")
            time.sleep(0.2)
        with self._cmd_lock:
            if cancelled():
                raise RuntimeError("cancelled")
            self._publish(q, kp, kd)
            self.active = True
            self.touched = True
        time.sleep(0.1)
        with self._cmd_lock:
            if not self.active:      # damped during the sleep: do not switch to custom on top of it
                raise RuntimeError("cancelled")
            self._publish(q, kp, kd)
        if not self._change_mode(self.RobotMode.kCustom, "CUSTOM"):
            raise RuntimeError("could not switch to custom mode")

    def send(self, q, kp, kd):
        with self._cmd_lock:
            if not self.active:
                return False
            if not self.dry:
                self._publish(q, kp, kd)
            return True

    def damp(self, reason):
        with self._cmd_lock:
            was_active = self.active
            self.active = False
            if self.dry:
                self.mode = "DAMPING (dry run: not switched)"
                return
            st = self.read()
            if st is not None:
                self._publish(st.q, np.zeros(22), np.full(22, DAMP_KD))   # limp + damped right now, before the RPC
        self.touched = True
        ok = self._change_mode(self.RobotMode.kDamping, "DAMPING")
        if not ok:
            print("\n[robot] !!! ChangeMode(kDamping) FAILED -- use the Booster remote / power off", flush=True)
        return was_active

    def walk(self):
        """Hand the robot to the firmware's walking controller (it stands in place): stop commanding, kWalking."""
        with self._cmd_lock:
            self.active = False
            if self.dry:
                self.mode = "WALKING (dry run: not switched)"
                return True
        self.touched = True
        return self._change_mode(self.RobotMode.kWalking, "WALKING")

    def advance(self, t_next):
        while True:
            dt = t_next - time.monotonic()
            if dt <= 0:
                return
            time.sleep(min(dt, 0.002) if dt > 0.003 else 0.0002)

    def close(self):
        self._stop.set()
        try:
            self.node.destroy_node()
            self.rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


class SimK1:
    """MuJoCo stand-in with the same interface. The motor-driver PD loop runs at 1 kHz (k1.xml, 1 ms physics), the
    torque is capped like the real drivers, DAMPING = kp 0 + joint damping, and an optional belt (a rope from a fixed
    point overhead) catches the robot once it sags more than --belt-slack, like the test belts."""

    name = "sim"

    def __init__(self, plant, mjcf, effort_mj, belt_slack=0.08, viewer=False, realtime=True, on_key=None):
        import mujoco

        self.mj, self.p = mujoco, plant
        self.m = mujoco.MjModel.from_xml_path(str(mjcf))
        self.m.opt.timestep = 0.001
        self.m.dof_armature[6:] = plant.armature[plant.i2m]
        self.m.actuator_ctrllimited[:] = 0
        self.m.actuator_forcelimited[:] = 0
        self.m.jnt_actfrclimited[:] = 0
        self.d = mujoco.MjData(self.m)
        self.effort = np.asarray(effort_mj, float)
        self.realtime = realtime
        self.trunk = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.belt = belt_slack is not None
        # standing in the default pose; the belt is a rope from a fixed point 1 m above an attachment on the back, right
        # above the robot's centre of mass, with --belt-slack of play: it hangs upright when limp, like the test rig
        self.d.qpos[:3] = [0.0, 0.0, 0.57]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qpos[7:] = plant.default[plant.i2m]
        self.qpos0 = self.d.qpos.copy()
        mujoco.mj_forward(self.m, self.d)
        com = self.d.subtree_com[self.trunk] - self.d.xpos[self.trunk]
        self.attach = np.array([com[0], com[1], 0.20])
        self.anchor = self._attach_point() + np.array([0.0, 0.0, 1.0])
        self.rope = 1.0 + (belt_slack or 0.0)
        self.lock = threading.Lock()
        self.active, self.touched = False, False
        self._idle_mode()
        self.t_wall = time.monotonic()
        self.viewer = None
        if viewer:
            import mujoco.viewer

            keymap = {32: "space"}
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d, key_callback=lambda c: on_key and on_key(keymap.get(c, chr(c).lower() if 32 < c < 127 else "")))
        self._physics(2.0)                              # settle

    def _idle_mode(self):
        """With a belt: hanging limp (DAMPING). Without one: standing, held stiff in the prepare pose like PREPARE mode."""
        with self.lock:
            if self.belt:
                self.target, self.kp, self.kd, self.mode = self.d.qpos[7:].copy(), np.zeros(22), np.full(22, DAMP_KD), "DAMPING"
            else:
                self.target = self.p.to_mj(self.p.stand)
                self.kp, self.kd, self.mode = self.p.to_mj(self.p.prep_kp), self.p.to_mj(self.p.prep_kd), "PREPARE (sim)"

    def reset(self):
        """Back to the start: under the belt and limp, or standing in PREPARE without a belt (the operator's job between
        real runs)."""
        self.active = False
        self.d.qpos[:], self.d.qvel[:] = self.qpos0, 0.0
        self._idle_mode()
        self._physics(2.0)

    def _attach_point(self):
        return self.d.xpos[self.trunk] + self.d.xmat[self.trunk].reshape(3, 3) @ self.attach

    def _physics(self, seconds):
        if self.viewer is not None:
            with self.viewer.lock():
                self._step(seconds)
        else:
            self._step(seconds)

    def _step(self, seconds):
        for _ in range(int(round(seconds / self.m.opt.timestep))):
            with self.lock:
                tau = self.kp * (self.target - self.d.qpos[7:]) - self.kd * self.d.qvel[6:]
            self.d.ctrl[:] = np.clip(tau, -self.effort, self.effort)
            self.d.qfrc_applied[:] = 0
            if self.belt:
                p = self._attach_point()
                r = self.anchor - p
                dist = float(np.linalg.norm(r))
                if dist > self.rope:
                    u = r / dist
                    v = np.zeros(6)
                    self.mj.mj_objectVelocity(self.m, self.d, self.mj.mjtObj.mjOBJ_BODY, self.trunk, v, 0)
                    vp = v[3:] + np.cross(v[:3], p - self.d.xpos[self.trunk])
                    f = max(0.0, 4000.0 * (dist - self.rope) + 150.0 * float(vp @ -u)) * u
                    self.mj.mj_applyFT(self.m, self.d, f, np.zeros(3), p, self.trunk, self.d.qfrc_applied)
            self.mj.mj_step(self.m, self.d)

    def read(self):
        q = self.d.qpos[3:7].copy()
        return State(quat_to_rpy(q), self.d.qvel[3:6].copy(), self.d.qpos[7:].copy(), self.d.qvel[6:].copy(),
                     self.d.qfrc_actuator[6:].copy(), time.monotonic())

    def enter_custom(self, q, kp, kd, cancelled):
        with self.lock:
            if cancelled():
                raise RuntimeError("cancelled")
            self.target, self.kp, self.kd = np.asarray(q, float), np.asarray(kp, float), np.asarray(kd, float)
            self.active, self.touched, self.mode = True, True, "CUSTOM"

    def send(self, q, kp, kd):
        with self.lock:
            if not self.active:
                return False
            self.target, self.kp, self.kd = np.asarray(q, float), np.asarray(kp, float), np.asarray(kd, float)
            return True

    def damp(self, reason):
        with self.lock:
            was_active = self.active
            self.active = False
            self.kp, self.kd = np.zeros(22), np.full(22, DAMP_KD)
            self.mode = "DAMPING"
        return was_active

    def walk(self):
        """Stand-in for the firmware walking controller: hold the stand pose with the stiff prepare gains."""
        with self.lock:
            self.active = False
            self.target = self.p.to_mj(self.p.stand)
            self.kp, self.kd = self.p.to_mj(self.p.prep_kp), self.p.to_mj(self.p.prep_kd)
            self.mode = "WALKING (sim: holds the stand pose)"
        return True

    def advance(self, t_next):
        self._physics(POLICY_DT)
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise SystemExit("viewer closed")
            self.viewer.sync()
        if self.realtime:
            self.t_wall = max(self.t_wall + POLICY_DT, time.monotonic() - 0.1)
            dt = self.t_wall - time.monotonic()
            if dt > 0:
                time.sleep(dt)

    def trunk_height(self):
        return float(self.d.xpos[self.trunk][2])

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

