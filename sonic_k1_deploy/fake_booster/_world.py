"""The fake robot behind the fake SDK: SimK1 physics in its own real-time thread, /low_state published at 500 Hz,
joint_ctrl applied only in custom mode, firmware damping otherwise. Every mode change is logged to fake_booster.log."""
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot import DAMP_KD, SimK1  # noqa: E402
from sonic_k1 import Plant, resolve  # noqa: E402

_world = None
_lock = threading.Lock()


def world():
    global _world
    with _lock:
        if _world is None:
            _world = World()
    return _world


class World:
    def __init__(self):
        mjcf, _, plant_json = resolve()
        self.P = Plant(plant_json, mjcf)
        self.sim = SimK1(self.P, mjcf, self.P.to_mj(self.P.booster_effort), belt_slack=None if os.environ.get("FAKE_BELT") == "0" else 0.08, realtime=False)
        self.mode = "DAMPING"
        self.cmd = None
        self.subs = []
        self.latest = None
        self.sim_t, self.t0 = 0.0, time.monotonic()
        self.log = open(os.environ.get("FAKE_BOOSTER_LOG", "fake_booster.log"), "a")
        self.event("boot, mode DAMPING")
        threading.Thread(target=self._run, daemon=True).start()

    def event(self, msg):
        self.log.write(f"{time.time():.3f} {msg}\n")
        self.log.flush()

    def change_mode(self, label):
        last = None if self.cmd is None else f"last joint_ctrl kp max {max(self.cmd[1]):.1f}"
        wall = time.monotonic() - self.t0
        last = f"{last}; physics/wall time {self.sim_t / max(wall, 1e-9):.3f}"
        self.mode = label
        if label in ("WALKING", "PREPARE"):
            self.sim.walk()
        elif label != "CUSTOM":
            self.sim.damp(label)
        else:
            self.sim.active = True
            if self.cmd is not None:
                self.sim.send(*self.cmd)
        self.event(f"ChangeMode {label} ({last})")

    def joint_ctrl(self, msg):
        mc = msg.motor_cmd
        self.cmd = (np.array([m.q for m in mc]), np.array([m.kp for m in mc]), np.array([m.kd for m in mc]))
        if self.mode == "CUSTOM":
            self.sim.send(*self.cmd)

    def _run(self):
        t = self.t0 = time.monotonic()
        while True:
            self.sim._physics(0.002)
            self.sim_t += 0.002
            st = self.sim.read()
            self.latest = SimpleNamespace(
                imu_state=SimpleNamespace(rpy=list(st.rpy), gyro=list(st.gyro)),
                motor_state_serial=[SimpleNamespace(q=st.q[i], dq=st.dq[i], tau_est=st.tau[i]) for i in range(22)])
            t += 0.002
            dt = t - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            elif dt < -0.5:
                t = time.monotonic()
