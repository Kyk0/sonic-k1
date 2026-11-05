#!/usr/bin/env python3
"""SONIC-K1 on the Booster K1: pick a motion from motions/, run it, robot goes to DAMPING when it ends.

    python3 app.py --sim               rehearse in MuJoCo (viewer, belt), same menu and keys
    python3 app.py                     the robot (on its computer; --net <ip> when running from a laptop on its network)
    python3 app.py --dry-run           the robot, but nothing is sent and the mode is never changed

Keys (no Enter needed, except after a motion number):
    SPACE      DAMPING, at any moment -- also any joystick button, and Ctrl+C
    <n> Enter  pick motion n          Enter alone = the last motion again
    g          GO (after picking)     anything else = back to the menu
    r          rescan motions/        q = quit (damps first if this app moved the robot)

Every run: hold current pose -> custom mode -> soft start to the clip's first frame (stiff prepare gains) -> hold ->
policy plays the clip -> DAMPING. Any guard (tilt, joint speed, stale state, slow loop) also ends in DAMPING.
Runs are logged to runs/<time>_<motion>/ (log.npz + run.json).
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import queue
import select
import signal
import sys
import termios
import threading
import time
import traceback
import tty
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sonic_k1 import POLICY_DT, Plant, Policy, Reference, quat_to_mat, resolve, rpy_to_quat  # noqa: E402

MOTION_EXT = (".npz", ".csv")


class Abort(Exception):
    pass


# ------------------------------------------------------------------------------------------- input
class Keyboard:
    """The only stdin reader (two readers steal each other's keys). cbreak mode, so keys arrive without Enter and Ctrl+C
    still raises SIGINT. SPACE calls on_damp right here in this thread; every other key goes to the queue."""

    def __init__(self, on_damp):
        self.q = queue.Queue()
        self.on_damp = on_damp
        self.lost = False
        self.tty = sys.stdin.isatty()
        self.old = termios.tcgetattr(sys.stdin.fileno()) if self.tty else None
        if self.tty:
            atexit.register(self.restore)
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        fd = sys.stdin.fileno()
        tty.setcbreak(fd)
        while True:
            r, _, _ = select.select([fd], [], [], 0.1)
            if not r:
                continue
            raw = os.read(fd, 1)
            if not raw:                        # terminal gone (ssh dropped): nobody can press SPACE any more
                self.lost = True
                self.on_damp("terminal lost")
                return
            ch = raw.decode(errors="ignore")
            if ch == " ":
                self.on_damp("SPACE")
            elif ch:
                self.q.put(ch)

    def restore(self):
        if self.old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old)

    def get(self, timeout):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


def joystick_thread(on_damp, wanted):
    """Any button on a gamepad (evdev) = DAMPING."""
    try:
        import evdev
    except ImportError:
        print("[joystick] python evdev not installed: keyboard only")
        return
    dev = None
    for path in evdev.list_devices():
        d = evdev.InputDevice(path)
        caps = d.capabilities()
        if evdev.ecodes.EV_KEY in caps and evdev.ecodes.EV_ABS in caps and (wanted in ("auto", "") or wanted in d.name or wanted == path):
            dev = d
            break
    if dev is None:
        print("[joystick] none found: keyboard only")
        return
    print(f"[joystick] {dev.name} ({dev.path}): ANY BUTTON = DAMPING")

    def run():
        try:
            for ev in dev.read_loop():
                if ev.type == evdev.ecodes.EV_KEY and ev.value == 1:
                    on_damp(f"joystick button {ev.code}")
        except OSError as e:
            print(f"\n[joystick] !!! LOST ({e}) -- keyboard SPACE / Ctrl+C still damp", flush=True)

    threading.Thread(target=run, daemon=True).start()


# ------------------------------------------------------------------------------------------- app
class App:
    def __init__(self, a):
        self.a = a
        self.motions_dir = Path(a.motions)
        self.runs_dir = Path(a.runs)
        if a.onnx and not Path(a.onnx).exists():
            sys.exit(f"--onnx {a.onnx}: no such file")
        mjcf, onnx, plant_json = resolve(a.mjcf, a.onnx, a.plant)
        missing = [n for n, p in (("k1.xml (--mjcf)", mjcf), ("policy_g1.onnx (--onnx)", onnx), ("k1_plant.json (--plant)", plant_json)) if p is None]
        if missing:
            sys.exit(f"missing: {', '.join(missing)} -- see README.md (make_kit.sh puts them in policy/ and assets/)")
        self.plant = Plant(plant_json, mjcf)
        self.policy = Policy(onnx, self.plant, a.threads)
        self.policy_file = str(Path(onnx).resolve())
        self.own_policies = {}                 # motions/<name>.onnx next to <name>.npz: that menu entry runs with it
        print(f"policy: {self.policy_file}", flush=True)
        self.effort = self.plant.effort if a.full_torque else self.plant.booster_effort
        self.damped = threading.Event()        # set by damp(); the control loop aborts on it
        self.damp_reason = ""
        self.running = False
        self.ticking = False                   # inside the 50 Hz loops (the stall monitor watches only these)
        self.beat = time.monotonic()
        self.quit = False
        self.last_sigint = 0.0
        self.last_motion = None
        self.end_mode = a.end                  # what a clean finish hands over to: "damp" or "walk"
        self.kd_scale = a.kd_scale             # policy-phase damping multiplier (d key); more kd stops the delay-driven leg shake
        self.info_cache = {}
        self.kb = Keyboard(self.damp)
        if a.sim:
            from robot import SimK1

            self.be = SimK1(self.plant, mjcf, self.plant.to_mj(self.effort), None if a.no_belt else a.belt_slack,
                            viewer=not a.no_viewer, realtime=not a.fast, on_key=self._sim_key)
        else:
            from robot import BoosterK1

            self.be = BoosterK1(a.net, a.dry_run)
        # the SDK's channel init can install its own SIGINT handler: install ours after it, from the main thread
        signal.signal(signal.SIGINT, self._sigint)
        signal.signal(signal.SIGTERM, lambda *_: (self.damp("SIGTERM"), setattr(self, "quit", True)))
        signal.signal(signal.SIGHUP, lambda *_: (self.damp("ssh / terminal closed"), setattr(self, "quit", True)))
        atexit.register(self._atexit)
        if a.joystick != "off":
            joystick_thread(self.damp, a.joystick)
        threading.Thread(target=self._stall_monitor, daemon=True).start()

    # ---------------------------------------------------------------- damping, from anywhere
    def damp(self, reason, quiet=False):
        first = not self.damped.is_set()
        self.damp_reason = reason if first else self.damp_reason
        self.damped.set()
        self.be.damp(reason)
        if first or not quiet:
            if self.a.dry_run:
                print(f"\n\033[1;33m*** DAMPING ({reason}) -- DRY RUN: nothing sent, the robot is unchanged ***\033[0m", flush=True)
            else:
                print(f"\n\033[1;31m*** DAMPING ({reason}) -> {self.be.mode} ***\033[0m", flush=True)
        return first

    def _sim_key(self, key):
        if key == "space":
            self.damp("SPACE (viewer)")

    def _sigint(self, *_):
        now = time.monotonic()
        self.damp("Ctrl+C")
        if not self.running and now - self.last_sigint < 3.0:
            self.quit = True
        elif not self.running:
            print("Ctrl+C again within 3 s to quit.", flush=True)
        self.last_sigint = now

    def _atexit(self):
        if self.be.active:
            self.damp("exit")

    def _stall_monitor(self):
        """The control loop itself cannot notice that it hangs (a stuck RPC, a stuck inference): damp from outside."""
        while True:
            time.sleep(0.05)
            if self.ticking and not self.damped.is_set() and time.monotonic() - self.beat > self.a.stall_ms / 1000:
                self.damp(f"control loop stalled > {self.a.stall_ms:.0f} ms")

    def policy_for(self, path):
        side = path.with_suffix(".onnx")
        if not side.exists():
            return self.policy, self.policy_file
        key = (str(side), side.stat().st_mtime)
        if key not in self.own_policies:
            self.own_policies[key] = Policy(side, self.plant, self.a.threads)
        return self.own_policies[key], str(side.resolve())

    # ---------------------------------------------------------------- menu
    def motions(self):
        files = sorted(p for p in self.motions_dir.rglob("*") if p.suffix in MOTION_EXT and not p.name.startswith("."))
        return files

    def info(self, p):
        key = (str(p), p.stat().st_mtime)
        if key not in self.info_cache:
            try:
                r = Reference(p, self.plant)
                warn = []
                if r.start_dev > self.a.max_start_dev:
                    warn.append(f"starts {r.start_dev:.2f} rad from standing")
                if r.start_speed > 1.0:
                    warn.append("starts moving")
                if r.peak_speed > 25:
                    warn.append(f"joint speed {r.peak_speed:.0f} rad/s")
                self.info_cache[key] = (f"{r.seconds:5.1f} s", "; ".join(warn))
            except Exception as e:  # noqa: BLE001
                self.info_cache[key] = ("  ???", f"unreadable: {e}")
        return self.info_cache[key]

    def show_menu(self, files):
        st = self.be.read()
        if st is None:
            pose = "no state yet"
        else:
            pose = f"roll {math.degrees(st.rpy[0]):+.0f}  pitch {math.degrees(st.rpy[1]):+.0f} deg"
        tag = "SIM" if self.a.sim else ("FAKE ROBOT (fake_booster)" if self.be.fake else "ROBOT") + (" DRY RUN" if self.a.dry_run else "")
        print(f"\n\033[1m== SONIC-K1  {tag}  | mode {self.be.mode} | {pose} ==\033[0m")
        if not files:
            print(f"  (no motions in {self.motions_dir}/ -- add .npz/.csv, see motions/README.md, then press r)")
        for i, p in enumerate(files, 1):
            dur, warn = self.info(p)
            name = str(p.relative_to(self.motions_dir)) + (" [own policy]" if p.with_suffix(".onnx").exists() else "")
            mark = " <" if p == self.last_motion else ""
            print(f"  {i:3d}  {name:<56s} {dur}{'  ! ' + warn if warn else ''}{mark}")
        print(f"  after a clip: {'WALKING (Booster controller)' if self.end_mode == 'walk' else 'DAMPING'}   (w = switch)")
        print(f"  policy damping: kd x{self.kd_scale:.1f}   (d = 1.0 / 1.5 / 2.0)")
        print("  SPACE = DAMP   <n> Enter = pick   Enter = last   r = rescan   q = quit" + ("   z = sim: reset the robot" if self.a.sim else ""))

    def idle_key(self, timeout=0.02):
        """Wait for one key while keeping the backend ticking (the sim needs physics, the robot nothing)."""
        t_end = time.monotonic() + timeout
        if self.kb.lost:
            self.quit = True
        while not self.quit:
            k = self.kb.get(0.0 if self.a.sim else min(0.02, max(0.0, t_end - time.monotonic())))
            if self.a.sim:
                self.be.advance(time.monotonic() + POLICY_DT)
            if k is not None or time.monotonic() >= t_end:
                return k
        return None

    def read_line(self, prompt):
        sys.stdout.write(prompt)
        sys.stdout.flush()
        buf = ""
        while not self.quit:
            k = self.idle_key(0.05)
            if k is None:
                continue
            if k in "\r\n":
                print()
                return buf
            if k in ("\x7f", "\b"):
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write("\b \b")
            elif k == "\x1b":
                buf = ""
                print(" (cleared)")
                sys.stdout.write(prompt)
            elif not buf and k in "qrzwd":
                print(k)
                return k
            elif k.isdigit():
                buf += k
                sys.stdout.write(k)
            sys.stdout.flush()
        return "q"

    def loop(self):
        if not self.kb.tty:
            sys.exit("needs an interactive terminal (the keyboard is the damping button)")
        print("Waiting for robot state ...", flush=True)
        t0 = time.monotonic()
        while self.be.read() is None:
            self.idle_key(0.1)
            if time.monotonic() - t0 > 10:
                print("!! no /low_state after 10 s (ROS 2 sourced? robot on? --net?)  -- SPACE still sends DAMPING")
                t0 = time.monotonic()
            if self.quit:
                return
        files = self.motions()
        while not self.quit:
            self.show_menu(files)
            s = self.read_line("> ")
            if self.quit or s == "q":
                break
            if s == "r":
                files = self.motions()
                continue
            if s == "w":
                self.end_mode = "damp" if self.end_mode == "walk" else "walk"
                continue
            if s == "d":
                self.kd_scale = {1.0: 1.5, 1.5: 2.0}.get(self.kd_scale, 1.0)
                continue
            if s == "z":
                if self.a.sim:
                    self.be.reset()
                continue
            if s == "":
                if self.last_motion is None or not self.last_motion.exists():
                    continue
                path = self.last_motion
            else:
                i = int(s)
                if not 1 <= i <= len(files):
                    print(f"no motion {i}")
                    continue
                path = files[i - 1]
            self.last_motion = path
            self.run_motion(path)
            files = self.motions()
        if self.be.active or self.be.mode == "CUSTOM":     # WALKING / DAMPING are left as they are
            self.damp("quit")

    # ---------------------------------------------------------------- one run
    @staticmethod
    def tilt(quat_or_rpy):
        """Angle (rad) between the body's up axis and the world's up axis."""
        q = rpy_to_quat(*quat_or_rpy) if len(quat_or_rpy) == 3 else quat_or_rpy
        return math.acos(max(-1.0, min(1.0, quat_to_mat(q)[2, 2])))

    def tilt_err(self, rpy, ref_quat):
        """Angle (rad) between the robot's up axis and the clip's up axis at this frame (heading ignored)."""
        return math.acos(max(-1.0, min(1.0, float(quat_to_mat(rpy_to_quat(*rpy))[:, 2] @ quat_to_mat(ref_quat)[:, 2]))))

    def get_state(self, ref_quat=None):
        st = self.be.read()
        if st is None or (not self.a.sim and time.monotonic() - st.stamp > self.a.watchdog_ms / 1000):
            raise Abort("stale robot state (watchdog)")
        q, qd = self.plant.to_il(st.q), self.plant.to_il(st.dq)
        over = np.abs(qd) / (self.a.vel_frac * self.plant.vmax)
        if np.max(over) > 1:
            raise Abort(f"joint speed guard: {self.plant.il[int(np.argmax(over))]} {qd[int(np.argmax(over))]:+.1f} rad/s")
        tilt = self.tilt(st.rpy)
        if tilt > self.a.max_tilt:
            raise Abort(f"tilt guard: {math.degrees(tilt):.0f} deg (roll {math.degrees(st.rpy[0]):+.0f} pitch {math.degrees(st.rpy[1]):+.0f})")
        if ref_quat is not None and self.tilt_err(st.rpy, ref_quat) > self.a.max_tilt_err:
            raise Abort(f"fall guard: tilt {math.degrees(tilt):.0f} deg, the clip has {math.degrees(self.tilt(ref_quat)):.0f} deg here")
        return st, q, qd

    def limit(self, target, q, qd, kp, kd):
        """Joint range clamp, then shrink q* so the PD torque the driver will produce stays inside the torque cap."""
        flags = 0
        t = np.clip(target, self.plant.lo, self.plant.hi)
        flags |= int(np.any(t != target))
        tau = kp * (t - q) - kd * qd
        over = np.abs(tau) > self.effort
        if np.any(over):
            t = np.where(over, q + (np.sign(tau) * self.effort + kd * qd) / np.maximum(kp, 1e-6), t)
            flags |= 2
        return t, flags

    def run_motion(self, path):
        P, a = self.plant, self.a
        try:
            ref = Reference(path, P)
            policy, policy_file = self.policy_for(path)
            self.cur_policy_file = policy_file
        except Exception as e:  # noqa: BLE001
            print(f"cannot load {path}: {e}")
            return
        dur, warn = self.info(path)
        print(f"\n  {path.name}: {ref.seconds:.1f} s  | soft start {a.soft_start:.1f} s + hold {a.hold:.1f} s"
              f"{'  | max ' + str(a.max_seconds) + ' s' if a.max_seconds else ''}"
              f"  | torque caps: {'training (full)' if a.full_torque else 'Booster deploy'}  | policy {Path(policy_file).name}  | kd x{self.kd_scale:.1f}")
        if warn:
            print(f"  \033[33m! {warn}\033[0m")
        sys.stdout.write("  g = GO, anything else = back  ")
        sys.stdout.flush()
        k = None
        while k is None and not self.quit:
            k = self.idle_key(0.05)
        print()
        if k != "g" or self.quit:
            return

        self.damped.clear()
        self.damp_reason = ""
        n_soft, n_hold = max(1, int(a.soft_start / POLICY_DT)), int(a.hold / POLICY_DT)
        n_ramp = max(1, int(a.gain_ramp / POLICY_DT))
        n_run = ref.n - 1 if not a.max_seconds else min(ref.n - 1, int(a.max_seconds / POLICY_DT))
        log = {k: [] for k in ("t", "phase", "rpy", "gyro", "q", "qd", "tau_est", "action", "target", "kp", "flags", "loop_ms")}
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_dir / f"{stamp}_{path.stem}"
        reason, steps_done, t_start = "clip finished", 0, time.monotonic()

        def record(phase, st, act, tgt, kp, fl, ms):
            for k_, v in (("t", time.monotonic() - t_start), ("phase", phase), ("rpy", st.rpy), ("gyro", st.gyro), ("q", st.q),
                          ("qd", st.dq), ("tau_est", st.tau), ("action", act), ("target", P.to_mj(tgt)), ("kp", P.to_mj(kp)),
                          ("flags", fl), ("loop_ms", ms)):
                log[k_].append(v)

        def check():
            self.beat = time.monotonic()
            if self.damped.is_set():
                raise Abort(self.damp_reason or "damped")
            if self.quit or self.kb.lost:
                raise Abort("quit")

        self.running, self.beat = True, time.monotonic()
        try:
            st, q, qd = self.get_state()
            # 1) take over: hold the current pose with the stiff prepare gains, custom mode
            self.be.enter_custom(st.q, P.to_mj(P.prep_kp), P.to_mj(P.prep_kd), self.damped.is_set)
            self.beat, self.ticking = time.monotonic(), True
            # 2) soft start to the clip's first frame, then hold (training episodes start at the reference pose)
            q0, q_goal = q.copy(), ref.jp[0].copy()
            t_next = time.monotonic()
            for k in range(n_soft + n_hold):
                check()
                w = 0.5 - 0.5 * math.cos(math.pi * min(1.0, (k + 1) / n_soft))
                st, q, qd = self.get_state()
                tgt, fl = self.limit(q0 + (q_goal - q0) * w, q, qd, P.prep_kp, P.prep_kd)
                if not self.be.send(P.to_mj(tgt), P.to_mj(P.prep_kp), P.to_mj(P.prep_kd)):
                    raise Abort(self.damp_reason or "damped")
                record(0, st, np.zeros(22), tgt, P.prep_kp, fl, 0.0)
                if k % 25 == 0:
                    self.status(f"soft start {k * POLICY_DT:4.1f}/{(n_soft + n_hold) * POLICY_DT:.1f} s", st)
                t_next += POLICY_DT
                self.be.advance(t_next)
            # the policy only knows starts from standing upright on its feet: refuse anything else (belt too short, feet
            # not on the floor, leaning on something)
            ref.align_heading(float(st.rpy[2]))
            if self.tilt_err(st.rpy, ref.ref_quat(0)) > a.start_tilt:
                raise Abort(f"not standing like the clip's first frame after the soft start (tilt {math.degrees(self.tilt(st.rpy)):.0f} deg, "
                            f"clip {math.degrees(self.tilt(ref.root_quat[0])):.0f} deg, roll {math.degrees(st.rpy[0]):+.0f} pitch "
                            f"{math.degrees(st.rpy[1]):+.0f}): feet on the floor? belt slack?")
            # 3) policy (clip heading aligned to the robot's yaw above), prepare gains blended into the policy gains
            policy.reset()
            overruns, n_torque = 0, 0
            t_next = time.monotonic()
            for k in range(n_run):
                check()
                t_step = time.monotonic()
                st, q, qd = self.get_state(ref.ref_quat(k))
                tgt, act = policy.step(ref, k, st.rpy, st.gyro, q, qd)
                w = min(1.0, (k + 1) / n_ramp)
                kp, kd = (1 - w) * P.prep_kp + w * P.kp, (1 - w) * P.prep_kd + w * P.kd * self.kd_scale
                tgt, fl = self.limit(tgt, q, qd, kp, kd)
                if not self.be.send(P.to_mj(tgt), P.to_mj(kp), P.to_mj(kd)):
                    raise Abort(self.damp_reason or "damped")
                ms = (time.monotonic() - t_step) * 1000
                record(1, st, act, tgt, kp, fl, ms)
                steps_done, n_torque = k + 1, n_torque + bool(fl & 2)
                if not a.sim:
                    overruns = overruns + 1 if ms > POLICY_DT * 1000 else 0
                    if overruns >= 5:
                        raise Abort("policy loop slower than 50 Hz, 5 ticks in a row")
                if k % 25 == 0:
                    self.status(f"RUN {k * POLICY_DT:5.1f}/{n_run * POLICY_DT:.1f} s  torque-capped {100 * n_torque / (k + 1):3.0f}%  "
                                f"loop {ms:4.1f} ms", st)
                t_next += POLICY_DT
                self.be.advance(t_next)
        except (Abort, RuntimeError) as e:
            reason = str(e)
        except SystemExit:
            reason = "viewer closed"
            self.quit = True
        except Exception as e:  # noqa: BLE001
            reason = f"error: {e!r}"
            traceback.print_exc()
        finally:
            self.running = self.ticking = False
            # belt test protocol: the robot goes limp after every run, finished or not
            if not self.damped.is_set():       # a SPACE / Ctrl+C / joystick stop has damped already
                if reason == "clip finished" and self.end_mode == "walk":
                    if self.be.walk():
                        print(f"\n\033[1;32m*** {self.be.mode} (end of clip): Booster's controller holds the robot; "
                              f"SPACE still damps ***\033[0m", flush=True)
                    else:
                        self.damp("end of clip, switch to WALKING failed")
                else:
                    self.damp("end of clip" if reason == "clip finished" else reason)
            self.save(run_dir, path, ref, log, reason, steps_done, n_run)

    def status(self, msg, st):
        sys.stdout.write(f"\r  {msg}  | roll {math.degrees(st.rpy[0]):+4.0f} pitch {math.degrees(st.rpy[1]):+4.0f} deg   ")
        sys.stdout.flush()

    def save(self, run_dir, path, ref, log, reason, steps_done, n_run):
        a = self.a
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run_dir / "log.npz", **{k: np.asarray(v) for k, v in log.items()})
        flags = np.asarray(log["flags"], dtype=int) if log["flags"] else np.zeros(0, dtype=int)
        meta = {
            "motion": str(path), "policy": self.cur_policy_file, "kd_scale": self.kd_scale, "clip_seconds": round(ref.seconds, 2), "policy_seconds_run": round(steps_done * POLICY_DT, 2),
            "policy_steps_planned": n_run, "completed": reason == "clip finished", "end_reason": reason,
            "backend": "sim" if a.sim else "robot", "dry_run": bool(a.dry_run), "torque_caps": "training" if a.full_torque else "booster_deploy",
            "range_clamped_steps": int(np.sum((flags & 1) > 0)), "torque_capped_steps": int(np.sum((flags & 2) > 0)), "soft_start_s": a.soft_start, "hold_s": a.hold, "gain_ramp_s": a.gain_ramp,
            "max_tilt_rad": a.max_tilt, "max_tilt_err_rad": a.max_tilt_err, "vel_frac": a.vel_frac, "time": datetime.now().isoformat(timespec="seconds"),
            "log_keys": "t phase(0 soft start, 1 policy) rpy gyro q qd tau_est (serial order) action (IsaacLab order) target kp (serial) flags(1 range, 2 torque) loop_ms",
        }
        (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
        color = "32" if meta["completed"] else "33"
        print(f"\033[{color}m  {'DONE' if meta['completed'] else 'STOPPED'}: {reason}  ({meta['policy_seconds_run']:.1f}/"
              f"{n_run * POLICY_DT:.1f} s of policy, torque-capped steps {meta['torque_capped_steps']})  -> {run_dir}\033[0m")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="MuJoCo rehearsal instead of the robot")
    ap.add_argument("--net", default="127.0.0.1", help="robot network address for ChannelFactory (booster_deploy --net)")
    ap.add_argument("--dry-run", action="store_true", help="robot: read state and run the policy, never send or switch mode")
    ap.add_argument("--motions", default=str(HERE / "motions"))
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--onnx", default=None, help="policy_g1.onnx (default policy/)")
    ap.add_argument("--plant", default=None, help="k1_plant.json (default policy/)")
    ap.add_argument("--mjcf", default=None, help="k1.xml (default assets/, then the repo copy)")
    ap.add_argument("--threads", type=int, default=2, help="onnxruntime threads")
    ap.add_argument("--joystick", default="auto", help="'auto', 'off', or a device name/path; any button = DAMPING")
    ap.add_argument("--soft-start", type=float, default=3.0, help="s from the current pose to the clip's first frame")
    ap.add_argument("--hold", type=float, default=1.0, help="s holding the first frame before the policy starts")
    ap.add_argument("--kd-scale", type=float, default=1.0, help="policy-phase kd multiplier (also the d key in the menu)")
    ap.add_argument("--gain-ramp", type=float, default=0.0, help="s blending prepare gains into policy gains (0 = switch at once, as booster_deploy; a blend made the real robot oscillate)")
    ap.add_argument("--max-seconds", type=float, default=None, help="cut every clip after this many seconds of policy")
    ap.add_argument("--end", choices=["damp", "walk"], default="walk",
                    help="after a clip that finishes cleanly: DAMPING, or Booster's WALKING controller (stops always damp)")
    ap.add_argument("--full-torque", action="store_true", help="training torque limits instead of the Booster deploy caps")
    ap.add_argument("--vel-frac", type=float, default=0.9, help="DAMP if any |qd| > this x joint speed limit")
    ap.add_argument("--max-tilt", type=float, default=1.2, help="DAMP if the trunk tilts more than this from upright (rad)")
    ap.add_argument("--max-tilt-err", type=float, default=0.6, help="DAMP if the trunk tilt differs from the clip's by more (rad)")
    ap.add_argument("--start-tilt", type=float, default=0.2, help="start the policy only within this of the clip's first-frame tilt (rad)")
    ap.add_argument("--watchdog-ms", type=float, default=50.0, help="robot: DAMP if /low_state is older than this")
    ap.add_argument("--stall-ms", type=float, default=200.0, help="DAMP if the control loop does not tick for this long")
    ap.add_argument("--max-start-dev", type=float, default=0.35, help="menu warning if the first frame is this far from standing")
    ap.add_argument("--no-viewer", action="store_true", help="sim: no window")
    ap.add_argument("--fast", action="store_true", help="sim: do not pace to real time")
    ap.add_argument("--no-belt", action="store_true", help="sim: no belt (the robot falls when it damps)")
    ap.add_argument("--belt-slack", type=float, default=0.08, help="sim: m the trunk can drop before the belt catches it")
    a = ap.parse_args()
    app = App(a)
    try:
        app.loop()
    finally:
        if app.be.active:
            app.damp("exit")
        app.be.close()
        app.kb.restore()
        print("bye")


if __name__ == "__main__":
    main()
