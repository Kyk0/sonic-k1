# SONIC-K1 on the Booster K1: motion menu for belt tests

Pick a motion from `motions/`, press `g`, the robot plays it with `policy_g1.onnx` and goes to **DAMPING** when it ends.
**SPACE damps at any moment.** Connected the way booster_deploy connects (ChannelFactory → B1LocoClient for modes,
ROS 2 `/low_state` in, `joint_ctrl` out, serial motor order), but standalone: booster_deploy itself is not needed.

**Used on a real Booster K1** with SONIC-K1 v1.0 (walking and turning, squats, lunges, shadow boxing). The same
code runs in MuJoCo (`--sim`) and against a fake Booster SDK/ROS 2 backed by MuJoCo (`fake_booster/`) for rehearsal.

```
app.py           the only entry point: menu, safety, logging
robot.py         BoosterK1 (real robot) and SimK1 (MuJoCo) with the same interface
sonic_k1.py      policy core: observation, reference clip, policy_g1.onnx (same as training / eval_k1_mujoco)
make_motion.py   put a clip into motions/ (convert, trim, slow down, lead-in/out to standing, checks)
make_kit.sh      build a copy-anywhere folder (code + policy/ + assets/k1.xml + stand clip + sample motions)
run.sh           one-command launcher on the robot (loads ROS 2 if needed, then app.py)
motions/         your clips; the menu lists whatever is here
runs/            one folder per run: log.npz + run.json
fake_booster/    fake SDK + ROS 2 for rehearsing the robot path on a laptop (never on the robot)
```

## Setup

```bash
# on a machine with the repo: build the kit (downloads the weights from Hugging Face), copy it over
bash sonic_k1_deploy/make_kit.sh sonic_k1_kit
scp -r sonic_k1_kit booster@<robot-ip>:~/

# on the robot (the same environment booster_deploy runs in: firmware >= v1.4, ROS 2 Humble, booster SDK python)
source /opt/ros/humble/setup.bash
source /opt/booster/BoosterRos2Interface/install/setup.bash
pip3 install onnxruntime numpy          # evdev too, for a gamepad as the damping button (optional)
cd ~/sonic_k1_kit
./run.sh --dry-run                      # or ./run.sh: sources ROS 2 + BoosterRos2Interface if needed (BOOSTER_ROS_SETUP)
```

Run it **on the robot's computer**, not over Wi-Fi. v1.0 is trained with 0-40 ms of command delay (sim, 1,985 held-out
clips at +40 ms: 96.6% vs 76.7% for v0.1), but every millisecond still costs accuracy.
From a laptop on the robot's wired network, `--net <laptop ip on that network>` as in booster_deploy.

## Run

```bash
python3 app.py --sim                            # MuJoCo rehearsal: same menu, keys, guards; robot hangs in a belt
python3 app.py --sim --no-belt                  # starts standing (PREPARE-like hold), falls when damped
PYTHONPATH=fake_booster python3 app.py          # the ROBOT code path against a fake robot (mode log: fake_booster.log)
python3 app.py --dry-run                        # robot: reads state, never sends a command, never changes mode
python3 app.py                                  # robot, live
```

macOS: the MuJoCo window needs `mjpython app.py --sim`.

| key | |
|---|---|
| **SPACE** | **DAMPING, at any moment** (menu, soft start, running). Also: any gamepad button, Ctrl+C |
| `<n>` Enter | pick motion n (Enter alone = the last one again), then `g` = GO, anything else = back |
| `r` | rescan `motions/` |
| `q` | quit (damps first if the app moved the robot) |
| `z` | sim only: reset the robot |

The keyboard is read in one place only (two stdin readers steal each other's keys). Ctrl+C damps and keeps the app
running; a second Ctrl+C in the menu within 3 s quits.

## What GO does

1. checks the state is fresh and the robot is not tilted beyond the guard
2. publishes "hold the current pose" with Booster's stiff prepare gains, then `ChangeMode(kCustom)` (as booster_deploy)
3. **soft start**: cosine blend from the current pose to the clip's first frame in 3 s (`--soft-start`), hold 1 s (`--hold`)
4. **gate**: the trunk must be within 11° of the first frame's tilt (`--start-tilt`); otherwise DAMPING: feet not on
   the floor, belt too short, leaning on something
5. **policy** plays the clip at 50 Hz; the clip's heading is turned to the robot's; gains blend from prepare to the
   policy's over 0.5 s
6. **DAMPING** when the clip ends: a kp = 0 command first (limp at once), then `ChangeMode(kDamping)`; back to the menu

DAMPING also on: SPACE / gamepad / Ctrl+C / SIGTERM; tilt > 69° (`--max-tilt`); trunk tilt more than 34° away from the clip's
(`--max-tilt-err`, a fall); any joint faster than 90% of its limit (`--vel-frac`); `/low_state` older than 50 ms
(`--watchdog-ms`); the 50 Hz loop late 5 ticks in a row; the loop not ticking for 200 ms (`--stall-ms`, checked from a
separate thread); any exception; exit. After damping nothing more is published (the publish lock checks it).

Every target is clamped to the k1.xml joint ranges, then shrunk so the PD torque stays inside **Booster's deploy torque
caps** (legs 30/35/20/40/20/20 Nm; `--full-torque` = the training limits). In sim with these caps 99% of 200
held-out clips still succeed (eval E6). Standing still, the policy locks a knee straight (its action saturates there); the range clamp
stops it from pushing into the hard stop.

## Belt session

1. Rehearse the exact files in sim first: `python3 app.py --sim` (and `--no-belt`).
2. On the robot: `--dry-run`. Check the menu shows live roll/pitch (tilt the robot by hand) and SPACE prints DAMPING.
3. Robot hanging in the belt, **feet on the floor with a little slack**, in DAMPING (Booster remote/app). Live: start
   with `stand_10s`, then the gentlest clips. Hand on SPACE.
4. The belt limits what can be tested: in sim (a rope 1 m above an attachment over the centre of mass, 8 cm slack)
   deep bends get held up and walks get pulled back once they leave the spot under the anchor; the fall guard then
   damps. Without the belt the same clips finish. Belt first for stand / upper body / walk in place; long walks need
   a trolley or slack.
5. If the Booster firmware refuses CUSTOM from DAMPING, the app says so and damps; put the robot in PREPARE with the
   remote first, then GO.

Logs: `runs/<time>_<motion>/run.json` (end reason, seconds played, clamp counts) and `log.npz` (per tick: phase,
rpy, gyro, q, qd, tau_est in serial order, raw action in IsaacLab order, targets, kp, flags, loop time).
