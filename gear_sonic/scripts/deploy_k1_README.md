# SONIC-K1 on a real Booster K1: deploy script and test ladder

`deploy_k1_booster.py` runs `policy_g1.onnx` (robot-motion tracking) at 50 Hz on the Booster low-level SDK
(`/low_state` in, `joint_ctrl` out, `B1LocoClient` for mode changes), with a safety layer. **It has not been run on
hardware yet.** Everything below was checked only in the MuJoCo bench backend (`--backend mujoco`), which runs the
same code path with the motor-driver PD emulated at 1 kHz.

## What the sim robustness sweep says (E6, MuJoCo, 200 held-out clips, it50000)

| condition | paper-style | official-like |
|---|---|---|
| baseline | 99.0% | 90.5% |
| action latency 20 ms / **40 ms** | 98.5% / **56.0%** | 89.5% / **53.5%** |
| Booster deploy torque caps (legs 30/35/20/40/20 N m) | 99.0% | 91.0% |
| caps + 20 ms latency | 98.5% | 85.5% |
| PD gains x0.6 / x1.4 | 99.0% / 98.0% | 87.5% / 90.5% |
| motor strength x0.7 | 99.0% | 90.5% |
| ground friction 0.2 | 94.0% | 82.5% |

**Keep the sensor-to-command latency under ~20 ms.** At 40 ms the policy fails on about half the clips.

## Known sim-to-real risks

- **Torque limits.** Training used the BeyondMimic K1 motor specs (hip pitch 68, hip roll 76, knee 112, ankle 38 N m).
  Booster's own deploy config caps the legs much lower (30/35/20/40/20 N m). Ordinary clips are fine under the caps
  in sim (table above), but athletic clips will saturate. Run with `--booster-effort-caps` until the real limits are
  confirmed.
- **Joint speeds.** Training had no velocity cap. In the bench the policy reaches 7-12 rad/s on ordinary clips and up
  to ~23 rad/s on the elbows for fast motions. The velocity guard (0.9 x motor limit: legs 14.7, ankles 17.6, arms 33.5,
  head 7.9 rad/s) stops the run before the motors are pushed past spec.
- **Start pose.** Training starts every episode at the reference pose. The script therefore soft-starts to the clip's
  first frame (`--soft-start-to clip`). Clips that start mid-stride (`*_loop_*`) make a standing robot jump straight
  into a gait and trip the velocity guard: use clips that start from standing.
- **Position clamp.** Targets are clipped to the k1.xml joint ranges. The policy often aims slightly past a limit
  (e.g. knee below 0), so the clamp fires on most steps. It only removes force pushing into the limit stop.
- **State estimation.** Root height and linear velocity are not observed by the actor, but the anchor orientation uses
  the IMU yaw: yaw drift over a long clip shows up as a heading error.
- Only the robot-motion (g1) encoder is wired up. Teleop and SMPL inputs are not.

## Test ladder

Every level: first `--dry-run` with the robot on its stand (logs actions, sends nothing, never changes mode),
compare the log with the same clip in `--backend mujoco`, then a live run on a gantry with a spotter and the e-stop
ready (press `e` + Enter, or Ctrl-C). Move up a level only after three clean runs.

Candidates below pass the paper-style and official-like checks in MuJoCo under **every** hardware-like condition
(baseline, 20 ms latency, PD x0.8 and x1.2, strength x0.7, Booster caps, caps + 20 ms latency) and start from standing
(legs within 0.3 rad of the default pose, root still). They are held-out evaluation clips, never trained on.
The full list (98 clips) is `deploy_test_ladder.json`.

| level | goal | clips (MuJoCo worst-case MPJPE-L) |
|---|---|---|
| 0 | stand: soft start, hold, policy on a still reference | first frame of any level-1 clip, `--max-seconds 3` |
| 1 | stand + upper body | `yawn_R_003__A167_M` (7 mm), `neutral_cry_standing_R_001__A543_M` (7 mm), `neutral_itching_head_R_001__A545` (9 mm), `drinking_bottle_idle_to_idle_R_001__A551_M` (8 mm) |
| 2 | walk | `neutral_walk_180_R_001__A075_M` (19 mm), `neutral_walk_180_R_001__A084` (18 mm), `walk_180_R_001__A346` (17 mm) |
| 3 | jog / turn | `jog_ff_start_180_R_001__A502` (24 mm), `neutral_turning_around_R_001__A098_M` (18 mm), `turn_run_270_R_001__A332` (26 mm) |
| 4 | athletic | `throw_ball_R_002__A345` (13 mm), `jog_in_place_increasing_speed_back_up_squat_R_001__A493_M` (20 mm), then the E6-robust fitness / boxing clips |

All six bench runs above (one per level, with `--booster-effort-caps`) finished their clips: peak joint speed
7-12.5 rad/s, peak tilt 0.48 rad (the squat clip).

## Commands

```bash
# bench (no robot): same code path in MuJoCo
python gear_sonic/scripts/deploy_k1_booster.py --backend mujoco --onnx onnx/policy_g1.onnx --plant k1_plant.json \
    --motion clip.npz --booster-effort-caps --log bench.npz
# robot, dry run (sends nothing)
python gear_sonic/scripts/deploy_k1_booster.py --backend booster --dry-run --onnx onnx/policy_g1.onnx \
    --plant k1_plant.json --motion clip.npz --booster-effort-caps --log dry.npz
# robot, live (after the dry runs; gantry, spotter, e-stop)
python gear_sonic/scripts/deploy_k1_booster.py --backend booster --onnx onnx/policy_g1.onnx --plant k1_plant.json \
    --motion clip.npz --booster-effort-caps --max-seconds 5 --log live.npz
```

The Booster backend needs the Booster SDK environment (`rclpy`, `booster_interface`, `booster_robotics_sdk_python`),
as used by Booster's `booster_deploy`. The bench backend needs only `numpy`, `onnxruntime` and `mujoco`.
