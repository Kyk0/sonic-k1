# SONIC-K1 data pipeline

**No motion data ships with this repo.** You need your own access to
[Bones-SEED](https://huggingface.co/datasets/bones-studio/seed) under its license.

Training data includes Motion Data by Bones Studio (https://bones.studio/). Use of the underlying dataset is subject
to the BONES Motion Capture Dataset License Agreement.

## 1. Retarget human → K1 with UMR
- Retargeter: [UMR](https://github.com/hanyang9/UMR). We only link to it; no UMR code is vendored here.
- Robot model: the official Booster `K1_22dof.xml` (booster_assets).
- Config: start from UMR's own `humanoid_retarget_booster_k1_example.json`. Switch it to the official K1 joint
  names and the official `K1_22dof` model, and retarget at **60 fps**. We don't ship our modified config because UMR
  has no license. Ask us for the settings if you need them.
- Gotcha: UMR's SOMA path needs the SOMA-X assets from HF revision `06f07a4ba1`. Newer `SOMA_neutral.npz` files lack `bind_shape`.

## 2. Quality check
Every retargeted clip is checked for:
- joint-limit violations;
- joint-velocity jumps;
- feet below or floating above the floor;
- NaNs.

142,024 of 142,537 retargeted clips pass (99.6%).

## 3. K1 feasibility tags
The K1 is 0.93 m with no waist and no wrists. Clips are tagged *feasible / maybe / infeasible* from keywords and
kinematics, e.g., elevated surfaces, props and lying on the floor. Training used feasible clips only.

## 4. Training list
- **9,991 clips** (`train_10k`), stratified across Bones-SEED categories. All of them are QA-passing and tagged feasible
  (the feasible pool is 62,427 clips).
- **Eval split frozen before any training.** 1,985 of its clips are used as the unseen set, none of them in `train_10k`.

## 5. Convert to SONIC motion_lib
```bash
python gear_sonic/data_process/convert_k1_npz_to_motion_lib.py --input <k1_npz_dir> --output <motion_lib_dir>
```
Then pass `<motion_lib_dir>` as `motion_lib_cfg.motion_file` (see README.md → Train your own).

## Your own motions (video → K1)
Any monocular human-motion estimator that outputs SMPL (we used GVHMR) → UMR (same K1 config) → converter above.
