# SONIC-K1 data pipeline

**No motion data ships with this repo.** You need your own access to
[Bones-SEED](https://huggingface.co/datasets/bones-studio/seed) under its license.

Training data includes Motion Data by Bones Studio (https://bones.studio/). Use of the underlying dataset is subject
to the BONES Motion Capture Dataset License Agreement.

## 1. Retarget human → K1 with UMR
- Retargeter: [UMR](https://github.com/hanyang9/UMR). We only link to it; no UMR code is vendored here.
- Robot model: the official Booster `K1_22dof.xml` (booster_assets).
- Config: [`umr_k1_official.json`](umr_k1_official.json), at **60 fps**.
- Gotcha: UMR's SOMA path needs the SOMA-X assets from HF revision `06f07a4ba1`. Newer `SOMA_neutral.npz` files lack `bind_shape`.

## 2. Quality check
Every retargeted clip is checked for:
- joint-limit violations;
- joint-velocity jumps;
- feet below or floating above the floor;
- NaNs.

About 99.6% of clips pass.

## 3. K1 feasibility tags
The K1 is 0.93 m with no waist and no wrists. Clips are tagged *feasible / maybe / infeasible* from keywords and
kinematics, e.g., elevated surfaces, props and lying on the floor. Training used feasible clips only.

## 4. Training list
- **9,991 clips** (`train_10k`), stratified across Bones-SEED categories.
- **1,985 clips** held out for evaluation before training.

## 5. Convert to SONIC motion_lib
```bash
python gear_sonic/data_process/convert_k1_npz_to_motion_lib.py --input <k1_npz_dir> --output <motion_lib_dir>
```
Then pass `<motion_lib_dir>` as `motion_lib_cfg.motion_file` (see README_K1.md → Train your own).

## Your own motions (video → K1)
Any monocular human-motion estimator that outputs SMPL (we used GVHMR) → UMR (same K1 config) → converter above.
