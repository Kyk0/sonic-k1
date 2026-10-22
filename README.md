<h1 align="center">SONIC-K1</h1>
<p align="center"><b>NVIDIA's SONIC whole-body motion tracker, ported to the Booster K1 humanoid.</b><br>The first open-weights general motion tracker for the K1, built for small humanoids.<br>
One policy, thousands of human motions, trained from scratch with under 1% of SONIC's compute.</p>

<p align="center">
  <a href="https://huggingface.co/Kyry10/sonic-k1"><img src="https://img.shields.io/badge/🤗%20weights-Kyry10%2Fsonic--k1-yellow"></a>
  <a href="https://kyk0.github.io/sonic-k1/"><img src="https://img.shields.io/badge/project-page-8fe04a"></a>
  <a href="https://arxiv.org/abs/2511.07820"><img src="https://img.shields.io/badge/SONIC-arXiv%202511.07820-b31b1b"></a>
  <img src="https://img.shields.io/badge/robot-Booster%20K1%2022--DoF-blue">
  <img src="https://img.shields.io/badge/sim2sim-Isaac%20→%20MuJoCo-green">
</p>

<p align="center"><img src="docs/k1/media/hero.webp" width="100%"></p>

<p align="center">
  <img src="docs/k1/media/loop_1.webp" width="32%">
  <img src="docs/k1/media/loop_2.webp" width="32%">
  <img src="docs/k1/media/loop_3.webp" width="32%">
</p>

> This is a fork of [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) that adds the
> **Booster K1** as a SONIC embodiment. Everything K1-specific is documented here and in [`docs/k1/`](docs/k1/).
> The original NVIDIA README is kept unchanged in [`README_SONIC_upstream.md`](README_SONIC_upstream.md).

## Results

| | Unseen (1,985 clips) | Seen (2,000 clips) |
|---|---|---|
| Success, SONIC's rule (root or end-effector height off > 0.25 m; MuJoCo) | **97.4%** | — |
| Success, no-fall rule (root height + tilt; Isaac) | **98.7%** | 98.8% |
| Success, strict repo eval (adds foot-position limit; Isaac) | 93.1% | 98.5% |
| MPJPE-L / PA-MPJPE | **14.7 / 10.8 mm** | 15.2 / 11.2 mm |

10 s windows, deterministic. Hold-out split of Bones-SEED. Full protocol: [`docs/k1/eval.md`](docs/k1/eval.md).

**How it compares** (SONIC's own success rule, root or end-effector height off by > 0.25 m; others as published on the Unitree G1, so this is indicative):

| Benchmark | **SONIC-K1** (ours, K1) | SONIC (G1) | BeyondMimic (G1) | Any2Track (G1) |
|---|---|---|---|---|
| PHUMA | **93.6%** | 97.2% | 73.8% | 78.5% |
| Held-out mocap | **97.4%** | 98.5% | 82.0% | 61.6% |

Details and LAFAN1 are in [`docs/k1/eval.md`](docs/k1/eval.md).

**How much compute?** About 100 GPU-hours, under 1% of what the SONIC paper used for the G1:

| | SONIC (G1, paper) | SONIC-K1 |
|---|---|---|
| GPU-hours | >9,000 | **~100** |
| Training motions | ~100M frames (700 h) | 9,991 clips |
| Unseen success, SONIC's rule | 98.5% (in-house test set) | 97.4% (MuJoCo) |


## Get started

### 1. Play the policy in MuJoCo (CPU is fine)
```bash
git clone https://github.com/Kyk0/sonic-k1 && cd sonic-k1
# same environment as upstream SONIC (see README.md → Installation), plus mujoco + onnxruntime
huggingface-cli download Kyry10/sonic-k1 --local-dir checkpoints/sonic-k1
python gear_sonic/scripts/eval_k1_mujoco.py \
    --checkpoint checkpoints/sonic-k1/sonic_k1_it50000_actor.pt \
    --onnx checkpoints/sonic-k1/onnx \
    --motion-dir <k1_motion_lib_dir> --out runs/mujoco --render-dir runs/mujoco/qpos
```

### 2. Bring your own motion
Retarget any SMPL/BVH clip to K1 with [UMR](https://github.com/hanyang9/UMR), using
UMR's Booster K1 example config, switched to the official K1 model (see [`docs/k1/data.md`](docs/k1/data.md)), then convert it with
[`gear_sonic/data_process/convert_k1_npz_to_motion_lib.py`](gear_sonic/data_process/convert_k1_npz_to_motion_lib.py).
Video → motion works too, through any monocular human-pose estimator (e.g. GVHMR) followed by UMR.

### 3. Train your own
```bash
# Isaac Sim 5.1 / Isaac Lab, one H100 (80–96 GB); same setup as upstream training.md
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_k1 \
    num_envs=16384 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/k1_motion_lib>
```
Training on headless H100 servers? SONIC-K1 was trained with the setup in our [Isaac Lab on H100 guide](https://github.com/Kyk0/isaaclab-h100): physics-only Isaac Lab on shared, compute-only GPUs. A pre-converted K1 USD (the exact asset used for training) is at [Kyry10/booster-k1-usd](https://huggingface.co/datasets/Kyry10/booster-k1-usd).

You need your own access to [Bones-SEED](https://huggingface.co/datasets/bones-studio/seed). We do **not** redistribute
any motion data, raw or retargeted. See [`docs/k1/data.md`](docs/k1/data.md) for the retarget → QA → feasibility pipeline.

## What changed vs. upstream SONIC

- `gear_sonic/envs/manager_env/robots/k1.py`: K1 robot config, joint and body maps, IsaacLab↔MuJoCo order.
- K1 experiment config:
  - head-joint tracking term;
  - pushes and COM randomization ×0.7;
  - mass 0.9–1.3;
  - init std 0.5.
- MuJoCo sim-to-sim eval: the actor gives the same outputs as in Isaac Lab, 5 ms physics, 50 Hz policy. 99.0% of 200 unseen clips run without a fall.
- ONNX export (fused `policy_g1.onnx` + split encoder/decoder), with `--parity-onnx` to check it against PyTorch.

See [`NOTICE`](NOTICE) for the complete list of modified files.

## Coming soon
- 🤖 **Real Booster K1 runs**: the policy on hardware, with videos
- 🥽 **Whole-body teleoperation**: drive the K1 live from a VR headset
- 🎮 **Deployment controller**: an open runtime for the Booster SDK
- 🎥 **Video → K1**: SMPL/video input, so you can film a move and the K1 does it
- 📈 **A bigger model**: trained on 62K motions (6× more)

## License and credits
- Code: Apache-2.0 (upstream NVIDIA license kept; our changes are listed in `NOTICE`).
- Weights: CC BY-NC 4.0 (see the [model card](https://huggingface.co/Kyry10/sonic-k1)).
- K1 model: BoosterRobotics/booster_assets, BSD-3-Clause.
- **Training data includes Motion Data by Bones Studio** ([bones.studio](https://bones.studio/)).
  Use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement.

Built on [SONIC](https://arxiv.org/abs/2511.07820) by NVIDIA GEAR. Please cite it (and this repo, through
[`docs/k1/CITATION.cff`](docs/k1/CITATION.cff)) if this helps your work.

Made by [Kyrylo Kolesnichenko](https://kyry.me) at ATR Lab.
