# SONIC-K1 evaluation protocol

All numbers are for checkpoint **it50000** (`model_step_050000.pt`), deterministic actions, Isaac Lab `im_eval`.

## Sets
- **Unseen:** 1,985 Bones-SEED clips held out before training. Never sampled during training.
- **Seen:** 2,000 clips from the 9,991-clip training list.
- Each clip is cut to a **10 s** window. ~2,000 parallel envs, one clip per env.

## Success rules
| Name | A rollout fails if… |
|---|---|
| **Official** (repo eval command, same preset as the G1 release config) | root/anchor height off > 0.25 m, orientation error > 1 rad, end-effector (ankles/elbows) error > 0.25 m, **or** foot position (world frame) off > 0.2 m |
| **No-foot** | same, with the foot-position limit disabled |
| **Paper-style** (SONIC paper scaling study) | root height off > 0.25 m or tilt > 1 rad |

The official foot term is strict for the K1: most of the gap between official and no-foot is slow foot drift, not falls.

## Metrics (Isaac: over successful rollouts, SONIC/PHC convention; MuJoCo: over all rollouts)
- **MPJPE-L:** local (root-relative) mean per-joint position error, mm.
- **PA-MPJPE:** after Procrustes alignment, mm.
- **MPJPE-G:** global, includes root drift, mm.

## Results, it50000
| Set | Official | No-foot | Paper-style | MPJPE-L | PA | G |
|---|---|---|---|---|---|---|
| Unseen (1,985) | **93.1%** | 97.9% | **98.7%** | 14.7 | 10.8 | 84.9 |
| Seen (2,000) | 98.5% | 98.8% | 98.8% | 15.2 | 11.2 | 82.6 |

Unseen per body group (MPJPE-L, mm): legs 17.4, hands + head 11.7, feet 18.7.

## Over training (unseen, same protocol)
| Iteration | Official | No-foot | Paper-style | MPJPE-L |
|---|---|---|---|---|
| 8,400 | 85.8% | 95.8% | 98.0% | 23.3 |
| 18,100 | 90.4% | 97.2% | 98.3% | 19.3 |
| 29,450 | 92.2% | 97.3% | 98.4% | 17.0 |
| **50,000** | **93.1%** | **97.9%** | **98.7%** | **14.7** |

Paired MPJPE-G on the 1,780 unseen clips that succeed at all three late checkpoints: 132 → 105 → **84 mm**.

## Sim-to-sim (MuJoCo)
- 5 ms physics, 50 Hz policy, explicit PD.
- Actor outputs are identical to Isaac Lab on the same state (TF32 off).
- `k1.xml` overrides: hip-roll effort 76, head armature 0.001.

| MuJoCo, it50000 (g1 encoder, 10 s, MPJPE over all clips) | Paper-style | Official-like | MPJPE-L | PA | G | Joint jerk vs ref |
|---|---|---|---|---|---|---|
| Unseen 200 | **99.0%** | 90.5% | 16.2 | 12.3 | 125 | 0.87× |
| Seen 200 | 96.0% | 93.5% | 16.3 | 11.9 | 83 | — |

Earlier unseen MuJoCo results: it29450 98.0% / 90.5% / 19.1 mm; it8400 97.0% / 84.5%. The seen-200 subset has
harder clips than the unseen-200, which is why seen scores lower here.

**ONNX vs PyTorch** (400 clips, ~190k control steps, PyTorch driving, ONNX evaluated on the same inputs):
- 392/400 clips: max |Δa| ≤ 1.4e-5, median 1.1e-5 (float32; ≈1e-5 rad in joint targets);
- 8/400 clips: one isolated jump of 8e-3 to 2e-2, from a single FSQ token sitting exactly on a quantization
  boundary (~1 step in 400). This comes from FSQ itself, not from the export. Fused and split graphs give the same numbers;
- closed loop with ONNX driving, 200 unseen: 99.0% / 90.5%, MPJPE-L 16.19 mm (PyTorch: 16.16 mm).

**Released-artifact check:** upstream Isaac `eval_agent_trl.py` with the shipped actor-only `.pt` and sanitized
`config.yaml` gives 92.5% official and MPJPE-L 15.3 mm on 200 unseen clips. This is consistent with the 1,985-clip result.

## Caveats
- This is a Bones-SEED hold-out, **not** the SONIC paper's AMASS test set. The K1 is ~0.7× the G1's size, so millimetre errors don't compare with G1 or X2 numbers.
- Evals use the stock mixed g1/teleop encoder sampling. A g1-encoder-only eval scored ~3 points higher (at it8400).
