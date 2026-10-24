# Sample K1 motions

Ten motions for `gear_sonic/scripts/play_k1.py`. None of them were in SONIC-K1's training data.

| File | Motion | Length |
|---|---|---|
| `shadow_boxing.npz` | Shadow boxing (the default) | 12.0 s |
| `boxing_dodge.npz` | Boxing dodge with a 270° turn | 3.9 s |
| `scissors_jump.npz` | Scissors jumps | 5.6 s |
| `mountain_climber.npz` | Standing mountain climber | 4.0 s |
| `high_knees.npz` | High-knees jog | 4.1 s |
| `forward_lunge.npz` | Forward lunges | 9.0 s |
| `squat.npz` | Squats | 7.9 s |
| `crouch_walk.npz` | Crouch walk with a turn | 4.8 s |
| `injured_leg_jog.npz` | Jogging on an injured leg | 4.1 s |
| `exaggerated_walk.npz` | Exaggerated walk | 26.2 s |

Format: `qpos` [T, 29] float32 (root xyz, root quat wxyz, 22 joints in MuJoCo `k1.xml` order) at `fps` 50. Each file
also carries `joint_names`, `source` and `processing` strings.

**What these are.** Robot trajectories recorded from SONIC-K1 itself: the policy tracked a held-out Bones-SEED clip
(retargeted to the K1) in MuJoCo, and the robot's own joint and root motion was saved at 50 Hz. They contain no human
motion capture data.

**Source and license.** Training data includes Motion Data by Bones Studio (https://bones.studio/). Use of the
underlying dataset is subject to the BONES Motion Capture Dataset License Agreement. Like the model weights, these
files are for non-commercial use (CC BY-NC 4.0).
