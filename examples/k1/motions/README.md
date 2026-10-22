# Sample K1 motions

Six walking styles for `gear_sonic/scripts/play_k1.py`. None of them were in SONIC-K1's training data.

| File | 100STYLE style |
|---|---|
| `penguin_walk.npz` | Penguin, forward walk |
| `zombie_walk.npz` | Zombie, forward walk |
| `robot_walk.npz` | Robot, forward walk |
| `chicken_walk.npz` | Chicken, forward walk |
| `dinosaur_walk.npz` | Dinosaur, forward walk |
| `superman_walk.npz` | Superman, forward walk |

Format: `qpos` [600, 29] float32 (root xyz, root quat wxyz, 22 joints in MuJoCo `k1.xml` order) and `fps` (30). Each file
also carries `joint_names`, `source` and `processing` strings.

**Source and license.** Derived from the [100STYLE dataset](https://doi.org/10.5281/zenodo.8127870) by Ian Mason,
Sebastian Starke and Taku Komura ("Real-Time Style Modelling of Human Locomotion via Feature-Wise Transformations and Local
Motion Phases", 2022), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
**Changes made:** BVH converted to SMPL-X, retargeted to the Booster K1 with UMR, and stored as K1 joint-space trajectories
at 30 fps. These derived files are shared under the same CC BY 4.0 license.
