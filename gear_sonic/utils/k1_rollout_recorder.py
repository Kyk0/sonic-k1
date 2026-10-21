"""Record per-env policy rollouts (robot vs reference) during eval_agent_trl.py for offline rendering.

Enabled with ++k1_rollout_dir=<dir>. One npz per env/motion: qpos [T,29] (robot) and ref_qpos [T,29]
(reference), both root pos (env-origin relative) + root quat wxyz + 22 joints in MuJoCo (k1.xml) order,
fps, motion name, and whether the episode ended by time-out (success) or early termination.
"""
import os

import numpy as np
import torch


class K1RolloutRecorder:
    def __init__(self, env, out_dir: str, max_steps: int = 1500):
        base = env.env if hasattr(env, "env") else env
        while hasattr(base, "env") and not hasattr(base, "command_manager"):
            base = base.env
        self.base, self.out = base, out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.cmd = base.command_manager.get_term("motion")
        self.robot = self.cmd.robot
        self.m2i = list(self.cmd.mujoco_to_isaaclab_dof)  # isaaclab-ordered = mujoco[m2i]; invert for mujoco order
        self.i2m = list(self.cmd.isaaclab_to_mujoco_dof)  # mujoco j -> isaaclab index
        self.E = base.num_envs
        self.buf = [[] for _ in range(self.E)]
        self.ref = [[] for _ in range(self.E)]
        self.done = torch.zeros(self.E, dtype=torch.bool)
        self.result = [None] * self.E
        self.motion_id = self.cmd.motion_ids.clone().cpu()
        self.fps = 1.0 / base.step_dt
        self.max_steps = max_steps
        self.tm = base.termination_manager

    def _names(self):
        ml = self.cmd.motion_lib
        keys = getattr(ml, "_motion_data_keys", None)
        cur = getattr(ml, "_curr_motion_ids", None)
        out = []
        for i in range(self.E):
            mid = int(self.motion_id[i])
            try:
                k = keys[int(cur[mid])] if cur is not None else keys[mid]
            except Exception:
                k = f"motion_{mid}"
            out.append(str(k))
        return out

    def _qpos(self, pos, quat, jpos_il):
        origin = self.base.scene.env_origins
        j = jpos_il[:, self.i2m]  # mujoco order
        return torch.cat([pos - origin, quat, j], dim=-1).cpu().numpy()

    def step(self, dones):
        rob = self._qpos(self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.robot.data.joint_pos)
        ref = self._qpos(self.cmd.anchor_pos_w, self.cmd.anchor_quat_w, self.cmd.joint_pos)
        d = dones.view(-1).bool().cpu()
        timeout = self.tm.get_term("time_out").cpu() if "time_out" in self.tm.active_terms else torch.zeros_like(d)
        for i in range(self.E):
            if self.done[i]:
                continue
            self.buf[i].append(rob[i]); self.ref[i].append(ref[i])
            if d[i] or len(self.buf[i]) >= self.max_steps:
                self.done[i] = True
                reason = "time_out" if (bool(timeout[i]) or len(self.buf[i]) >= self.max_steps) else "terminated"
                if d[i] and not bool(timeout[i]):
                    for n in self.tm.active_terms:
                        if bool(self.tm.get_term(n)[i]):
                            reason = n; break
                self.result[i] = reason
        return bool(self.done.all())

    def save(self):
        names = self._names()
        rows = []
        for i in range(self.E):
            if not self.buf[i]:
                continue
            nm = names[i].replace("/", "_")
            np.savez_compressed(os.path.join(self.out, f"{i:03d}_{nm}.npz"), qpos=np.array(self.buf[i]),
                                ref_qpos=np.array(self.ref[i]), fps=self.fps, motion=nm, result=str(self.result[i]))
            rows.append(f"{i},{nm},{len(self.buf[i])},{self.result[i]}")
        with open(os.path.join(self.out, "summary.csv"), "w") as f:
            f.write("env,motion,steps,result\n" + "\n".join(rows) + "\n")
        ok = sum(1 for i in range(self.E) if self.result[i] == "time_out")
        print(f"[k1_rollout] saved {len(rows)} rollouts to {self.out}; reached end: {ok}/{len(rows)}")
