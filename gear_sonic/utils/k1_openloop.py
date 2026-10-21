"""Open-loop survival test: reference joints as PD targets, no policy. Tracks, per env, the first termination
step and root xy drift (vs start and vs reference anchor). Triggered by ++openloop_check=True."""
import torch


def run_openloop(env, steps: int = 500):
    base = env.env if hasattr(env, "env") else env
    while hasattr(base, "env") and not hasattr(base, "command_manager"):
        base = base.env
    cmd = base.command_manager.get_term("motion"); robot = cmd.robot
    act = base.action_manager.get_term("joint_pos"); scale, offset = act._scale, act._offset
    tm = base.termination_manager; names = tm.active_terms
    env.reset()
    E = base.num_envs; dev = base.device
    first_done = torch.full((E,), steps, device=dev, dtype=torch.long)
    first_reason = [""] * E
    root0 = robot.data.root_pos_w[:, :2].clone()
    alive = torch.ones(E, dtype=torch.bool, device=dev)
    for t in range(steps):
        base.step((cmd.joint_pos - offset) / scale)
        done = base.reset_buf.clone()
        newly = done & alive
        if newly.any():
            first_done[newly] = t + 1
            for n in names:
                hit = (tm.get_term(n) & newly).nonzero().flatten().tolist()
                for i in hit:
                    if not first_reason[i]: first_reason[i] = n
            alive &= ~done
        if t + 1 in (1, 10, 25, 50, 100, 200, 300, 400, 500) and alive.any():
            d0 = (robot.data.root_pos_w[alive, :2] - root0[alive]).norm(dim=-1)
            dr = (robot.data.root_pos_w[alive, :2] - cmd.anchor_pos_w[alive, :2]).norm(dim=-1)
            dz = (robot.data.root_pos_w[alive, 2] - cmd.anchor_pos_w[alive, 2])
            print(f"[openloop2] step {t+1:3d}: alive {int(alive.sum())}/{E} | root xy drift vs start {d0.mean():.4f} (max {d0.max():.4f}) m | vs ref {dr.mean():.4f} m | root z - ref {dz.mean():+.4f} m")
        elif t + 1 in (1, 10, 25, 50, 100, 200, 300, 400, 500):
            print(f"[openloop2] step {t+1:3d}: alive 0/{E}")
    from collections import Counter
    surv = (first_done >= steps).float().mean().item()
    print(f"[openloop2] survived {steps} steps: {surv*100:.1f}% | median first termination step {first_done.float().median().item():.0f}")
    print(f"[openloop2] first-termination reasons: {dict(Counter(r for r in first_reason if r))}")
