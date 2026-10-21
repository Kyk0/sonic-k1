"""Frame-0 consistency check: after reset the sim robot must sit on the reference (per body, anchor, joints).
Triggered from train_agent_trl.py with ++reset_check=True (run with all bodies tracked)."""
import torch


def run_reset_check(env, n_resets: int = 3):
    base = env.env if hasattr(env, "env") else env
    while hasattr(base, "env") and not hasattr(base, "command_manager"):
        base = base.env
    cmd = base.command_manager.get_term("motion")
    names = list(cmd.cfg.body_names)
    worst_body = torch.zeros(len(names), device=cmd.device)
    worst_anchor, worst_joint = 0.0, torch.zeros(cmd.robot_joint_pos.shape[1], device=cmd.device)
    for r in range(n_resets):
        env.reset()
        base.scene.write_data_to_sim(); base.sim.forward(); base.scene.update(dt=0.0)
        db = (cmd.body_pos_w - cmd.robot_body_pos_w).norm(dim=-1)  # [E, B]
        da = (cmd.anchor_pos_w - cmd.robot_anchor_pos_w).norm(dim=-1)  # [E]
        dj = (cmd.joint_pos - cmd.robot_joint_pos).abs()  # [E, J]
        worst_body = torch.maximum(worst_body, db.max(dim=0).values)
        worst_anchor = max(worst_anchor, da.max().item())
        worst_joint = torch.maximum(worst_joint, dj.max(dim=0).values)
        print(f"[reset_check] reset {r}: body err mean {db.mean():.4f} m, max {db.max():.4f} m | anchor max {da.max():.4f} m | joint max {dj.max():.4f} rad")
        # relative (anchor-aligned) reference, which the terminations use
        dr = (cmd.body_pos_relative_w - cmd.robot_body_pos_w).norm(dim=-1)
        print(f"[reset_check] reset {r}: relative-ref body err mean {dr.mean():.4f} m, max {dr.max():.4f} m")
    print("[reset_check] per-body max error over envs/resets (m):")
    for n, e in zip(names, worst_body.tolist()):
        print(f"[reset_check]   {n:30s} {e:.4f}{'   <-- >2cm' if e > 0.02 else ''}")
    jn = list(cmd.robot.joint_names)
    bad = [(jn[i], round(v, 4)) for i, v in enumerate(worst_joint.tolist()) if v > 0.01]
    print(f"[reset_check] anchor max {worst_anchor:.4f} m; joints >0.01 rad: {bad if bad else 'none'}")
    ok = worst_body.max().item() < 0.02 and worst_anchor < 0.02
    print(f"[reset_check] RESULT: {'PASS' if ok else 'FAIL'} (frame-0 max body err {worst_body.max().item():.4f} m)")
