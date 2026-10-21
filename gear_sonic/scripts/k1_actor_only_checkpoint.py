"""Strip a SONIC training checkpoint down to the actor (no critic, optimizer, scheduler or env state).

The result loads in eval_agent_trl.py and eval_k1_mujoco.py like the full checkpoint (config.yaml next to it).
Usage: python gear_sonic/scripts/k1_actor_only_checkpoint.py <model_step_XXXXXX.pt> <out_actor.pt>
"""

import hashlib
import sys
import types

import torch


def main(src: str, dst: str):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = {k: v.detach().clone().contiguous() for k, v in ck["policy_state_dict"].items()}
    step = int(ck["state"].global_step)
    torch.save({
        "policy_state_dict": sd,
        "state": types.SimpleNamespace(global_step=step),  # eval_agent_trl.py reads checkpoint["state"].global_step
        "meta": {"global_step": step, "source_md5": hashlib.md5(open(src, "rb").read()).hexdigest(),
                 "contents": "policy_state_dict only"},
    }, dst)
    back = torch.load(dst, map_location="cpu", weights_only=False)["policy_state_dict"]
    assert all(torch.equal(back[k], ck["policy_state_dict"][k]) for k in sd)
    print(f"wrote {dst}: step {step}, {len(sd)} tensors, {sum(v.numel() for v in sd.values()):,} params")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
