# experiments/train_tinygrid.py
"""
Train an evolutionary policy on the TinyGrid world using all CPU cores.

- Compatible with LinearPolicy and MLPPolicy (if eng/policies_mlp.py exists)
- Uses multiprocessing automatically (via eng/evolve.run_ga)
- Greedy evaluation only (no exploration during inference)
"""

import argparse
import os
import json
import numpy as np
import sys

import numpy as np
from eng.policies import LinearPolicy
from eng.policies_mlp import MLPPolicy


# Allow imports when running as module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eng.evolve import GAConfig, run_ga
from tasks.tinygrid import TinyGrid


from eng.io_policies import load_policy_npz  # already used in evolve
from eng.evolve import save_policy_npz      # we defined this earlier


def save_policy(policy, path_npz):
    """Save policy weights (supports linear, MLP, graph)."""
    os.makedirs(os.path.dirname(path_npz), exist_ok=True)

    # If policy exposes as_dict(), use that.
    if hasattr(policy, "as_dict"):
        np.savez_compressed(path_npz, **policy.as_dict())
        return

    # Linear fallback
    if hasattr(policy, "W") and hasattr(policy, "b"):
        np.savez_compressed(path_npz, W=policy.W, b=policy.b, kind="linear")
        return

    # Generic params list (e.g. some MLP variants)
    if hasattr(policy, "params"):
        params = {}
        for i in range(0, len(policy.params), 2):
            W = policy.params[i]
            b = policy.params[i + 1] if (i + 1) < len(policy.params) else None
            layer_idx = i // 2
            params[f"W{layer_idx}"] = W
            if b is not None:
                params[f"b{layer_idx}"] = b
        params["kind"] = "mlp"
        np.savez_compressed(path_npz, **params)
        return

    raise ValueError(f"Don't know how to save policy of type: {type(policy)}")

def main():
    ap = argparse.ArgumentParser(description="Train an evolutionary TinyGrid agent.")
    ap.add_argument("--pop_size", type=int, default=128)
    ap.add_argument("--elites", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--mutation_sigma", type=float, default=0.12)
    ap.add_argument("--sigma_decay", type=float, default=0.98)
    ap.add_argument("--crossover_rate", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
    "--policy",
    type=str,
    default="linear",
    choices=["linear", "mlp", "graph"],)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--outdir", type=str, default="artifacts")
    ap.add_argument("--mutation_sigma_floor", type=float, default=0.06)
    ap.add_argument("--init_policy", type=str, default=None,
                help="Path to a .npz champion to seed the initial population")
    ap.add_argument(
    "--processes",
    type=int,
    default=None,
    help="Number of worker processes for GA evaluation (default: use all CPU cores)",
)
    args = ap.parse_args()

    # Configure GA
    cfg = GAConfig(
        pop_size=args.pop_size,
        elites=args.elites,
        episodes=args.episodes,
        max_steps=args.max_steps,
        mutation_sigma=args.mutation_sigma,
        sigma_decay=args.sigma_decay,
        crossover_rate=args.crossover_rate,
        seed=args.seed,
        init_policy=args.init_policy,   # NEW
        policy=args.policy,
        generations=args.generations,
        processes=args.processes,  
    )

    # Run GA (multi-core)
    n_procs = args.processes or os.cpu_count()
    print(f"🧠 Starting GA with {n_procs} worker processes (system cores: {os.cpu_count()})...")
    winner, history = run_ga(
        env_ctor=TinyGrid,
        env_kwargs={"max_steps": args.max_steps},
        cfg=cfg,
    )

    # Save results
    os.makedirs(args.outdir, exist_ok=True)
    policy_path = os.path.join(args.outdir, f"best_{args.policy}_policy.npz")
    save_policy(winner, policy_path)

    hist_path = os.path.join(args.outdir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"✅ Training complete.")
    print(f"   Saved best policy → {policy_path}")
    print(f"   Saved history     → {hist_path}")


if __name__ == "__main__":
    main()
