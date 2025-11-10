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


def save_policy(policy, path_npz):
    """Save policy weights (supports linear and MLP)."""
    os.makedirs(os.path.dirname(path_npz), exist_ok=True)
    if hasattr(policy, "W"):  # LinearPolicy
        np.savez(path_npz, W=policy.W, b=policy.b, kind="linear")
    else:
        # MLPPolicy
        W1, b1, W2, b2, W3, b3 = policy.params
        np.savez(path_npz, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, kind="mlp")


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
    ap.add_argument("--policy", type=str, default="linear", choices=["linear", "mlp"])
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--outdir", type=str, default="artifacts")
    ap.add_argument("--mutation_sigma_floor", type=float, default=0.06)
    ap.add_argument("--init_policy", type=str, default=None,
                help="Path to a .npz champion to seed the initial population")
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
    )

    # Run GA (multi-core)
    print(f"🧠 Starting GA with {os.cpu_count()} CPU cores...")
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
