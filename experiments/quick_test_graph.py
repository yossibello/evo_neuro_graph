#!/usr/bin/env python3
"""
Quick test training run for GraphPolicy (20 generations)
to verify the system works before longer experiments.
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from eng.evolve import GAConfig, run_ga, save_policy_npz
from tasks.tinygrid import TinyGrid


def main():
    print("🧪 Quick test of GraphPolicy evolution")
    print("=" * 60)
    
    cfg = GAConfig(
        pop_size=64,        # Smaller for speed
        elites=4,
        episodes=8,         # Fewer episodes
        max_steps=200,
        generations=20,     # Quick test
        mutation_sigma=0.15,
        sigma_decay=0.97,
        mutation_sigma_floor=0.06,
        crossover_rate=0.35,
        seed=42,
        policy="graph",
        processes=None,     # Use all cores
    )
    
    print(f"Configuration:")
    print(f"  Policy:      {cfg.policy}")
    print(f"  Pop size:    {cfg.pop_size}")
    print(f"  Generations: {cfg.generations}")
    print(f"  Episodes:    {cfg.episodes}")
    print()
    
    # Run evolution
    winner, history = run_ga(
        env_ctor=TinyGrid,
        env_kwargs={
            "max_steps": 200,
            "size": 7,
            "difficulty": "medium",
        },
        cfg=cfg,
    )
    
    # Save results
    os.makedirs("artifacts_test", exist_ok=True)
    
    policy_path = "artifacts_test/test_graph_policy.npz"
    save_policy_npz(winner, policy_path)
    
    hist_path = "artifacts_test/test_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print()
    print("=" * 60)
    print("✅ Test complete!")
    print(f"   Policy saved: {policy_path}")
    print(f"   History:      {hist_path}")
    
    # Print final stats
    final = history[-1]
    print()
    print(f"📈 Final Generation ({final['gen']}):")
    print(f"   Best fitness:    {final['best_f']:+.3f}")
    print(f"   Best reward:     {final['best_r']:+.3f}")
    print(f"   Best success:    {final['best_s']:.2%}")
    print(f"   Mean fitness:    {final['mean_f']:+.3f}")
    print(f"   Mean success:    {final['mean_s']:.2%}")


if __name__ == "__main__":
    main()
