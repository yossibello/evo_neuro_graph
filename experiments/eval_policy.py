# experiments/eval_policy.py

import argparse
import numpy as np

from tasks.tinygrid import TinyGrid
from eng.io_policies import load_policy_npz


def run_episode(policy,
                seed: int,
                max_steps: int,
                size: int = 7,
                difficulty: str = "medium"):
    """
    Run one TinyGrid episode and return (total_reward, success_flag).
    Success = reached goal having used the key to open the door
              (env.used_key == True and env.agent == env.goal_pos).
    """
    env = TinyGrid(size=size, max_steps=max_steps, difficulty=difficulty)
    obs = env.reset(seed=seed)

    total = 0.0
    success = False

    for t in range(max_steps):
        logits = policy(obs)
        action = int(np.argmax(logits))
        obs, reward, done, _info = env.step(action)
        total += reward

        if done:
            if env.used_key and env.agent == env.goal_pos:
                success = True
            break

    return total, success


def main():
    ap = argparse.ArgumentParser(description="Evaluate a TinyGrid policy over many seeds.")
    ap.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to .npz policy file (e.g. artifacts/best_graph_policy.npz)",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        default=50,
        help="Number of different env seeds to evaluate.",
    )
    ap.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Max steps per episode.",
    )
    ap.add_argument(
        "--size",
        type=int,
        default=7,
        help="Grid size (must match what you trained on).",
    )
    ap.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="TinyGrid difficulty level",
    )
    args = ap.parse_args()

    policy = load_policy_npz(args.policy)

    rewards = []
    successes = 0
    base_seed = 0

    for i in range(args.seeds):
        seed = base_seed + i
        R, ok = run_episode(
            policy,
            seed=seed,
            max_steps=args.max_steps,
            size=args.size,
            difficulty=args.difficulty,
        )
        rewards.append(R)
        if ok:
            successes += 1

    success_rate = successes / max(1, args.seeds)
    avg_reward = float(np.mean(rewards)) if rewards else 0.0

    print(f"✅ Evaluated {args.seeds} episodes")
    print(f"Success rate: {success_rate * 100:.1f}%")
    print(f"Average reward: {avg_reward:+.3f}")


if __name__ == "__main__":
    main()