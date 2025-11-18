# experiments/eval_policy.py

import argparse
import numpy as np

from tasks.tinygrid import TinyGrid
from eng.io_policies import load_policy_npz   # unified loader


def run_episode(policy, seed: int, max_steps: int):
    """
    Run one episode of TinyGrid with a given policy and seed.
    Returns (total_reward, success_flag).
    Success = reached goal after using key/door in the proper order.
    """
    env = TinyGrid(max_steps=max_steps)
    obs = env.reset(seed=seed)
    total = 0.0
    success = False

    for t in range(max_steps):
        logits = policy(obs)          # shape (5,)
        action = int(np.argmax(logits))
        obs, reward, done, _info = env.step(action)
        total += reward

        if done:
            # Strict success: must have used key and be on goal tile
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
        help="Path to .npz policy file (e.g. artifacts/best_mlp_policy.npz)",
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
    args = ap.parse_args()

    # Load policy (handles linear & mlp)
    policy = load_policy_npz(args.policy)

    rewards = []
    successes = 0

    base_seed = 0
    for i in range(args.seeds):
        seed = base_seed + i
        R, ok = run_episode(policy, seed, args.max_steps)
        rewards.append(R)
        if ok:
            successes += 1

    success_rate = successes / args.seeds
    avg_reward = float(np.mean(rewards)) if rewards else 0.0

    print(f"✅ Evaluated {args.seeds} episodes")
    print(f"Success rate: {success_rate * 100:.1f}%")
    print(f"Average reward: {avg_reward:+.3f}")


if __name__ == "__main__":
    main()