# experiments/visualize_policy.py

import argparse
import time
import numpy as np

from tasks.tinygrid import TinyGrid
from eng.io_policies import load_policy_npz  # unified loader for linear + mlp


def render_ascii(env: TinyGrid):
    """
    Simple ASCII render of the 7x7 grid.
    """
    chars = {
        0: ".",  # empty
        1: "#",  # wall
        2: "k",  # key
        3: "D",  # door
        4: "G",  # goal
    }

    lines = []
    for r in range(env.size):
        row = []
        for c in range(env.size):
            if (r, c) == env.agent:
                row.append("A")
            else:
                tile = int(env.grid[r, c])
                row.append(chars.get(tile, "?"))
        lines.append("".join(row))
    print("\n".join(lines))


def run_episode(policy_path: str, seed: int, max_steps: int, render: str, pause: float):
    # 1) Load policy via unified loader (handles linear + mlp)
    policy = load_policy_npz(policy_path)

    # 2) Create env
    env = TinyGrid(max_steps=max_steps)
    obs = env.reset(seed=seed)
    total = 0.0

    for t in range(max_steps):
        # 3) Get logits and pick greedy action
        logits = policy(obs)          # shape (5,)
        action = int(np.argmax(logits))

        # 4) Step env
        obs, reward, done, _info = env.step(action)
        total += reward

        # 5) Render if requested
        if render == "ascii":
            print()
            render_ascii(env)
            hud_parts = [
                f"t={env.t} has_key={env.has_key} used_key={env.used_key} "
                f"orient={env.orientation}",
                f"step={env.t} reward={reward:+.3f} total={total:+.3f}",
            ]
            if reward <= -0.49 and not env.used_key and env.agent == env.goal_pos:
                hud_parts.append("[penalized: goal before door]")
            print("  ".join(hud_parts))

            if pause > 0.0:
                time.sleep(pause)

        if done:
            break

    print(f"Episode finished in {env.t} steps with total reward {total:+.3f}")


def main():
    ap = argparse.ArgumentParser(description="Visualize TinyGrid policy behavior")
    ap.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to .npz weights (e.g. artifacts/best_mlp_policy.npz)",
    )
    ap.add_argument("--seed", type=int, default=0, help="Environment seed")
    ap.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    ap.add_argument(
        "--render",
        type=str,
        default="ascii",
        choices=["ascii", "none"],
        help="Render mode",
    )
    ap.add_argument(
        "--pause",
        type=float,
        default=0.1,
        help="Delay between frames (seconds) when rendering ascii",
    )
    args = ap.parse_args()

    run_episode(args.policy, args.seed, args.max_steps, args.render, args.pause)


if __name__ == "__main__":
    main()