# experiments/visualize_policy.py

import argparse
import time
import numpy as np

from eng.evolve import select_action
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

def run_episode(policy_path: str, seed: int, max_steps: int, size: int, difficulty: str, render: str, pause: float):
    from eng.io_policies import load_policy_npz
    from tasks.tinygrid import TinyGrid
    import time

    # Load ANY policy (linear / mlp / graph)
    policy = load_policy_npz(policy_path)

    env = TinyGrid(max_steps=max_steps, size=size, difficulty=difficulty)
    obs = env.reset(seed=seed)

    total_reward = 0.0

    for step in range(max_steps):
        action = select_action(policy, obs)   # <-- generic; linear/mlp/graph all define .act()
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if render == "ascii":
            env.render()
            time.sleep(pause)
        elif render == "none":
            pass
        else:
            raise ValueError("render must be 'ascii' or 'none'")

        if done:
            break

    print(
        f"Episode finished in {step+1} steps with total reward {total_reward:+.3f}"
    )


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
    ap.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="TinyGrid difficulty level"
    )
    ap.add_argument(
        "--size",
        type=int,
        default=7,
        help="Grid size"
    )
    args = ap.parse_args()

    run_episode(args.policy, args.seed, args.max_steps, args.size, args.difficulty, args.render, args.pause)

if __name__ == "__main__":
    main()