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




def run_episode(policy_path,
                seed: int,
                max_steps: int,
                size: int,
                difficulty: str,
                render: bool = True,
                pause: float = 0.1):
    # Load policy (linear / mlp / graph all supported)
    policy = load_policy_npz(policy_path)

    env = TinyGrid(size=size, max_steps=max_steps, difficulty=difficulty)
    obs = env.reset(seed=seed)
    total_reward = 0.0

    for t in range(max_steps):
        if render:
            env.render()
            print(
                f"t={t} has_key={env.has_key} used_key={env.used_key} "
                f"orient={env.orientation}"
            )
            if pause > 0:
                time.sleep(pause)

        # ---- GREEDY ACTION (no select_action here) ----
        logits = policy(obs)            # shape (5,)
        action = int(np.argmax(logits)) # 0..4

        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(f"Episode finished in {t+1} steps with total reward {total_reward:+.3f}")
            return

    print(f"Episode finished in {max_steps} steps with total reward {total_reward:+.3f}")

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