
import argparse, os, time
import numpy as np
import matplotlib.pyplot as plt


from tasks.tinygrid import TinyGrid
from eng.policies import LinearPolicy
try:
    from eng.policies_mlp import MLPPolicy
    HAS_MLP = True
except Exception:
    HAS_MLP = False

def load_policy(policy_path: str):
    data = np.load(policy_path)
    kind = str(data.get("kind", "linear"))
    if kind == "mlp":
        params = [data["W1"], data["b1"], data["W2"], data["b2"], data["W3"], data["b3"]]
        return MLPPolicy(params)
    else:
        return LinearPolicy(data["W"], data["b"])



TILE_EMPTY=0; TILE_WALL=1; TILE_KEY=2; TILE_DOOR=3; TILE_GOAL=4; TILE_AGENT=5

CHARS = {
    TILE_EMPTY: ".",
    TILE_WALL:  "#",
    TILE_KEY:   "k",
    TILE_DOOR:  "D",
    TILE_GOAL:  "G",
    TILE_AGENT: "A",
}

def ascii_render(env: TinyGrid):
    g = env.grid.copy()
    r,c = env.agent
    g[r,c] = TILE_AGENT
    lines = []
    for i in range(env.size):
        line = "".join(CHARS[int(x)] for x in g[i,:])
        lines.append(line)
    hud = f"t={env.t} has_key={env.has_key} used_key={env.used_key} orient={env.orientation}"
    return "\n".join(lines + [hud])

def to_rgb(grid2d):
    # Map tile IDs to simple RGB triples for plotting
    # empty:[1,1,1], wall:[0.2,0.2,0.2], key:[1,0.9,0.2], door:[0.6,0.4,0.1], goal:[0.2,0.9,0.2], agent:[0.2,0.4,1.0]
    color_map = {
        TILE_EMPTY: (1.0, 1.0, 1.0),
        TILE_WALL:  (0.2, 0.2, 0.2),
        TILE_KEY:   (1.0, 0.9, 0.2),
        TILE_DOOR:  (0.6, 0.4, 0.1),
        TILE_GOAL:  (0.2, 0.9, 0.2),
        TILE_AGENT: (0.2, 0.4, 1.0),
    }
    H,W = grid2d.shape
    rgb = np.zeros((H,W,3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            rgb[i,j] = color_map[int(grid2d[i,j])]
    return rgb

def run_episode(policy_path: str, seed: int, max_steps: int, render: str, pause: float):
    data = np.load(policy_path)
    policy = load_policy(policy_path)

    env = TinyGrid(max_steps=max_steps)
    obs = env.reset(seed=seed)
    total = 0.0

    if render == "matplotlib":
        plt.ion()
        fig = plt.figure()
        # initial frame
        g = env.grid.copy()
        ar, ac = env.agent
        g[ar,ac] = TILE_AGENT
        img = to_rgb(g)
        im = plt.imshow(img, interpolation="nearest")
        txt = plt.title(f"t={env.t} reward=0.00 has_key={env.has_key} used_key={env.used_key}")
        plt.axis("off")
        plt.show(block=False)

    for t in range(max_steps):
        # choose action (replace with your own policy/scheduler lines if needed)
        action = policy.act(obs, explore=False)
        z = policy.logits(obs)  # optional; comment out if your policy lacks this

        obs, reward, done, info = env.step(action)
        total += reward

        # 1) build base HUD FIRST
        hud_parts = [
            f"t={env.t} reward={reward:+.3f} total={total:+.3f} "
            f"has_key={env.has_key} used_key={env.used_key}"
        ]

        # 2) add annotations AFTER base HUD creation
        if reward <= -0.49 and (not env.used_key) and (env.agent == env.goal_pos):
            hud_parts.append("[penalized: goal before door]")

        # (optional) show action/logits
        # hud_parts.append(f"action={action} logits={np.round(z,2)}")

        hud = "  ".join(hud_parts)

        # (optional) ASCII render
        print(ascii_render(env)); print(hud)

        if done:
            break


    if render == "matplotlib":
        plt.ioff()
        plt.show()

    print(f"Episode finished in {t+1} steps with total reward {total:+.3f}")

def main():
    ap = argparse.ArgumentParser(description="Visualize TinyGrid policy behavior")
    ap.add_argument("--policy", type=str, default="artifacts/best_linear_policy.npz", help="Path to .npz weights")
    ap.add_argument("--seed", type=int, default=0, help="Environment seed")
    ap.add_argument("--max_steps", type=int, default=128, help="Max steps per episode")
    ap.add_argument("--render", type=str, default="ascii", choices=["ascii","matplotlib"], help="Render mode")
    ap.add_argument("--pause", type=float, default=0.1, help="Delay between frames (seconds)")
    args = ap.parse_args()

    run_episode(args.policy, args.seed, args.max_steps, args.render, args.pause)

if __name__ == "__main__":
    main()
