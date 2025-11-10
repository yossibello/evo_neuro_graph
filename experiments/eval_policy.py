import argparse, os, sys, numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tasks.tinygrid import TinyGrid
from eng.policies import LinearPolicy

try:
    from eng.policies_mlp import MLPPolicy
except Exception:
    MLPPolicy = None


def load_policy(path: str):
    """Load linear or MLP policy from npz file."""
    data = np.load(path)
    kind = str(data.get("kind", "linear"))
    if kind == "mlp" and MLPPolicy is not None:
        params = [data["W1"], data["b1"], data["W2"], data["b2"], data["W3"], data["b3"]]
        return MLPPolicy(params)
    else:
        return LinearPolicy(data["W"], data["b"])


def eval_many(policy, seeds=50, max_steps=300):
    """Run multiple episodes to estimate success rate and average reward."""
    wins = 0
    total_reward = 0.0

    for s in range(seeds):
        env = TinyGrid(max_steps=max_steps)
        obs = env.reset(seed=s)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < max_steps:
            action = policy.act(obs, explore=False)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            steps += 1

        if ep_reward >= 0.99:  # considered success
            wins += 1
        total_reward += ep_reward

    success_rate = wins / seeds
    avg_reward = total_reward / seeds
    return success_rate, avg_reward


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained TinyGrid policy.")
    ap.add_argument("--policy", required=True, help="Path to the policy npz file")
    ap.add_argument("--seeds", type=int, default=50, help="Number of random seeds to test")
    ap.add_argument("--max_steps", type=int, default=300)
    args = ap.parse_args()

    policy = load_policy(args.policy)
    success_rate, avg_reward = eval_many(policy, seeds=args.seeds, max_steps=args.max_steps)

    print(f"✅ Evaluated {args.seeds} episodes")
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"Average reward: {avg_reward:+.3f}")


if __name__ == "__main__":
    main()
