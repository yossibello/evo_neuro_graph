
import argparse, sys, os, time
from statistics import mean
import numpy as np
from tqdm import trange

# local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from eng.core import GA, Scheduler, Genome
from tasks.tinygrid import TinyGrid, encode_packets

def run_random_baseline(episodes=10, max_steps=128, seed=0):
    ga = GA(cfg={})
    avg_reward = ga.evaluate_random(lambda: TinyGrid(max_steps=max_steps), episodes, max_steps, seed)
    return avg_reward

def main():
    parser = argparse.ArgumentParser(description="Run TinyGrid experiment (scaffold).")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes for baseline eval")
    parser.add_argument("--max_steps", type=int, default=128, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    avg_reward = run_random_baseline(episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)
    print(f"[Random policy] Avg reward over {args.episodes} eps: {avg_reward:.3f}")

    env = TinyGrid(max_steps=args.max_steps)
    obs = env.reset(seed=args.seed)
    genome = Genome()  # empty genome for now
    sched = Scheduler(genome, seeds={})
    total = 0.0
    for t in range(args.max_steps):
        packets = encode_packets(obs, K=8)
        logits = sched.run_tick(packets)  # returns 5 action scores
        action = int(np.argmax(logits))
        obs, reward, done, info = env.step(action)
        total += reward

        # 1) build base HUD
        hud_parts = [
            f"t={env.t} reward={reward:+.3f} total={total:+.3f} has_key={env.has_key} used_key={env.used_key}"
        ]

        # 2) add annotations
        if reward <= -0.49 and not env.used_key and env.agent == env.goal_pos:
            hud_parts.append("[penalized: goal before door]")
        
        hud = "  ".join(hud_parts)
        if done:
            break
    print(f"[Scheduler smoke test] Episode reward: {total:.3f}, steps: {t+1}")

if __name__ == "__main__":
    main()
