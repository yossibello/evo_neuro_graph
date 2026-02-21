#!/usr/bin/env python3
"""
Comprehensive evaluation suite for all policy types.
Compares Linear, MLP, and Graph policies on multiple metrics.
"""

import argparse
import os
import sys
import json
import numpy as np
from typing import Dict, List, Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tasks.tinygrid import TinyGrid
from eng.io_policies import load_policy_npz
from eng.evolve import select_action


def evaluate_policy_detailed(
    policy,
    env_kwargs: Dict[str, Any],
    num_episodes: int = 100,
    max_steps: int = 200,
    seed_base: int = 42,
) -> Dict[str, Any]:
    """
    Deep evaluation with multiple metrics:
    - Success rate (strict: key + door + goal)
    - Goal reach rate (regardless of key usage)
    - Key pickup rate
    - Door opening rate
    - Average reward
    - Average steps to success
    - Action distribution
    """
    
    rng = np.random.RandomState(seed_base)
    
    total_reward = 0.0
    strict_successes = 0
    goal_hits = 0
    key_pickups = 0
    door_opens = 0
    steps_to_success = []
    action_counts = np.zeros(5, dtype=np.int64)
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        env = TinyGrid(**env_kwargs)
        obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        
        ep_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = select_action(policy, obs, rng, eps=0.0)  # greedy
            action_counts[action] += 1
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
        
        total_reward += ep_reward
        episode_rewards.append(ep_reward)
        
        # End-of-episode stats
        at_goal = (env.agent == env.goal_pos)
        has_key = env.has_key or env.used_key
        used_key = env.used_key
        
        if at_goal:
            goal_hits += 1
            if used_key:
                strict_successes += 1
                steps_to_success.append(steps)
        
        if has_key:
            key_pickups += 1
        if used_key:
            door_opens += 1
    
    # Calculate metrics
    avg_reward = total_reward / num_episodes
    strict_sr = strict_successes / num_episodes
    goal_rate = goal_hits / num_episodes
    key_rate = key_pickups / num_episodes
    door_rate = door_opens / num_episodes
    
    avg_steps_success = np.mean(steps_to_success) if steps_to_success else None
    action_distribution = (action_counts / action_counts.sum()).tolist()
    
    return {
        "avg_reward": float(avg_reward),
        "strict_success_rate": float(strict_sr),
        "goal_reach_rate": float(goal_rate),
        "key_pickup_rate": float(key_rate),
        "door_open_rate": float(door_rate),
        "num_successes": strict_successes,
        "avg_steps_to_success": float(avg_steps_success) if avg_steps_success else None,
        "action_distribution": action_distribution,
        "reward_std": float(np.std(episode_rewards)),
        "reward_min": float(np.min(episode_rewards)),
        "reward_max": float(np.max(episode_rewards)),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate all policy types comprehensively")
    ap.add_argument("--linear", type=str, default="artifacts/best_linear_policy.npz")
    ap.add_argument("--mlp", type=str, default="artifacts/best_mlp_policy.npz")
    ap.add_argument("--graph", type=str, default="artifacts/best_graph_policy.npz")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--size", type=int, default=7)
    ap.add_argument("--out", type=str, default="artifacts/evaluation_results.json")
    args = ap.parse_args()
    
    env_kwargs = {
        "max_steps": args.max_steps,
        "size": args.size,
        "difficulty": args.difficulty,
    }
    
    results = {
        "config": {
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "difficulty": args.difficulty,
            "size": args.size,
        },
        "policies": {}
    }
    
    policies_to_test = [
        ("linear", args.linear),
        ("mlp", args.mlp),
        ("graph", args.graph),
    ]
    
    for policy_name, policy_path in policies_to_test:
        if not os.path.exists(policy_path):
            print(f"⚠️  {policy_name.upper()} policy not found at {policy_path}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {policy_name.upper()} policy...")
        print(f"{'='*60}")
        
        try:
            policy = load_policy_npz(policy_path)
            metrics = evaluate_policy_detailed(
                policy,
                env_kwargs=env_kwargs,
                num_episodes=args.episodes,
                max_steps=args.max_steps,
            )
            
            results["policies"][policy_name] = metrics
            
            # Print summary
            print(f"\n📊 {policy_name.upper()} Results:")
            print(f"  Strict Success Rate:    {metrics['strict_success_rate']*100:6.2f}%")
            print(f"  Goal Reach Rate:        {metrics['goal_reach_rate']*100:6.2f}%")
            print(f"  Door Open Rate:         {metrics['door_open_rate']*100:6.2f}%")
            print(f"  Key Pickup Rate:        {metrics['key_pickup_rate']*100:6.2f}%")
            print(f"  Average Reward:         {metrics['avg_reward']:+8.3f} (σ={metrics['reward_std']:.3f})")
            print(f"  Reward Range:           [{metrics['reward_min']:+.2f}, {metrics['reward_max']:+.2f}]")
            if metrics['avg_steps_to_success']:
                print(f"  Avg Steps to Success:   {metrics['avg_steps_to_success']:7.1f}")
            print(f"  Action Distribution:    {[f'{x:.3f}' for x in metrics['action_distribution']]}")
            
        except Exception as e:
            print(f"❌ Error evaluating {policy_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Results saved to {args.out}")
    print(f"{'='*60}")
    
    # Comparative analysis
    if len(results["policies"]) > 1:
        print(f"\n🔬 COMPARATIVE ANALYSIS:")
        print(f"\nPolicy Rankings by Strict Success Rate:")
        sorted_policies = sorted(
            results["policies"].items(),
            key=lambda x: x[1]["strict_success_rate"],
            reverse=True
        )
        for rank, (name, metrics) in enumerate(sorted_policies, 1):
            print(f"  {rank}. {name.upper():8s}: {metrics['strict_success_rate']*100:6.2f}%")


if __name__ == "__main__":
    main()
