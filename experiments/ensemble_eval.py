import argparse
import glob
import os
from typing import List, Optional, Tuple

import numpy as np

# --- project imports ---
# TinyGrid env
from tasks.tinygrid import TinyGrid
# Policies (your files may be eng/policies.py and eng/policies_mlp.py)
try:
    from eng.policies_mlp import MLPPolicy  # expects .forward(obs)->logits
except ImportError:
    MLPPolicy = None

try:
    from eng.policies import LinearPolicy  # expects .forward(obs)->logits or logits = W@obs+b
except ImportError:
    LinearPolicy = None


def to_logits_fn(pol):
    """
    Return a callable logits_fn(obs)->np.ndarray shape (5,)
    that works no matter whether the policy exposes forward/__call__/act/predict/logits.
    If the policy returns an action index (scalar), convert to a one-hot logits vector.
    """
    # Try common method names in order
    for name in ("forward", "__call__", "logits", "act", "predict"):
        meth = getattr(pol, name, None)
        if callable(meth):
            def _fn(obs, _meth=meth):
                out = _meth(obs)
                # Normalize to numpy array
                if isinstance(out, np.ndarray):
                    arr = out
                else:
                    try:
                        arr = np.asarray(out)
                    except Exception:
                        arr = None
                # If it looks like logits (shape (5,)), return as-is
                if isinstance(arr, np.ndarray) and arr.shape == (5,):
                    return arr
                # If it looks like a scalar action index, convert to one-hot logits
                try:
                    a = int(out)
                    if 0 <= a < 5:
                        onehot = np.zeros(5, dtype=np.float32)
                        onehot[a] = 1.0
                        return onehot
                except Exception:
                    pass
                raise RuntimeError("Policy method did not return logits or action index.")
            return _fn
    raise RuntimeError("Policy has no usable method (forward/__call__/logits/act/predict).")


def load_policy_npz(path: str):
    """
    Load a saved policy .npz (MLP or Linear). Returns a callable logits_fn(obs)->(5,)
    """
    d = np.load(path, allow_pickle=True)  # allow_pickle in case 'kind' was saved as object
    keys = set(d.files)
    kind = d["kind"].item() if "kind" in keys else None

    # MLP saved as W1,b1,W2,b2,W3,b3 (with optional 'kind')
    is_mlp = (kind == "mlp") or {"W1", "b1", "W2", "b2", "W3", "b3"}.issubset(keys)
    # Linear saved as W,b
    is_linear = (kind == "linear") or {"W", "b"}.issubset(keys)

    if is_mlp:
        if MLPPolicy is None:
            raise RuntimeError("MLPPolicy not found. Ensure eng/policies_mlp.py is available.")
        W1, b1 = d["W1"], d["b1"]
        W2, b2 = d["W2"], d["b2"]
        W3, b3 = d["W3"], d["b3"]

        # Construct according to your repo’s constructor
        try:
            pol = MLPPolicy([W1, b1, W2, b2, W3, b3])
        except TypeError:
            # fallback: some repos expect layers or separate args
            pol = MLPPolicy(W1, b1, W2, b2, W3, b3)
        return to_logits_fn(pol)

    if is_linear:
        if LinearPolicy is None:
            raise RuntimeError("LinearPolicy not found. Ensure eng/policies.py is available.")
        W, b = d["W"], d["b"]
        try:
            pol = LinearPolicy(W, b)
            return to_logits_fn(pol)
        except Exception:
            # No class? Compute logits directly from arrays.
            def logits_fn(obs: np.ndarray) -> np.ndarray:
                return (W @ obs) + b
            return logits_fn

    # Fallback (linear arrays without 'kind')
    if {"W", "b"}.issubset(keys):
        W, b = d["W"], d["b"]
        def logits_fn(obs: np.ndarray) -> np.ndarray:
            return (W @ obs) + b
        return logits_fn

    raise ValueError(f"Unrecognized policy format in {path}; keys={sorted(keys)}")


def make_ensemble(checkpoint_paths: List[str], vote: str = "avg"):
    """
    Create an ensemble callable from a list of checkpoint paths.
    vote = 'avg'    -> average logits then argmax
    vote = 'majority' -> majority vote over individual argmax
    """
    if len(checkpoint_paths) == 0:
        raise ValueError("No checkpoint paths provided for ensemble.")

    members = [load_policy_npz(p) for p in checkpoint_paths]

    if vote == "avg":
        def act(obs: np.ndarray) -> int:
            # average logits across members
            logits = None
            for m in members:
                l = m(obs)
                logits = l if logits is None else (logits + l)
            logits = logits / len(members)
            return int(np.argmax(logits))
        return act

    elif vote == "majority":
        def act(obs: np.ndarray) -> int:
            votes = np.zeros(5, dtype=np.int32)
            for m in members:
                a = int(np.argmax(m(obs)))
                votes[a] += 1
            return int(np.argmax(votes))
        return act

    else:
        raise ValueError(f"Unknown vote mode: {vote}")


def run_episode(actor, seed: int, max_steps: int = 300) -> Tuple[float, bool]:
    """
    Run one TinyGrid episode with the given actor(obs)->action int.
    Returns (total_reward, success_flag).
    Success is True only if agent ends on goal *after* opening the door.
    """
    env = TinyGrid(max_steps=max_steps)
    obs = env.reset(seed=seed)
    total = 0.0
    success = False

    for _ in range(max_steps):
        # choose action greedily
        a = int(actor(obs))
        obs, r, done, _info = env.step(a)
        total += r
        if done:
            # strict success: reached goal AND door used
            success = (env.agent == env.goal_pos) and env.used_key
            break

    return total, success


def evaluate_actor(actor, seeds: int, max_steps: int) -> Tuple[float, float]:
    """Return (success_rate, avg_reward) over a number of random seeds."""
    succ = 0
    rewards = []
    base = 1000  # offset seeds to avoid edge cases at very small seeds
    for s in range(seeds):
        R, ok = run_episode(actor, base + s, max_steps)
        rewards.append(R)
        succ += 1 if ok else 0
    success_rate = succ / seeds
    avg_reward = float(np.mean(rewards))
    return success_rate, avg_reward


def main():
    ap = argparse.ArgumentParser(description="Evaluate single or ensemble policies on TinyGrid.")
    ap.add_argument("--policy", type=str, default=None,
                    help="Path to a single policy .npz (e.g., artifacts/best_mlp_policy.npz)")
    ap.add_argument("--checkpoints_glob", type=str, default=None,
                    help="Glob for ensemble checkpoints, e.g. 'artifacts/checkpoints/gen*_rank*.npz'")
    ap.add_argument("--top_k", type=int, default=5, help="Use top-K newest checkpoints from the glob")
    ap.add_argument("--vote", type=str, default="avg", choices=["avg", "majority"],
                    help="Ensemble fusion: average logits or majority voting over argmax")
    ap.add_argument("--seeds", type=int, default=100, help="Number of evaluation seeds")
    ap.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    args = ap.parse_args()

    # Evaluate single policy if provided
    if args.policy:
        print(f"→ Evaluating single policy: {args.policy}")
        single = load_policy_npz(args.policy)
        s, r = evaluate_actor(lambda obs: int(np.argmax(single(obs))), args.seeds, args.max_steps)
        print(f"[Single] Success rate: {s*100:.1f}% | Avg reward: {r:+.3f}")

    # Evaluate ensemble if provided
    if args.checkpoints_glob:
        all_paths = sorted(glob.glob(args.checkpoints_glob))
        if len(all_paths) == 0:
            print(f"⚠️  No checkpoints matched: {args.checkpoints_glob}")
            return
        # pick the newest top-K by filename order (adjust if you prefer best-by-fitness)
        paths = all_paths[-args.top_k:]
        print(f"→ Evaluating ensemble ({args.vote}) with {len(paths)} members:")
        for p in paths:
            print("   •", p)

        actor = make_ensemble(paths, vote=args.vote)
        s, r = evaluate_actor(actor, args.seeds, args.max_steps)
        print(f"[Ensemble-{args.vote}] Success rate: {s*100:.1f}% | Avg reward: {r:+.3f}")

    if not args.policy and not args.checkpoints_glob:
        print("Nothing to evaluate. Provide --policy and/or --checkpoints_glob.")
        return


if __name__ == "__main__":
    main()
