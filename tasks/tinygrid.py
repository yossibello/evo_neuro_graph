# tasks/tinygrid.py
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import random

# --- add near top of file ---
from collections import deque

import json
import os
from typing import Any, Dict

OBS_DIM = 61  # keep this in sync with _encode_obs()

DEFAULT_REWARD_CONFIG: Dict[str, Any] = {
    "step_penalty": -0.01,

    "reward_pick_key": 0.2,
    "reward_open_door": 0.3,
    "reward_reach_goal": 10.0,
    "penalty_goal_without_door": -0.5,

    "bonus_near_door": 0.02,
    "bonus_near_goal": 0.02,

    "bonus_new_tile": 0.01,

    "stall_mild_steps": 4,
    "stall_mild_penalty": -0.03,

    "stall_hard_steps": 10,
    "stall_hard_penalty": -1.3,
}


def load_reward_config(path: str | None) -> Dict[str, Any]:
    """
    Load reward config from JSON file if provided.
    Missing keys fall back to DEFAULT_REWARD_CONFIG.
    """
    cfg = DEFAULT_REWARD_CONFIG.copy()
    if path is None:
        return cfg

    if not os.path.exists(path):
        print(f"[TinyGrid] Warning: reward config '{path}' not found, using defaults.")
        return cfg

    try:
        with open(path, "r") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    except Exception as e:
        print(f"[TinyGrid] Error loading reward config '{path}': {e}")
        print("[TinyGrid] Falling back to defaults.")
    return cfg



def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def _bfs_path_exists(grid, start, goal, passable):
    """Simple BFS on positions using passable(rr,cc)->bool. Returns True if a path exists."""
    H, W = grid.shape
    q = deque([start])
    seen = {start}
    while q:
        r,c = q.popleft()
        if (r,c) == goal: return True
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr,cc = r+dr, c+dc
            if 0 <= rr < H and 0 <= cc < W and (rr,cc) not in seen and passable(rr,cc):
                seen.add((rr,cc)); q.append((rr,cc))
    return False

from collections import deque
def _bfs(grid, start, goal, passable):
    H,W = grid.shape
    q = deque([start]); seen={start}
    while q:
        r,c = q.popleft()
        if (r,c)==goal: return True
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr,cc = r+dr, c+dc
            if 0<=rr<H and 0<=cc<W and (rr,cc) not in seen and passable(rr,cc):
                seen.add((rr,cc)); q.append((rr,cc))
    return False


# Tile IDs
TILE_EMPTY = 0
TILE_WALL  = 1
TILE_KEY   = 2
TILE_DOOR  = 3
TILE_GOAL  = 4
TILE_AGENT = 5

class TinyGrid:
    """
    7x7 grid with outer walls plus a few random inner walls, exactly one key, one door, one goal.
    Observations: 61-d vector = (3x3 egocentric one-hot over 6 tiles -> 54) + (has_key, used_key -> 2)
                  + (orientation one-hot N/E/S/W -> 4) + (step budget -> 1)
    Actions: 0=up, 1=right, 2=down, 3=left, 4=toggle (no-op kept for compatibility)
    Rewards: -0.01 per step, +0.2 pick up key, +0.2 open door, +1.0 reach goal (ONLY if door was opened),
             otherwise stepping onto goal early gives -0.5 and ends episode.
    Strict rule is ALWAYS ON here (no flag) per your request.
    """


    def __init__(
        self,
        size: int = 7,
        max_steps: int = 128,
        difficulty: str = "medium",
    ):
        self.size = size
        self.max_steps = max_steps
        self.difficulty = difficulty

        # Difficulty → number of inner walls
        if self.difficulty == "easy":
            self.num_walls = 0
        elif self.difficulty == "hard":
            self.num_walls = 8
        else:  # "medium" / default
            self.num_walls = 5

        # Stall / exploration tracking
        self.last_pos = None
        self.stall_steps = 0
        self.visit_counts = None 

        self.rng = random.Random(0)

        # These will be created in reset()
        self.grid = None
        self.visited = None

        self.reset()
    # --------------------------
    # Public API
    # --------------------------
        # --------------------------
    # ASCII render (for visualize_policy)
    # --------------------------
    def render(self):
        """
        Print an ASCII view of the grid:
        . = empty
        # = wall
        k = key
        D = door
        G = goal
        A = agent
        """
        # map tile IDs→ chars
        char_map = {
            TILE_EMPTY: ".",
            TILE_WALL:  "#",
            TILE_KEY:   "k",
            TILE_DOOR:  "D",
            TILE_GOAL:  "G",
        }

        lines = []
        for r in range(self.size):
            row_chars = []
            for c in range(self.size):
                if (r, c) == self.agent:
                    row_chars.append("A")
                else:
                    t = int(self.grid[r, c])
                    row_chars.append(char_map.get(t, "?"))
            lines.append("".join(row_chars))

        print("\n".join(lines))
        print(
            f"t={self.t} has_key={self.has_key} used_key={self.used_key} "
            f"orient={self.orientation}"
        )

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

        max_tries = 10_000
        tries = 0

        while True:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(
                    f"TinyGrid.reset: failed to sample valid layout after "
                    f"{max_tries} attempts (difficulty={getattr(self, 'difficulty', 'unknown')}, "
                    f"size={self.size})"
                )

            # fresh episode state
            self.t = 0
            self.has_key = False
            self.used_key = False
            self.orientation = 0

            # build grid
            self.grid = np.zeros((self.size, self.size), dtype=np.int32)
            self.grid[0, :]  = 1
            self.grid[-1, :] = 1
            self.grid[:, 0]  = 1
            self.grid[:, -1] = 1

            # inner walls (use difficulty / num_walls if available)
            num_walls = getattr(self, "num_walls", 5)
            for _ in range(num_walls):
                r = self.rng.randrange(1, self.size - 1)
                c = self.rng.randrange(1, self.size - 1)
                self.grid[r, c] = TILE_WALL

            def place(tile_id):
                while True:
                    rr = self.rng.randrange(1, self.size - 1)
                    cc = self.rng.randrange(1, self.size - 1)
                    if self.grid[rr, cc] == 0:
                        self.grid[rr, cc] = tile_id
                        return (rr, cc)

            self.key_pos  = place(TILE_KEY)
            self.door_pos = place(TILE_DOOR)
            self.goal_pos = place(TILE_GOAL)

            # agent
            while True:
                ar = self.rng.randrange(1, self.size - 1)
                ac = self.rng.randrange(1, self.size - 1)
                if self.grid[ar, ac] == 0:
                    self.agent = (ar, ac)
                    break

            g = self.grid

            # A->key with door CLOSED
            def pass_to_key(rr, cc):
                t = g[rr, cc]
                return t != TILE_WALL and t != TILE_DOOR

            ok1 = _bfs(g, self.agent, self.key_pos, pass_to_key)

            # key->door (door becomes target; walls block)
            def pass_open(rr, cc):
                t = g[rr, cc]
                return t != TILE_WALL

            ok2 = _bfs(g, self.key_pos, self.door_pos, pass_open)

            # door->goal (after opening door; walls block)
            ok3 = _bfs(g, self.door_pos, self.goal_pos, pass_open)

            # forbid A->G path with door CLOSED
            def pass_block_door(rr, cc):
                t = g[rr, cc]
                return t != TILE_WALL and t != TILE_DOOR

            forbid_skip = _bfs(g, self.agent, self.goal_pos, pass_block_door)

            # -------------------------------
            # Difficulty-aware acceptance
            # -------------------------------
            if ok1 and ok2 and ok3:
                diff = getattr(self, "difficulty", "medium")
                if diff == "easy":
                    # On easy: allow layouts even if agent *could* reach goal
                    # without going through door first.
                    break
                else:
                    # On medium / hard: require door to be a true chokepoint.
                    if not forbid_skip:
                        break
            # else: resample layout

        # auto-pickup if spawned on key (no reward at reset)
        if self.agent == self.key_pos and not self.has_key:
            self.has_key = True
            self.grid[self.key_pos] = TILE_EMPTY

        # --------------------------------
        # Exploration / stall state
        # --------------------------------
        H, W = self.grid.shape

        # track which tiles have ever been seen (for +0.01 exploration bonus)
        if (not hasattr(self, "visited")) or (self.visited is None) or (self.visited.shape != (H, W)):
            self.visited = np.zeros((H, W), dtype=bool)
        else:
            self.visited[:, :] = False

        # track how many times we've stepped on each tile (for loop penalty)
        if (not hasattr(self, "visit_counts")) or (self.visit_counts is None) or (self.visit_counts.shape != (H, W)):
            self.visit_counts = np.zeros((H, W), dtype=np.int32)
        else:
            self.visit_counts[:, :] = 0

        r, c = self.agent
        self.visited[r, c] = True
        self.visit_counts[r, c] = 1

        self.last_pos = self.agent
        self.stall_steps = 0

        return self._encode_obs()


    def step(self, action: int):
        reward = -0.01   # base step penalty
        done = False

        old_r, old_c = self.agent

        # -----------------------------------
        # Movement
        # -----------------------------------
        if action in (0, 1, 2, 3):
            self.orientation = action
            dr, dc = [(-1,0),(0,1),(1,0),(0,-1)][action]
            nr, nc = old_r + dr, old_c + dc
            tile = self.grid[nr, nc]

            if tile == TILE_WALL:
                pass

            elif tile == TILE_DOOR:
                if self.has_key and not self.used_key:
                    self.used_key = True
                    reward += 6.0
                    self.grid[nr, nc] = TILE_EMPTY
                    self.agent = (nr, nc)
                else:
                    reward -= 0.2
            
            else:
                self.agent = (nr, nc)

        # -----------------------------------
        # Distance shaping
        # -----------------------------------
        new_r, new_c = self.agent

        def dist(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

        # Move toward key
        if not self.has_key:
            if dist((new_r,new_c), self.key_pos) < dist((old_r,old_c), self.key_pos):
                reward += 0.03

        # Move toward door
        if self.has_key and not self.used_key:
            if dist((new_r,new_c), self.door_pos) < dist((old_r,old_c), self.door_pos):
                reward += 0.1

        # Move toward goal
        if self.used_key:
            if dist((new_r,new_c), self.goal_pos) < dist((old_r,old_c), self.goal_pos):
                reward += 0.1

        # -----------------------------------
        # Key pickup
        # -----------------------------------
        if self.agent == self.key_pos and not self.has_key:
            self.has_key = True
            reward += 3
            self.grid[self.key_pos] = TILE_EMPTY

        # -----------------------------------
        # Door/Goal proximity bonuses
        # -----------------------------------
        if self.has_key and not self.used_key:
            if dist(self.agent, self.door_pos) == 1:
                reward += 0.02

        if self.used_key:
            if dist(self.agent, self.goal_pos) == 1:
                reward += 0.02

        # -----------------------------------
        # Goal logic
        # -----------------------------------
        if self.agent == self.goal_pos:
            if self.used_key:
                reward += 20.0
            else:
                reward -= 0.5    # strong punishment
            return self._encode_obs(), float(reward), True, {
                "has_key": self.has_key,
                "used_key": self.used_key
            }

        # --------------------------
        # Exploration bonus
        # --------------------------
        rr, cc = self.agent
        if not self.visited[rr, cc]:
            self.visited[rr, cc] = True
            reward += 0.05

        # Count visits to this tile (penalize loops like A-B-A-B-...)
        self.visit_counts[rr, cc] += 1
        if self.visit_counts[rr, cc] > 4:
            reward -= 0.01   # small penalty for over-visiting same tile

        # -----------------------------------
        # Stall logic (camping on same tile)
        # -----------------------------------
        if self.agent == self.last_pos:
            self.stall_steps += 1
        else:
            self.stall_steps = 0
            self.last_pos = self.agent

        # Mild bounce penalty (true stall, not A<->B)
        if self.stall_steps > 3:
            reward -= 0.05   # stronger

        # Hard reset bounce
        if self.stall_steps >= 10:
            reward -= 1.0
            self.stall_steps = 0

        # -----------------------------------
        # Time limit
        # -----------------------------------
        self.t += 1
        done = (self.t >= self.max_steps)

        return self._encode_obs(), float(reward), done, {
            "has_key": self.has_key,
            "used_key": self.used_key
        }


    # --------------------------
    # Encoding
    # --------------------------
    def _encode_obs(self) -> np.ndarray:
        """
        3x3 egocentric one-hot patch over 6 tile IDs (agent cell marked as TILE_AGENT),
        plus inventory, orientation, and step budget.
        """
        # 3x3 one-hot patch: 9 * 6 = 54
        patch = np.zeros((3, 3, 6), dtype=np.float32)
        ar, ac = self.agent
        for i, dr in enumerate((-1, 0, 1)):
            for j, dc in enumerate((-1, 0, 1)):
                rr, cc = ar + dr, ac + dc
                tile = TILE_WALL  # default out-of-bounds to wall (but we have outer walls anyway)
                if 0 <= rr < self.size and 0 <= cc < self.size:
                    tile = int(self.grid[rr, cc])
                if rr == ar and cc == ac:
                    tile = TILE_AGENT
                patch[i, j, tile] = 1.0
        patch_flat = patch.reshape(-1)

        # Inventory / orientation / step budget
        inv = np.array(
            [1.0 if self.has_key else 0.0, 1.0 if self.used_key else 0.0],
            dtype=np.float32,
        )
        orient = np.zeros(4, dtype=np.float32)
        orient[self.orientation] = 1.0
        step_budget = np.array(
            [1.0 - (self.t / max(1, self.max_steps))],
            dtype=np.float32,
        )

        obs = np.concatenate([patch_flat, inv, orient, step_budget]).astype(np.float32)
        assert obs.shape[0] == OBS_DIM, f"Expected obs dim {OBS_DIM}, got {obs.shape[0]}"
        return obs


# --------------------------
# Optional packetizer (unchanged)
# --------------------------
def encode_packets(obs: np.ndarray, K: int = 8):
    """
    Splits the 61-d observation into small "packets" (for future message-passing / CPU-ish models).
    Returns a list of tuples (vector, channel_id).
    """
    assert obs.shape[0] == 61
    packets = []
    v = obs[0:54]
    for start in range(0, 54, K):
        packets.append((v[start:start + K], 1))  # local 3x3 patch chunks
    packets.append((obs[54:56], 2))             # inventory
    packets.append((obs[56:60], 3))             # orientation
    packets.append((obs[60:61], 4))             # step budget
    return packets
