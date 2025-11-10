# tasks/tinygrid.py
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import random

# --- add near top of file ---
from collections import deque

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

    def __init__(self, size: int = 7, max_steps: int = 128):
        self.size = size
        self.max_steps = max_steps
        self.rng = random.Random(0)
        self.reset()

    # --------------------------
    # Public API
    # --------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

        while True:
            # fresh state
            self.t=0; self.has_key=False; self.used_key=False; self.orientation=0
            self.grid = np.zeros((self.size,self.size), dtype=np.int32)
            self.grid[0,:]=1; self.grid[-1,:]=1; self.grid[:,0]=1; self.grid[:,-1]=1

            # a few inner walls (tune count; fewer = easier)
            for _ in range(5):
                r = self.rng.randrange(1,self.size-1)
                c = self.rng.randrange(1,self.size-1)
                self.grid[r,c]=1

            def place(tile_id):
                while True:
                    r = self.rng.randrange(1,self.size-1)
                    c = self.rng.randrange(1,self.size-1)
                    if self.grid[r,c]==0:
                        self.grid[r,c]=tile_id
                        return (r,c)

            self.key_pos  = place(TILE_KEY)
            self.door_pos = place(TILE_DOOR)
            self.goal_pos = place(TILE_GOAL)
            # agent
            while True:
                r = self.rng.randrange(1,self.size-1); c = self.rng.randrange(1,self.size-1)
                if self.grid[r,c]==0:
                    self.agent=(r,c); break

            g = self.grid
            # A->key with door CLOSED
            def pass_to_key(rr,cc):
                t=g[rr,cc]; return t!=TILE_WALL and t!=TILE_DOOR
            ok1 = _bfs(g, self.agent, self.key_pos, pass_to_key)

            # key->door (door becomes target; walls block)
            def pass_open(rr,cc):
                t=g[rr,cc]; return t!=TILE_WALL
            ok2 = _bfs(g, self.key_pos, self.door_pos, pass_open)

            # door->goal (after opening door; walls block)
            ok3 = _bfs(g, self.door_pos, self.goal_pos, pass_open)

            # forbid A->G path with door CLOSED (or agent will be tempted to skip)
            def pass_block_door(rr,cc):
                t=g[rr,cc]; return t!=TILE_WALL and t!=TILE_DOOR
            forbid_skip = _bfs(g, self.agent, self.goal_pos, pass_block_door)

            if ok1 and ok2 and ok3 and not forbid_skip:
                break
            # else resample

        # auto-pickup if spawned on key (no reward at reset)
        if self.agent == self.key_pos:
            self.has_key = True; self.grid[self.key_pos]=TILE_EMPTY
        return self._encode_obs()


    def step(self, action: int):
        reward = -0.01  # step penalty
        done = False

        r, c = self.agent

        # --------------------------
        # Movement (open door on entry if carrying key)
        # --------------------------
        if action in (0, 1, 2, 3):
            self.orientation = action
            dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
            nr, nc = r + dr, c + dc
            target = self.grid[nr, nc]

            if target == TILE_WALL:
                pass  # blocked
            elif target == TILE_DOOR:
                if self.has_key and not self.used_key:
                    # Open the door and move through
                    self.used_key = True
                    reward += 0.3
                    self.grid[nr, nc] = TILE_EMPTY
                    self.agent = (nr, nc)
                else:
                    pass  # locked door blocks
            else:
                # empty / key / goal — move is allowed
                self.agent = (nr, nc)

        elif action == 4:
            # toggle/use — kept for compatibility; not needed with auto-pickup/door-open
            # You can make this a no-op or add small behaviors later if desired.
            pass

        # near door (only after key)
        if self.has_key and not self.used_key:
            if abs(self.agent[0]-self.door_pos[0]) + abs(self.agent[1]-self.door_pos[1]) == 1:
                reward += 0.02

        # near goal (only after door)
        if self.used_key:
            if abs(self.agent[0]-self.goal_pos[0]) + abs(self.agent[1]-self.goal_pos[1]) == 1:
                reward += 0.02
        # --------------------------
        # Auto pick-up key (after movement)
        # --------------------------
        if self.agent == self.key_pos and not self.has_key:
            self.has_key = True
            reward += 0.3
            self.grid[self.key_pos] = TILE_EMPTY  # remove key from map

        # --------------------------
        # Strict Goal Logic
        # --------------------------
        if self.agent == self.goal_pos:
            if self.used_key:
                reward += 1.0
                done = True
            else:
                reward -= 0.5  # tried to skip the sequence

        # Time limit
        self.t += 1
        if self.t >= self.max_steps:
            done = True
        info = {"has_key": self.has_key, "used_key": self.used_key}
        return self._encode_obs(), float(reward), done, info

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

        obs = np.concatenate([patch_flat, inv, orient, step_budget])  # 54+2+4+1=61
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
