# eng/policies_graph.py
"""
Evolved computational graph — a register machine whose topology and weights
are optimised by a genetic algorithm.

Design principles
-----------------
1. **Multiple execution ticks** — information propagates across ticks (like
   recurrent "brain cycles").  Observation is re-injected each tick so inputs
   are never overwritten.
2. **Sigmoid gates** — every node is alive (soft gate in (0, 1)).  No binary
   dead/alive cliff that freezes evolution.
3. **Learnable weights in every op** — all 8 ops use an affine transform
   (w0·x0 + w1·x1 + bias), making the parameter space smooth for mutation.
4. **Vectorised forward pass** — all nodes fire simultaneously each tick with
   scatter-add to shared registers (like synaptic summation).
5. **Register clamping** — values clipped to [-10, 10] each tick to prevent
   numerical explosion.
"""

import numpy as np

NUM_OPS = 8  # number of distinct node operations


class GraphPolicy:
    """
    A policy that evolves a small "brain circuit" of N computational nodes
    operating on a shared register file.

    Register layout
    ---------------
    [0 .. obs_dim)           — observation input (re-injected each tick)
    [obs_dim .. action_base) — scratch / intermediate computation
    [action_base .. num_reg) — action logits (5 outputs)

    Node parameter row (8 columns)
    -------------------------------
    0: src0   — index of first source register
    1: src1   — index of second source register
    2: dst    — index of destination register
    3: op_id  — which of the 8 operations to apply
    4: w0     — weight on first source
    5: w1     — weight on second source
    6: bias   — additive bias
    7: gate   — sigmoid gate input (controls contribution strength)
    """

    def __init__(
        self,
        num_registers: int = 96,
        num_nodes: int = 64,
        num_ticks: int = 3,
        node_params: np.ndarray | None = None,
    ):
        self.num_registers = int(num_registers)
        self.num_ticks = int(num_ticks)
        self.n_actions = 5
        self.action_base = self.num_registers - self.n_actions

        if node_params is None:
            node_params = self._init_random(int(num_nodes))
        else:
            node_params = np.asarray(node_params, dtype=np.float32)

        self.node_params = node_params
        self.num_nodes = self.node_params.shape[0]

        # Shared register file
        self.registers = np.zeros(self.num_registers, dtype=np.float32)

        self._rebuild_cache()
        self._sync_params_list()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_random(self, N: int) -> np.ndarray:
        """
        Smart initial wiring:
          - ~10 backbone nodes:     obs -> action  (ensures non-trivial output)
          - ~10 intermediate nodes: obs -> scratch  (enables multi-tick processing)
          - rest:                   random -> random (diversity for evolution)
        All gates start positive so every node is alive from the start.
        """
        nr = self.num_registers
        obs_dim = 61  # TinyGrid observation dimension
        ab = self.action_base
        na = self.n_actions

        p = np.zeros((N, 8), dtype=np.float32)

        # --- Backbone: input -> action ---
        n_bb = min(10, N)
        for i in range(n_bb):
            p[i, 0] = np.random.randint(0, obs_dim)       # src0 from obs
            p[i, 1] = np.random.randint(0, obs_dim)       # src1 from obs
            p[i, 2] = ab + (i % na)                        # dst to action
            p[i, 3] = 1                                     # tanh op
            p[i, 4] = np.random.randn() * 0.5              # w0
            p[i, 5] = np.random.randn() * 0.5              # w1
            p[i, 6] = np.random.randn() * 0.1              # bias
            p[i, 7] = np.random.uniform(1.5, 3.0)          # gate (strongly on)

        # --- Intermediate: input -> scratch ---
        n_mid = min(10, N - n_bb)
        for ii in range(n_mid):
            i = n_bb + ii
            p[i, 0] = np.random.randint(0, obs_dim)
            p[i, 1] = np.random.randint(0, obs_dim)
            scratch_reg = np.random.randint(obs_dim, ab) if obs_dim < ab else 0
            p[i, 2] = scratch_reg
            p[i, 3] = np.random.randint(0, NUM_OPS)
            p[i, 4] = np.random.randn() * 0.3
            p[i, 5] = np.random.randn() * 0.3
            p[i, 6] = np.random.randn() * 0.1
            p[i, 7] = np.random.uniform(1.0, 2.5)

        # --- Random: any -> any ---
        for i in range(n_bb + n_mid, N):
            p[i, 0] = np.random.randint(0, nr)
            p[i, 1] = np.random.randint(0, nr)
            p[i, 2] = np.random.randint(0, nr)
            p[i, 3] = np.random.randint(0, NUM_OPS)
            p[i, 4] = np.random.randn() * 0.3
            p[i, 5] = np.random.randn() * 0.3
            p[i, 6] = np.random.randn() * 0.1
            p[i, 7] = np.random.uniform(0.5, 2.5)

        return p

    # ------------------------------------------------------------------
    # GA params API
    # ------------------------------------------------------------------
    def _sync_params_list(self):
        self._params = [self.node_params]

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self.node_params = np.asarray(new_params[0], dtype=np.float32)
        self.num_nodes = self.node_params.shape[0]
        self._rebuild_cache()
        self._sync_params_list()

    def clone(self):
        return GraphPolicy(
            num_registers=self.num_registers,
            num_nodes=self.num_nodes,
            num_ticks=self.num_ticks,
            node_params=self.node_params.copy(),
        )

    def as_dict(self):
        return {
            "kind": "graph",
            "graph_params": self.node_params,
            "num_registers": np.array(self.num_registers, dtype=np.int32),
            "num_ticks": np.array(self.num_ticks, dtype=np.int32),
        }

    # ------------------------------------------------------------------
    # Cache (integer indices + float weights extracted for fast forward)
    # ------------------------------------------------------------------
    def _rebuild_cache(self):
        p = self.node_params
        nr = self.num_registers
        self.src0 = np.clip(np.round(p[:, 0]).astype(np.int32), 0, nr - 1)
        self.src1 = np.clip(np.round(p[:, 1]).astype(np.int32), 0, nr - 1)
        self.dst  = np.clip(np.round(p[:, 2]).astype(np.int32), 0, nr - 1)
        self.op   = np.clip(np.round(p[:, 3]).astype(np.int32), 0, NUM_OPS - 1)
        self.w0   = p[:, 4].copy()
        self.w1   = p[:, 5].copy()
        self.bias = p[:, 6].copy()
        self.gate = p[:, 7].copy()

    # ------------------------------------------------------------------
    # Forward pass (vectorised — all nodes fire simultaneously per tick)
    #
    #   Ops (all use learned weights):
    #     0 = linear:      w0*x0 + w1*x1 + bias
    #     1 = tanh:        tanh(w0*x0 + w1*x1 + bias)
    #     2 = relu:        max(0, w0*x0 + w1*x1 + bias)
    #     3 = softsign:    z / (1 + |z|)
    #     4 = product:     (w0*x0) * (w1*x1) + bias
    #     5 = abs:         |w0*x0 + w1*x1 + bias|
    #     6 = weighted min: min(w0*x0, w1*x1) + bias
    #     7 = weighted max: max(w0*x0, w1*x1) + bias
    # ------------------------------------------------------------------
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        obs_dim = obs.shape[0]

        R = self.registers
        R.fill(0.0)

        N = self.num_nodes
        node_idx = np.arange(N)

        # Pre-compute sigmoid gates (constant across ticks)
        gate_val = 1.0 / (1.0 + np.exp(-np.clip(self.gate, -10.0, 10.0)))

        for _tick in range(self.num_ticks):
            # Re-inject observation every tick (sensory input refresh)
            R[:obs_dim] = obs

            # --- Gather source register values ---
            x0 = R[self.src0]           # (N,)
            x1 = R[self.src1]           # (N,)

            # --- Affine building blocks ---
            wx0 = self.w0 * x0          # (N,)
            wx1 = self.w1 * x1          # (N,)
            z   = wx0 + wx1 + self.bias

            # --- Compute all 8 ops for all nodes (vectorised) ---
            y_all = np.empty((NUM_OPS, N), dtype=np.float32)
            y_all[0] = z                                      # linear
            y_all[1] = np.tanh(z)                             # tanh
            y_all[2] = np.maximum(0.0, z)                     # relu
            y_all[3] = z / (1.0 + np.abs(z))                 # softsign
            y_all[4] = wx0 * wx1 + self.bias                 # product
            y_all[5] = np.abs(z)                              # abs
            y_all[6] = np.minimum(wx0, wx1) + self.bias      # weighted min
            y_all[7] = np.maximum(wx0, wx1) + self.bias      # weighted max

            # --- Select per-node op ---
            y = y_all[self.op, node_idx]                      # (N,)

            # --- Apply soft sigmoid gate (synaptic strength) ---
            y_gated = gate_val * y

            # --- Scatter-add to registers (synaptic summation) ---
            np.add.at(R, self.dst, y_gated)

            # --- Clamp & sanitise ---
            np.clip(R, -10.0, 10.0, out=R)
            np.nan_to_num(R, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)

        return R[self.action_base : self.action_base + self.n_actions].copy()

    # ------------------------------------------------------------------
    # Action helper
    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, explore: bool = False) -> int:
        logits = self(obs)
        return int(np.argmax(logits))