# eng/policies_graph.py
"""
Evolved computational graph — a register machine whose topology, weights,
and **learning rules** are optimised by a genetic algorithm.

Design principles
-----------------
1. **Multiple execution ticks** — information propagates across ticks (like
   recurrent "brain cycles").  Observation is re-injected each tick.
2. **Sigmoid gates** — every node is alive (soft gate in (0, 1)).
3. **Learnable weights in every op** — all 8 ops use an affine transform.
4. **Persistent memory** — a subset of registers survive between time steps.
5. **Hebbian plasticity** — evolved per-node learning rates allow the circuit
   to learn DURING its lifetime via reward-modulated Hebbian updates:
       Δw = η × pre × post × reward_signal
   Evolution doesn't just evolve a solution — it evolves the ability to learn.
6. **Vectorised forward pass** — all nodes fire simultaneously each tick.
7. **Register clamping** — values clipped to [-10, 10] each tick.
"""

import numpy as np

NUM_OPS = 8          # number of distinct node operations
NUM_NODE_COLS = 11   # src0, src1, dst, op, w0, w1, bias, gate, eta_w0, eta_w1, eta_bias


class GraphPolicy:
    """
    A policy that evolves a small "brain circuit" of N computational nodes
    operating on a shared register file **with persistent memory**.

    Register layout
    ---------------
    [0 .. obs_dim)             — observation input (re-injected each tick)
    [obs_dim .. mem_base)      — scratch / intermediate computation (reset each call)
    [mem_base .. action_base)  — MEMORY (persists between time steps, decays slowly)
    [action_base .. num_reg)   — action logits (5 outputs, reset each call)

    Node parameter row (11 columns)
    -------------------------------
    0: src0      — index of first source register
    1: src1      — index of second source register
    2: dst       — index of destination register
    3: op_id     — which of the 8 operations to apply
    4: w0        — weight on first source  (base / genome)
    5: w1        — weight on second source (base / genome)
    6: bias      — additive bias           (base / genome)
    7: gate      — sigmoid gate input (controls contribution strength)
    8: eta_w0    — Hebbian learning rate for w0 (evolved)
    9: eta_w1    — Hebbian learning rate for w1 (evolved)
   10: eta_bias  — Hebbian learning rate for bias (evolved)
    """

    def __init__(
        self,
        num_registers: int = 128,
        num_nodes: int = 64,
        num_ticks: int = 3,
        num_memory: int = 16,
        memory_decay: float = 0.95,
        node_params: np.ndarray | None = None,
    ):
        self.num_registers = int(num_registers)
        self.num_ticks = int(num_ticks)
        self.num_memory = int(num_memory)
        self.memory_decay = float(memory_decay)
        self.n_actions = 5
        self.action_base = self.num_registers - self.n_actions
        # Memory sits right before actions
        self.mem_base = self.action_base - self.num_memory

        if node_params is None:
            node_params = self._init_random(int(num_nodes))
        else:
            node_params = np.asarray(node_params, dtype=np.float32)
            # Backward compat: pad old 8-col checkpoints → 11 cols
            if node_params.shape[1] < NUM_NODE_COLS:
                pad = np.zeros(
                    (node_params.shape[0], NUM_NODE_COLS - node_params.shape[1]),
                    dtype=np.float32,
                )
                node_params = np.hstack([node_params, pad])

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
          - ~8 memory nodes:        obs -> memory  (seed memory usage)
          - ~8 memory-read nodes:   memory -> action (seed memory readout)
          - ~10 intermediate nodes: obs -> scratch  (enables multi-tick processing)
          - rest:                   random -> random (diversity for evolution)
        All gates start positive so every node is alive from the start.
        """
        nr = self.num_registers
        obs_dim = 61  # TinyGrid observation dimension
        ab = self.action_base
        mb = self.mem_base
        nm = self.num_memory
        na = self.n_actions

        p = np.zeros((N, NUM_NODE_COLS), dtype=np.float32)
        idx = 0

        # --- Backbone: input -> action ---
        n_bb = min(10, N - idx)
        for i in range(n_bb):
            p[idx, 0] = np.random.randint(0, obs_dim)       # src0 from obs
            p[idx, 1] = np.random.randint(0, obs_dim)       # src1 from obs
            p[idx, 2] = ab + (i % na)                       # dst to action
            p[idx, 3] = 1                                    # tanh op
            p[idx, 4] = np.random.randn() * 0.5             # w0
            p[idx, 5] = np.random.randn() * 0.5             # w1
            p[idx, 6] = np.random.randn() * 0.1             # bias
            p[idx, 7] = np.random.uniform(1.5, 3.0)         # gate (strongly on)
            idx += 1

        # --- Memory writers: obs -> memory ---
        n_mw = min(8, N - idx)
        for i in range(n_mw):
            p[idx, 0] = np.random.randint(0, obs_dim)
            p[idx, 1] = np.random.randint(0, obs_dim)
            p[idx, 2] = mb + (i % nm)                       # dst to memory
            p[idx, 3] = np.random.choice([0, 1, 3])         # linear/tanh/softsign
            p[idx, 4] = np.random.randn() * 0.3
            p[idx, 5] = np.random.randn() * 0.3
            p[idx, 6] = np.random.randn() * 0.1
            p[idx, 7] = np.random.uniform(1.0, 2.5)
            idx += 1

        # --- Memory readers: memory -> action/scratch ---
        n_mr = min(8, N - idx)
        for i in range(n_mr):
            p[idx, 0] = mb + np.random.randint(0, nm)        # src0 from memory
            p[idx, 1] = np.random.randint(0, obs_dim)        # src1 from obs (context)
            p[idx, 2] = ab + (i % na) if i < na else np.random.randint(obs_dim, mb)
            p[idx, 3] = np.random.choice([0, 1, 3])
            p[idx, 4] = np.random.randn() * 0.3
            p[idx, 5] = np.random.randn() * 0.3
            p[idx, 6] = np.random.randn() * 0.1
            p[idx, 7] = np.random.uniform(1.0, 2.5)
            idx += 1

        # --- Intermediate: input -> scratch ---
        n_mid = min(10, N - idx)
        for ii in range(n_mid):
            p[idx, 0] = np.random.randint(0, obs_dim)
            p[idx, 1] = np.random.randint(0, obs_dim)
            scratch_reg = np.random.randint(obs_dim, mb) if obs_dim < mb else 0
            p[idx, 2] = scratch_reg
            p[idx, 3] = np.random.randint(0, NUM_OPS)
            p[idx, 4] = np.random.randn() * 0.3
            p[idx, 5] = np.random.randn() * 0.3
            p[idx, 6] = np.random.randn() * 0.1
            p[idx, 7] = np.random.uniform(1.0, 2.5)
            idx += 1

        # --- Random: any -> any ---
        for i in range(idx, N):
            p[i, 0] = np.random.randint(0, nr)
            p[i, 1] = np.random.randint(0, nr)
            p[i, 2] = np.random.randint(0, nr)
            p[i, 3] = np.random.randint(0, NUM_OPS)
            p[i, 4] = np.random.randn() * 0.3
            p[i, 5] = np.random.randn() * 0.3
            p[i, 6] = np.random.randn() * 0.1
            p[i, 7] = np.random.uniform(0.5, 2.5)

        # --- Evolved learning rates for ALL nodes ---
        p[:, 8]  = np.random.randn(N) * 0.01   # eta_w0
        p[:, 9]  = np.random.randn(N) * 0.01   # eta_w1
        p[:, 10] = np.random.randn(N) * 0.005  # eta_bias

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
        c = GraphPolicy(
            num_registers=self.num_registers,
            num_nodes=self.num_nodes,
            num_ticks=self.num_ticks,
            num_memory=self.num_memory,
            memory_decay=self.memory_decay,
            node_params=self.node_params.copy(),
        )
        return c

    def as_dict(self):
        return {
            "kind": "graph",
            "graph_params": self.node_params,
            "num_registers": np.array(self.num_registers, dtype=np.int32),
            "num_ticks": np.array(self.num_ticks, dtype=np.int32),
            "num_memory": np.array(self.num_memory, dtype=np.int32),
            "memory_decay": np.array(self.memory_decay, dtype=np.float32),
        }

    def reset_memory(self):
        """Reset all ephemeral state between episodes."""
        # Clear memory registers
        self.registers[self.mem_base : self.action_base] = 0.0
        # Reset live weights back to evolved base weights
        self.live_w0[:] = self.w0
        self.live_w1[:] = self.w1
        self.live_bias[:] = self.bias
        # Clear activity traces
        N = self.num_nodes
        self._trace_pre0 = np.zeros(N, dtype=np.float32)
        self._trace_pre1 = np.zeros(N, dtype=np.float32)
        self._trace_post = np.zeros(N, dtype=np.float32)
        # Clear eligibility traces
        self._elig_pre0 = np.zeros(N, dtype=np.float32)
        self._elig_pre1 = np.zeros(N, dtype=np.float32)
        self._elig_post = np.zeros(N, dtype=np.float32)

    # ------------------------------------------------------------------
    # Cache (integer indices + float weights extracted for fast forward)
    # ------------------------------------------------------------------
    def _rebuild_cache(self):
        p = self.node_params
        nr = self.num_registers
        N = p.shape[0]
        self.src0 = np.clip(np.round(p[:, 0]).astype(np.int32), 0, nr - 1)
        self.src1 = np.clip(np.round(p[:, 1]).astype(np.int32), 0, nr - 1)
        self.dst  = np.clip(np.round(p[:, 2]).astype(np.int32), 0, nr - 1)
        self.op   = np.clip(np.round(p[:, 3]).astype(np.int32), 0, NUM_OPS - 1)
        # Base weights (the "genome" — never change during an episode)
        self.w0   = p[:, 4].copy()
        self.w1   = p[:, 5].copy()
        self.bias = p[:, 6].copy()
        self.gate = p[:, 7].copy()
        # Evolved learning rates (Hebbian plasticity)
        self.eta_w0   = p[:, 8].copy()
        self.eta_w1   = p[:, 9].copy()
        self.eta_bias = p[:, 10].copy()
        # Live weights (modified by Hebbian learning during episode)
        self.live_w0   = self.w0.copy()
        self.live_w1   = self.w1.copy()
        self.live_bias = self.bias.copy()
        # Activity traces for Hebbian update
        self._trace_pre0 = np.zeros(N, dtype=np.float32)
        self._trace_pre1 = np.zeros(N, dtype=np.float32)
        self._trace_post = np.zeros(N, dtype=np.float32)
        # Eligibility traces (decaying accumulation across time steps)
        # These allow past activity to receive credit when reward arrives later.
        # Biologically: synaptic tags that persist and get reinforced by dopamine.
        self._elig_pre0 = np.zeros(N, dtype=np.float32)
        self._elig_pre1 = np.zeros(N, dtype=np.float32)
        self._elig_post = np.zeros(N, dtype=np.float32)
        self._elig_decay = 0.92  # how fast eligibility fades (≈12-step half-life)

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

        # Zero scratch + action registers, but KEEP memory registers!
        R[:obs_dim] = 0.0                           # obs (will be overwritten)
        R[obs_dim:self.mem_base] = 0.0              # scratch
        # Memory decays but persists:
        R[self.mem_base:self.action_base] *= self.memory_decay
        R[self.action_base:] = 0.0                  # action logits

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

            # --- Affine building blocks (using LIVE weights) ---
            wx0 = self.live_w0 * x0     # (N,)
            wx1 = self.live_w1 * x1     # (N,)
            z   = wx0 + wx1 + self.live_bias

            # --- Compute all 8 ops for all nodes (vectorised) ---
            y_all = np.empty((NUM_OPS, N), dtype=np.float32)
            y_all[0] = z                                      # linear
            y_all[1] = np.tanh(z)                             # tanh
            y_all[2] = np.maximum(0.0, z)                     # relu
            y_all[3] = z / (1.0 + np.abs(z))                 # softsign
            y_all[4] = wx0 * wx1 + self.live_bias            # product
            y_all[5] = np.abs(z)                              # abs
            y_all[6] = np.minimum(wx0, wx1) + self.live_bias  # weighted min
            y_all[7] = np.maximum(wx0, wx1) + self.live_bias  # weighted max

            # --- Select per-node op ---
            y = y_all[self.op, node_idx]                      # (N,)

            # --- Apply soft sigmoid gate (synaptic strength) ---
            y_gated = gate_val * y

            # --- Scatter-add to registers (synaptic summation) ---
            np.add.at(R, self.dst, y_gated)

            # --- Clamp & sanitise ---
            np.clip(R, -10.0, 10.0, out=R)
            np.nan_to_num(R, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)

        # Store traces from final tick (for Hebbian update)
        self._trace_pre0 = x0.copy()
        self._trace_pre1 = x1.copy()
        self._trace_post = y_gated.copy()

        # Update eligibility traces (decaying accumulation across time steps)
        # This is the key to temporal credit assignment:
        # past activity leaves a "synaptic tag" that decays slowly.
        # When reward arrives later, the tag lets credit flow back in time.
        self._elig_pre0 = self._elig_decay * self._elig_pre0 + self._trace_pre0
        self._elig_pre1 = self._elig_decay * self._elig_pre1 + self._trace_pre1
        self._elig_post = self._elig_decay * self._elig_post + self._trace_post

        return R[self.action_base : self.action_base + self.n_actions].copy()

    # ------------------------------------------------------------------
    # Hebbian plasticity — reward-modulated lifetime learning
    #
    # Three-factor rule:  Δw = η × pre × post × reward
    #   - pre:    source register value (what fired into this node)
    #   - post:   gated output of this node (what this node produced)
    #   - reward: environment reward signal (dopamine-like modulation)
    #   - η:      evolved per-node learning rate (genome decides HOW to learn)
    #
    # This means the GA doesn't evolve a solution — it evolves a LEARNER.
    # ------------------------------------------------------------------
    def hebbian_update(self, reward: float):
        """Reward-modulated Hebbian update with eligibility traces.

        Uses BOTH immediate traces and decaying eligibility traces.
        - Immediate traces: reinforce what just happened (fast learning)
        - Eligibility traces: reinforce what happened in the past (temporal credit)

        This is analogous to how dopamine in the brain reinforces not just
        the current synapse activity, but also recent past activity via
        synaptic eligibility tags.
        """
        mod = np.float32(np.clip(reward, -2.0, 2.0))  # modulatory signal

        # Blend immediate + eligibility traces (50/50)
        # Immediate: what just fired this step
        # Eligibility: accumulated decaying trace of past steps
        eff_pre0 = 0.5 * self._trace_pre0 + 0.5 * self._elig_pre0
        eff_pre1 = 0.5 * self._trace_pre1 + 0.5 * self._elig_pre1
        eff_post = 0.5 * self._trace_post + 0.5 * self._elig_post

        # Three-factor Hebbian rule with eligibility
        self.live_w0   += self.eta_w0   * eff_pre0 * eff_post * mod
        self.live_w1   += self.eta_w1   * eff_pre1 * eff_post * mod
        self.live_bias += self.eta_bias * eff_post * mod

        # Clamp live weights to prevent runaway
        np.clip(self.live_w0,   -5.0, 5.0, out=self.live_w0)
        np.clip(self.live_w1,   -5.0, 5.0, out=self.live_w1)
        np.clip(self.live_bias, -5.0, 5.0, out=self.live_bias)

    # ------------------------------------------------------------------
    # Action helper
    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, explore: bool = False) -> int:
        logits = self(obs)
        return int(np.argmax(logits))