# eng/policies_graph.py
import numpy as np


class GraphPolicy:
    """
    Tiny code-graph policy.

    - num_registers: total scalar registers R[0..num_registers-1]
    - node_params: shape (N, 8):
        [src0, src1, dst, op_id, w0, w1, bias, gate]
    - last 5 registers are used as action logits: up, right, down, left, toggle
    """

    def _sync_params_list(self):
        """
        Keep a generic `.params` list in sync with the main node parameter
        matrix `self.node_params`.

        The GA code in evolve.py expects policy.params to be a list of arrays,
        so for GraphPolicy we just expose [node_params].
        """
        self._params = [self.node_params]

    def __init__(
        self,
        num_registers: int = 72,
        num_nodes: int = 32,
        node_params: np.ndarray | None = None,
    ):
        self.num_registers = int(num_registers)

        if node_params is None:
            # Random initial graph
            N = int(num_nodes)
            # src0, src1, dst, op_id, w0, w1, bias, gate
            node_params = np.zeros((N, 8), dtype=np.float32)

            # Inputs can be from any register (we'll mostly use obs in R[0:obs_dim])
            node_params[:, 0] = np.random.randint(-4, self.num_registers, size=N)   # src0
            node_params[:, 1] = np.random.randint(-4, self.num_registers, size=N)   # src1
            node_params[:, 2] = np.random.randint(-4, self.num_registers, size=N)   # dst
            node_params[:, 3] = np.random.randint(0, 6, size=N)                     # op_id
            node_params[:, 4] = np.random.randn(N).astype(np.float32)               # w0
            node_params[:, 5] = np.random.randn(N).astype(np.float32)               # w1
            node_params[:, 6] = np.random.randn(N).astype(np.float32)               # bias
            node_params[:, 7] = np.random.randn(N).astype(np.float32)               # gate
        else:
            node_params = np.asarray(node_params, dtype=np.float32)

        self.node_params = node_params
        self.num_nodes = self.node_params.shape[0]

        # registers (reused each call)
        self.registers = np.zeros(self.num_registers, dtype=np.float32)

        # last 5 registers = action logits
        self.n_actions = 5
        self.action_base = self.num_registers - self.n_actions

        self._rebuild_cache()
        self._sync_params_list()

    # -----------------------
    # Parameters API for GA
    # -----------------------
    @property
    def params(self):
        # GA expects a list of arrays
        return [self.node_params]

    @params.setter
    def params(self, new_params):
        # new_params is a list; first entry is node_params
        self.node_params = np.asarray(new_params[0], dtype=np.float32)
        self.num_nodes = self.node_params.shape[0]
        self._rebuild_cache()
        self._sync_params_list()

    def clone(self):
        return GraphPolicy(
            num_registers=self.num_registers,
            node_params=self.node_params.copy(),
        )

    def as_dict(self):
        # for save_policy_npz
        return {
            "kind": "graph",
            "graph_params": self.node_params,
            "num_registers": self.num_registers,
        }

    # -----------------------
    # Internal: unpack node params
    # -----------------------
    def _rebuild_cache(self):
        p = self.node_params
        self.src0 = p[:, 0].astype(np.int32)
        self.src1 = p[:, 1].astype(np.int32)
        self.dst  = p[:, 2].astype(np.int32)
        self.op   = p[:, 3].astype(np.int32)
        self.w0   = p[:, 4]
        self.w1   = p[:, 5]
        self.bias = p[:, 6]
        self.gate = p[:, 7]

    def _resolve_idx(self, idx: int) -> int:
        """
        Allow negative indices to refer to registers from the end.
        E.g., -1 = last register, -2 = second last, etc.
        """
        if idx >= 0:
            return idx
        return self.num_registers + idx  # Python negative indexing style

    def _apply_node(self, i: int, R: np.ndarray):
        src0, src1, dst, op_id, w0, w1, bias, gate = self.node_params[i]

        # --- Discretize & clamp indices ---
        s0 = int(round(src0))
        s1 = int(round(src1))
        d  = int(round(dst))

        # Allow negative indices as Python negatives (scratch / tail regs),
        # but keep them within [-num_registers, num_registers-1].
        max_idx = self.num_registers - 1
        min_idx = -self.num_registers

        if s0 > max_idx: s0 = max_idx
        if s1 > max_idx: s1 = max_idx
        if s0 < min_idx: s0 = min_idx
        if s1 < min_idx: s1 = min_idx

        # Very simple gate: if gate <= 0, skip this node
        if gate <= 0.0:
            return

        # Safe reads from registers
        x0 = R[s0]
        x1 = R[s1]

        # --- Primitive operations ---
        if   op_id == 0:   # ADD
            y = x0 + x1
        elif op_id == 1:   # MUL
            y = x0 * x1
        elif op_id == 2:   # MIN
            y = min(x0, x1)
        elif op_id == 3:   # MAX
            y = max(x0, x1)
        elif op_id == 4:   # OP-1
            y = np.tanh(w0 * x0 + w1 * x1 + bias)
        elif op_id == 5:   # OP-2
            y = np.tanh(w0 * x0 - w1 * x1 + bias)
        else:
            # fallback
            y = x0

        # --- Write-back: only if d is a valid non-negative register index ---
        if 0 <= d < self.num_registers:
            R[d] = y

    # -----------------------
    # Forward / callable
    # -----------------------
    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        Run the graph policy: copy input obs into registers,
        apply all graph nodes, return logits for 5 actions.
        """
        obs = np.asarray(obs, dtype=np.float32)

        # clear working registers
        self.registers.fill(0.0)

        obs_dim = obs.shape[0]
        max_in = self.num_registers - self.n_actions

        # Copy observation features into registers (truncate if ever needed)
        n_in = min(obs_dim, max_in)
        self.registers[:n_in] = obs[:n_in]

        # Run all graph nodes
        for i in range(self.num_nodes):
            self._apply_node(i, self.registers)

        # Output logits (last 5 registers)
        logits = self.registers[self.action_base : self.action_base + self.n_actions]
        return logits

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Make policy instances directly callable: policy(obs) -> logits.
        """
        return self.forward(obs)

    # -----------------------
    # Action API (used by GA / eval)
    # -----------------------
    def act(self, obs: np.ndarray, explore: bool = False) -> int:
        logits = self.forward(obs)
        # pure greedy (no exploration) for now
        return int(np.argmax(logits))