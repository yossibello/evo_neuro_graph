# eng/policies_graph.py
import numpy as np

class GraphPolicy:
    def __init__(self,
                 num_registers: int = 72,
                 num_nodes: int = 32,
                 node_params: np.ndarray | None = None):
        self.num_registers = int(num_registers)

        if node_params is None:
            N = int(num_nodes)
            node_params = np.zeros((N, 8), dtype=np.float32)
            node_params[:, 0] = np.random.randint(-4, self.num_registers, size=N)  # src0
            node_params[:, 1] = np.random.randint(-4, self.num_registers, size=N)  # src1
            node_params[:, 2] = np.random.randint(-4, self.num_registers, size=N)  # dst
            node_params[:, 3] = np.random.randint(0, 6, size=N)                    # op_id
            node_params[:, 4] = np.random.randn(N).astype(np.float32)              # w0
            node_params[:, 5] = np.random.randn(N).astype(np.float32)              # w1
            node_params[:, 6] = np.random.randn(N).astype(np.float32)              # bias
            node_params[:, 7] = np.random.randn(N).astype(np.float32)              # gate
        else:
            node_params = np.asarray(node_params, dtype=np.float32)

        self.node_params = node_params
        self.num_nodes = self.node_params.shape[0]

        # registers
        self.registers = np.zeros(self.num_registers, dtype=np.float32)

        # last 5 registers = action logits
        self.n_actions = 5
        self.action_base = self.num_registers - self.n_actions

        self._rebuild_cache()
        self._sync_params_list()

    # -----------------------
    # GA params API
    # -----------------------
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
            node_params=self.node_params.copy(),
        )

    def as_dict(self):
        return {
            "kind": "graph",
            "graph_params": self.node_params,
            "num_registers": self.num_registers,
        }

    # -----------------------
    # Internal
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

    def _apply_node(self, i: int, R: np.ndarray):
        src0, src1, dst, op_id, w0, w1, bias, gate = self.node_params[i]

        # discretize / clamp indices
        s0 = int(round(src0))
        s1 = int(round(src1))
        d  = int(round(dst))

        max_idx = self.num_registers - 1
        min_idx = -self.num_registers

        s0 = max(min(s0, max_idx), min_idx)
        s1 = max(min(s1, max_idx), min_idx)

        if gate <= 0.0:
            return

        x0 = R[s0]
        x1 = R[s1]

        if   op_id == 0:   # ADD
            y = x0 + x1
        elif op_id == 1:   # MUL
            y = x0 * x1
        elif op_id == 2:   # MIN
            y = min(x0, x1)
        elif op_id == 3:   # MAX
            y = max(x0, x1)
        elif op_id == 4:
            y = np.tanh(w0 * x0 + w1 * x1 + bias)
        elif op_id == 5:
            y = np.tanh(w0 * x0 - w1 * x1 + bias)
        else:
            y = x0

        if 0 <= d < self.num_registers:
            R[d] = y

    # -----------------------
    # Forward: returns *logits*
    # -----------------------
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        self.registers.fill(0.0)

        obs_dim = obs.shape[0]
        max_in = self.num_registers - self.n_actions

        if obs_dim > max_in:
            raise ValueError(
                f"Observation dim {obs_dim} exceeds available registers {max_in}. "
                f"Increase num_registers in GraphPolicy."
            )

        self.registers[:obs_dim] = obs

        for i in range(self.num_nodes):
            self._apply_node(i, self.registers)

        logits = self.registers[self.action_base : self.action_base + self.n_actions]
        return logits

    # -----------------------
    # Action helper
    # -----------------------
    def act(self, obs: np.ndarray, explore: bool = False) -> int:
        logits = self(obs)
        return int(np.argmax(logits))