# eng/policies_graph.py

from __future__ import annotations
import numpy as np
from typing import List


class GraphPolicy:
    """
    Tiny 'neuron graph' / programmatic policy.

    - num_registers: how many registers (memory cells) we have.
    - node_params: (num_nodes, 8) array, where each row is:

        [ src0_idx_raw, src1_idx_raw, gate_idx_raw, out_idx_raw,
          w0, w1, bias, gate_th ]

      At runtime we interpret:

        src0 = int(abs(src0_idx_raw)) % num_registers
        src1 = int(abs(src1_idx_raw)) % num_registers
        gate = int(abs(gate_idx_raw)) % num_registers
        out  = int(abs(out_idx_raw))  % num_registers

        s = w0 * R[src0] + w1 * R[src1] + bias
        if R[gate] > gate_th:
            R[out] = tanh(s)

    - After running all nodes, the last 5 registers are treated as
      logits for the 5 TinyGrid actions.
    """

    def __init__(
        self,
        num_registers: int = 48,
        num_nodes: int = 32,
        node_params: np.ndarray | None = None,
    ):
        self.num_registers = int(num_registers)
        self.params_per_node = 8

        # bounds for structural evolution
        self.min_nodes = 8
        self.max_nodes = 64

        if node_params is None:
            num_nodes = int(num_nodes)
            # small random init
            self.node_params = (
                np.random.randn(num_nodes, self.params_per_node).astype(np.float32) * 0.2
            )
        else:
            node_params = np.asarray(node_params, dtype=np.float32)
            assert node_params.ndim == 2
            assert node_params.shape[1] == self.params_per_node
            self.node_params = node_params

        self._sync_params_list()

    # ---------- basic properties ----------

    @property
    def num_nodes(self) -> int:
        return self.node_params.shape[0]

    def _sync_params_list(self):
        """
        GA mutate/crossover in eng/evolve.py expects .params to be a list of arrays.
        For GraphPolicy, we just expose [node_params].
        """
        self.params: List[np.ndarray] = [self.node_params]

    def clone(self) -> "GraphPolicy":
        return GraphPolicy(
            num_registers=self.num_registers,
            node_params=self.node_params.copy(),
        )

    # ---------- forward / act ----------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Run the tiny program once.

        obs: (61,) observation
        returns: (5,) logits for TinyGrid actions
        """
        obs = np.asarray(obs, dtype=np.float32)
        R = np.zeros((self.num_registers,), dtype=np.float32)

        # Load observation into the first registers
        n = min(self.num_registers, obs.shape[0])
        R[:n] = obs[:n]

        nr = self.num_registers

        for k in range(self.num_nodes):
            p = self.node_params[k]  # shape (8,)
            (
                src0_raw,
                src1_raw,
                gate_raw,
                out_raw,
                w0,
                w1,
                bias,
                gate_th,
            ) = p

            src0 = int(abs(src0_raw)) % nr
            src1 = int(abs(src1_raw)) % nr
            gate = int(abs(gate_raw)) % nr
            out  = int(abs(out_raw))  % nr

            s = w0 * R[src0] + w1 * R[src1] + bias
            if R[gate] > gate_th:
                R[out] = np.tanh(s)

        if self.num_registers < 5:
            raise ValueError("num_registers must be >= 5 for 5 actions.")
        logits = R[-5:]  # last 5 registers
        return logits

    def act(self, obs: np.ndarray, explore: bool = False) -> int:
        """
        Greedy action (no exploration; GA creates exploration via mutation).
        """
        logits = self(obs)
        return int(np.argmax(logits))

    # ---------- structural mutation ----------

    def structural_mutate(
        self,
        rng: np.random.RandomState,
        p_rewire: float = 0.3,
        p_add: float = 0.1,
        p_del: float = 0.1,
        p_dup: float = 0.1,
    ):
        """
        Structural mutations in addition to numeric ones:

        - rewire: random change of one of the index fields
        - add:    add a new random node
        - del:    delete an existing node
        - dup:    duplicate and slightly perturb a node
        """
        # Rewire
        if rng.rand() < p_rewire and self.num_nodes > 0:
            k = rng.randint(0, self.num_nodes)
            node = self.node_params[k]
            idx_field = rng.randint(0, 4)  # 0=src0,1=src1,2=gate,3=out
            node[idx_field] = rng.randn() * 5.0  # big jump in index-space

        # Add new node
        if rng.rand() < p_add and self.num_nodes < self.max_nodes:
            new_node = rng.randn(self.params_per_node).astype(np.float32) * 0.5
            self.node_params = np.vstack([self.node_params, new_node[None, :]])

        # Delete a node
        if rng.rand() < p_del and self.num_nodes > self.min_nodes:
            k = rng.randint(0, self.num_nodes)
            self.node_params = np.delete(self.node_params, k, axis=0)

        # Duplicate
        if rng.rand() < p_dup and self.num_nodes < self.max_nodes and self.num_nodes > 0:
            k = rng.randint(0, self.num_nodes)
            clone = self.node_params[k].copy()
            clone += rng.randn(self.params_per_node).astype(np.float32) * 0.1
            self.node_params = np.vstack([self.node_params, clone[None, :]])

        self._sync_params_list()

    # ---------- saving ----------

    def as_dict(self) -> dict:
        """
        For np.savez_compressed via eng.evolve.save_policy_npz.
        """
        return {
            "graph_params": self.node_params,
            "num_registers": np.array(self.num_registers, dtype=np.int32),
            "kind": "graph",
        }