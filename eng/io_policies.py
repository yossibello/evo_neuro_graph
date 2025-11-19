# eng/io_policies.py
"""
Helpers to load policies (linear / mlp / graph) from .npz files.
Used by visualize_policy.py, eval_policy.py, etc.
"""

from __future__ import annotations
import numpy as np

from eng.policies import LinearPolicy
try:
    from eng.policies_mlp import MLPPolicy
    HAS_MLP = True
except Exception:
    HAS_MLP = False

try:
    from eng.policies_graph import GraphPolicy
    HAS_GRAPH = True
except Exception:
    HAS_GRAPH = False


def load_policy_npz(path: str):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)

    kind = None
    if "kind" in keys:
        k = d["kind"]
        kind = k.item() if hasattr(k, "item") else str(k)

    # -------- Linear ----------
    if kind == "linear" or ({"W", "b"} <= keys):
        W = d["W"]
        b = d["b"]
        return LinearPolicy(W=W, b=b)

    # -------- MLP ----------
    if kind == "mlp" or any(k.startswith("W0") for k in keys):
        if not HAS_MLP:
            raise ValueError("MLPPolicy not available but MLP npz was loaded.")
        # Assume W0,b0,W1,b1,... in order
        Ws = []
        bs = []
        i = 0
        while f"W{i}" in keys:
            Ws.append(d[f"W{i}"])
            bi = d.get(f"b{i}", None)
            if bi is None:
                raise ValueError(f"Missing b{i} for MLP in {path}")
            bs.append(bi)
            i += 1
        params = []
        for W, b in zip(Ws, bs):
            params.append(W)
            params.append(b)
        return MLPPolicy(params=params)

    # -------- GraphPolicy ----------
    if (kind == "graph") or ("graph_params" in keys):
        if not HAS_GRAPH:
            raise ValueError("GraphPolicy not available but graph npz was loaded.")
        gp = d["graph_params"]
        nr = int(d.get("num_registers", 48))
        return GraphPolicy(num_registers=nr, node_params=gp)

    raise ValueError(f"Unrecognized policy format in {path}: keys={keys}, kind={kind}")