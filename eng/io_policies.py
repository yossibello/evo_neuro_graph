import numpy as np
from eng.policies import LinearPolicy
from eng.policies_mlp import MLPPolicy


def load_policy_npz(path: str):
    """
    Load either linear or mlp policy from a .npz created by save_policy_npz()
    or older formats with W1,b1,W2,b2,... but no W0.
    """
    data = np.load(path, allow_pickle=True)
    files = set(data.files)
    kind = None

    if "kind" in files:
        k = data["kind"]
        kind = k.item() if hasattr(k, "item") else str(k)

    # ---- Linear: W, b ----
    if kind == "linear" or {"W", "b"}.issubset(files):
        W, b = data["W"], data["b"]
        return LinearPolicy(W, b)

    # ---- MLP: W0/b0, W1/b1, ... or W1/b1,... only ----
    if kind == "mlp" or any(name.startswith("W") for name in files):
        # collect all W* indices that are of the form "W<number>"
        idxs = []
        for name in files:
            if name.startswith("W"):
                suffix = name[1:]
                if suffix.isdigit():
                    idxs.append(int(suffix))

        if not idxs:
            raise ValueError(f"No MLP layers found in {path} (no W<i> keys).")

        idxs = sorted(idxs)  # e.g. [1,2,3] or [0,1,2]

        layers = []
        for i in idxs:
            Wi = data[f"W{i}"]
            bi_name = f"b{i}"
            if bi_name in data.files:
                bi = data[bi_name]
            else:
                # fallback if bias missing
                bi = np.zeros((Wi.shape[0],), dtype=Wi.dtype)
            layers.append((Wi, bi))

        return MLPPolicy(layers=layers)

    # ---- Heuristic fallback: raw arrays with shapes ----
    if {"W", "b"}.intersection(files):
        W = data.get("W", None)
        b = data.get("b", None)
        if W is not None and b is not None and W.ndim == 2 and b.ndim == 1:
            return LinearPolicy(W, b)

    raise ValueError(f"Unrecognized policy format in {path}; keys={sorted(files)}")