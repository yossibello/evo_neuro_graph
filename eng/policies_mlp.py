# eng/policies_mlp.py
import numpy as np
import copy

class MLPPolicy:
    """
    Simple feed-forward MLP policy for TinyGrid:
    61 -> hidden1 -> hidden2 -> 5 (action logits)
    """

    def __init__(self, layers=None, hidden_sizes=(64, 32), input_size=61, output_size=5):
        """
        If `layers` is given (list of (W,b)), we reuse it.
        Otherwise, initialize random weights.
        """
        self.layers = []
        if layers is not None:
            # assume list of (W,b)
            self.layers = [(np.asarray(W, dtype=np.float32), np.asarray(b, dtype=np.float32))
                           for (W, b) in layers]
        else:
            sizes = [input_size, *hidden_sizes, output_size]
            rng = np.random.default_rng()
            for i in range(len(sizes) - 1):
                W = rng.normal(scale=0.1, size=(sizes[i + 1], sizes[i])).astype(np.float32)
                b = np.zeros((sizes[i + 1],), dtype=np.float32)
                self.layers.append((W, b))

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Compute forward pass: returns logits of shape [5]."""
        x = np.asarray(obs, dtype=np.float32)
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                # hidden layer: ReLU
                x = np.maximum(0, x)
        return x  # logits

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.forward(obs)

    def clone(self):
        """Deep copy."""
        new_layers = [(W.copy(), b.copy()) for (W, b) in self.layers]
        return MLPPolicy(layers=new_layers)

    def mutate(self, sigma: float, rng: np.random.Generator):
        """In-place Gaussian mutation."""
        for (W, b) in self.layers:
            W += rng.normal(scale=sigma, size=W.shape)
            b += rng.normal(scale=sigma, size=b.shape)

    def as_dict(self) -> dict:
        """Return dict suitable for np.savez_compressed(..., **dict)."""
        d = {}
        for i, (W, b) in enumerate(self.layers):
            d[f"W{i}"] = W
            d[f"b{i}"] = b
        d["kind"] = "mlp"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MLPPolicy":
        """Create MLPPolicy from npz dict."""
        layers = []
        i = 0
        while f"W{i}" in d and f"b{i}" in d:
            W, b = d[f"W{i}"], d[f"b{i}"]
            layers.append((W, b))
            i += 1
        return cls(layers=layers)