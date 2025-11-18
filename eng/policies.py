# eng/policies.py
import numpy as np
import copy

class LinearPolicy:
    """
    Simple linear policy: logits = W @ obs + b
    Used for tiny experiments and baseline GA training.
    """

    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = np.asarray(W, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return logits for given observation (shape [61,])."""
        return self.W @ obs + self.b

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Alias for frameworks expecting .forward()."""
        return self.__call__(obs)

    def clone(self):
        """Deep copy of the policy."""
        return LinearPolicy(self.W.copy(), self.b.copy())

    def mutate(self, sigma: float, rng: np.random.Generator):
        """In-place Gaussian mutation."""
        self.W += rng.normal(scale=sigma, size=self.W.shape)
        self.b += rng.normal(scale=sigma, size=self.b.shape)

    def as_dict(self) -> dict:
        """For saving to .npz"""
        return {"W": self.W, "b": self.b, "kind": "linear"}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearPolicy":
        """For loading from .npz"""
        return cls(d["W"], d["b"])