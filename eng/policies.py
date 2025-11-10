
import numpy as np


class LinearPolicy:
    """
    Simple baseline policy: logits = W x + b
    x: 61-d observation
    actions: 5
    """
    def __init__(self, W=None, b=None):
        self.in_dim = 61
        self.out_dim = 5
        if W is None:
            limit = np.sqrt(6/(self.in_dim + self.out_dim))
            W = np.random.uniform(-limit, limit, size=(self.out_dim, self.in_dim)).astype(np.float32)
        if b is None:
            b = np.zeros(self.out_dim, dtype=np.float32)
        self.W = W.astype(np.float32)
        self.b = b.astype(np.float32)

    def logits(self, obs: np.ndarray) -> np.ndarray:
        return (self.W @ obs.astype(np.float32)) + self.b

    def act(self, obs: np.ndarray, explore: bool = True, temperature: float = 1.0) -> int:
        z = self.logits(obs)
        if explore and temperature > 0:
            z = z / max(1e-6, temperature)
            e = np.exp(z - np.max(z))
            p = e / np.clip(np.sum(e), 1e-6, 1e9)
            return int(np.random.choice(len(z), p=p))
        return int(np.argmax(z))

    def clone(self):
        return LinearPolicy(self.W.copy(), self.b.copy())
