import numpy as np

class MLPPolicy:
    """
    Small nonlinear policy: 61 -> 64 -> 32 -> 5 with ReLU.
    Greedy inference (no exploration).
    """    
    def __init__(self, params=None):
        self.in_dim = 61
        self.h1 = 64
        self.h2 = 32
        self.out_dim = 5
        if params is None:
            self.params = self._init_params()
        else:
            # expect [W1,b1,W2,b2,W3,b3]
            self.params = [p.astype(np.float32) for p in params]

    def _init_params(self):
        def xavier(fan_in, fan_out):
            limit = np.sqrt(6.0/(fan_in+fan_out))
            W = np.random.uniform(-limit, limit, size=(fan_out, fan_in)).astype(np.float32)
            b = np.zeros((fan_out,), dtype=np.float32)
            return W, b
        W1,b1 = xavier(self.in_dim, self.h1)
        W2,b2 = xavier(self.h1, self.h2)
        W3,b3 = xavier(self.h2, self.out_dim)
        return [W1,b1,W2,b2,W3,b3]

    def logits(self, obs: np.ndarray) -> np.ndarray:
        W1,b1,W2,b2,W3,b3 = self.params
        x = obs.astype(np.float32)
        h1 = np.maximum(0, W1 @ x + b1)
        h2 = np.maximum(0, W2 @ h1 + b2)
        z  = (W3 @ h2 + b3)
        return z

    def act(self, obs: np.ndarray, explore: bool=False, temperature: float=1.0) -> int:
        # Greedy by default (no exploration)
        return int(np.argmax(self.logits(obs)))

    def clone(self):
        return MLPPolicy([p.copy() for p in self.params])
