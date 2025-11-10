
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import random

MAX_INS = 48  # instruction cap placeholder

class NeuronVM:
    """
    Minimal placeholder VM. Does nothing yet; returns no emissions.
    """
    def __init__(self, program=None, weights=None, mem_size=8, seed=0):
        self.program = program or []
        self.w = np.array(weights or [], dtype=np.float32)
        self.m = np.zeros(mem_size, dtype=np.float32)
        self.rng = random.Random(seed)

    def step(self, x: np.ndarray):
        emitted = []  # list of (port, payload_vector, tag)
        ins_exec = 0
        # TODO: implement bytecode execution
        return emitted, ins_exec


@dataclass
class Edge:
    src: int; src_port: int; dst: int; dst_port: int
    weight: float = 1.0
    gate: float = 0.0
    delay: int = 1


@dataclass
class Genome:
    neurons: List[dict] = field(default_factory=list)
    edges: List['Edge'] = field(default_factory=list)
    io_map: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


class Scheduler:
    """
    Event-driven placeholder: forwards inputs to action heads.
    """
    def __init__(self, genome: Genome, seeds: Dict[int,int]):
        self.genome = genome
        self.seeds = seeds

    def run_tick(self, input_packets: List[Tuple[np.ndarray, int]]):
        """
        input_packets: list of (payload vector, tag)
        Returns a dict of action scores for 5 actions.
        For now, just a random policy to allow end-to-end testing.
        """
        logits = np.array([random.uniform(-1,1) for _ in range(5)], dtype=np.float32)
        return logits


class GA:
    """
    Super-minimal evolutionary loop scaffold that evaluates random policies.
    Replace with real genome mutation/crossover later.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def evaluate_random(self, env_factory, episodes: int = 5, max_steps: int = 128, seed: int = 0):
        rng = random.Random(seed)
        total = 0.0
        for ep in range(episodes):
            env = env_factory()
            obs = env.reset(seed=rng.randrange(10**9))
            done = False
            steps = 0
            while not done and steps < max_steps:
                action = rng.randrange(5)
                obs, reward, done, info = env.step(action)
                total += reward
                steps += 1
        return total / max(1, episodes)
