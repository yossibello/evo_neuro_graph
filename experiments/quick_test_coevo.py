"""Quick smoke test: graph policy + env co-evolution, tiny settings."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eng.evolve import GAConfig, run_ga
from tasks.tinygrid import TinyGrid

if __name__ == "__main__":
    cfg = GAConfig(
        policy="graph",
        pop_size=20,
        elites=4,
        episodes=8,
        max_steps=80,
        generations=8,
        graph_nodes=32,
        graph_ticks=3,
        graph_registers=128,
        graph_memory=8,
        processes=2,
        novelty_enabled=False,
        cambrian_period=0,
        # env co-evolution ON
        env_coevolve=True,
        env_target_success=0.20,
        env_coevolve_lookback=3,
        env_walls_min=0,
        env_walls_max=10,
        env_size_min=7,
        env_size_max=11,
    )

    # Use env_factory (lambda) to run single-process — avoids multiprocessing
    # spawn issues when running as a quick local test.
    winner, history = run_ga(
        env_factory=lambda: TinyGrid(max_steps=80, size=7, num_walls=3),
        cfg=cfg,
    )

    print("\n=== Summary ===")
    for h in history:
        env_info = ""
        if "env_size" in h:
            env_info = (
                f" | env size={h['env_size']} walls={h['env_walls']}"
                f" sr_smooth={h.get('env_sr_smooth', 0):.2f}"
            )
        print(
            f"Gen {h['gen']:03d} best_f={h['best_f']:+.3f}"
            f" best_s={h['best_s']:.2f}{env_info}"
        )
    print("PASS — co-evolution test complete.")
