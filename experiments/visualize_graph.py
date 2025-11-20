# experiments/visualize_graph.py
"""
Visualize a GraphPolicy as a Graphviz DOT graph.

Usage:
    python -m experiments.visualize_graph \
      --policy artifacts/best_graph_policy.npz \
      --out graph.dot \
      --max_nodes 64
"""

import argparse
import os
import sys
import numpy as np

# Make sure repo root is on path when run as module
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from eng.io_policies import load_policy_npz
try:
    from eng.policies_graph import GraphPolicy
    HAS_GRAPH = True
except Exception:
    HAS_GRAPH = False


# ---- Node layout assumptions ----
# Adjust these indices if your GraphPolicy uses a different layout.
IDX_SRC0 = 0
IDX_SRC1 = 1
IDX_DST  = 2
IDX_OP   = 3
IDX_W0   = 4
IDX_W1   = 5
IDX_BIAS = 6
IDX_GATE = 7


def op_to_name(op_code: int) -> str:
    """
    Human-readable name for op codes.
    Adjust to match your actual GraphPolicy implementation.
    """
    mapping = {
        0: "NOP",
        1: "ADD",
        2: "SUB",
        3: "MUL",
        4: "MAX",
        5: "MIN",
        6: "SIGMOID",
        7: "TANH",
        8: "RELU",
    }
    return mapping.get(int(op_code), f"OP{int(op_code)}")


def build_dot(policy: "GraphPolicy", max_nodes: int | None = None) -> str:
    """
    Build a Graphviz DOT string from a GraphPolicy.
    We treat:
      - registers as boxes: r0, r1, ...
      - nodes as circles:   n0, n1, ...
    Edges:
      r[src0] -(w0)-> nX, r[src1] -(w1)-> nX, nX -> r[dst]
    """
    node_params = np.array(policy.node_params)
    num_nodes = node_params.shape[0]
    num_registers = getattr(policy, "num_registers", 32)

    if max_nodes is not None:
        num_nodes = min(num_nodes, max_nodes)
        node_params = node_params[:num_nodes]

    lines: list[str] = []
    lines.append('digraph PolicyGraph {')
    lines.append('  rankdir=LR;')
    lines.append('  node [fontsize=10];')

    # Registers cluster
    lines.append('  subgraph cluster_regs {')
    lines.append('    label="Registers";')
    lines.append('    style=dashed;')
    for r in range(num_registers):
        lines.append(f'    r{r} [shape=box,label="R{r}"];')
    lines.append('  }')

    # Nodes cluster
    lines.append('  subgraph cluster_nodes {')
    lines.append('    label="Nodes";')
    lines.append('    style=dotted;')

    for i in range(num_nodes):
        row = node_params[i]
        src0 = int(row[IDX_SRC0])
        src1 = int(row[IDX_SRC1])
        dst  = int(row[IDX_DST])
        op   = int(row[IDX_OP])
        w0   = float(row[IDX_W0])
        w1   = float(row[IDX_W1])
        bias = float(row[IDX_BIAS])
        gate = float(row[IDX_GATE])

        op_name = op_to_name(op)
        gstr = f"{gate:.2f}"
        bstr = f"{bias:.2f}"

        # Node label: op + small params
        label = f"n{i}\\n{op_name}\\n gate={gstr}\\n bias={bstr}"
        lines.append(f'    n{i} [shape=circle,label="{label}"];')

        # Only draw edges for valid register indexes
        if 0 <= src0 < num_registers:
            lines.append(f'    r{src0} -> n{i} [label="w0={w0:.2f}"];')
        if 0 <= src1 < num_registers:
            lines.append(f'    r{src1} -> n{i} [label="w1={w1:.2f}"];')
        if 0 <= dst < num_registers:
            lines.append(f'    n{i} -> r{dst};')

    lines.append('  }')
    lines.append('}')
    return "\n".join(lines)


def print_text_summary(policy: "GraphPolicy", max_nodes: int = 16):
    """Small human-readable dump in the terminal."""
    node_params = np.array(policy.node_params)
    num_nodes = node_params.shape[0]
    num_registers = getattr(policy, "num_registers", 32)

    print(f"GraphPolicy summary:")
    print(f"  registers: {num_registers}")
    print(f"  nodes:     {num_nodes}")
    print()

    max_nodes = min(max_nodes, num_nodes)
    print(f"First {max_nodes} nodes:")
    print("idx | src0 src1 -> dst | op       | w0      w1      bias    gate")
    print("----+------------------+----------+-------------------------------")
    for i in range(max_nodes):
        row = node_params[i]
        src0 = int(row[IDX_SRC0])
        src1 = int(row[IDX_SRC1])
        dst  = int(row[IDX_DST])
        op   = op_to_name(int(row[IDX_OP]))
        w0   = float(row[IDX_W0])
        w1   = float(row[IDX_W1])
        bias = float(row[IDX_BIAS])
        gate = float(row[IDX_GATE])

        print(
            f"{i:3d} | {src0:4d} {src1:4d} -> {dst:3d} | "
            f"{op:8s} | {w0:+7.3f} {w1:+7.3f} {bias:+7.3f} {gate:+7.3f}"
        )


def main():
    ap = argparse.ArgumentParser(description="Visualize a GraphPolicy as a DOT graph.")
    ap.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to .npz graph policy (e.g. artifacts/best_graph_policy.npz)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="graph.dot",
        help="Output DOT file path (default: graph.dot)",
    )
    ap.add_argument(
        "--max_nodes",
        type=int,
        default=64,
        help="Limit number of nodes shown (default: 64)",
    )
    args = ap.parse_args()

    if not HAS_GRAPH:
        raise SystemExit("GraphPolicy not available (eng/policies_graph.py not importable).")

    policy = load_policy_npz(args.policy)
    if not isinstance(policy, GraphPolicy):
        raise SystemExit(f"Loaded policy from {args.policy} is not a GraphPolicy, got {type(policy)}")

    print_text_summary(policy, max_nodes=args.max_nodes)

    dot = build_dot(policy, max_nodes=args.max_nodes)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(dot)

    print()
    print(f"DOT graph written to: {args.out}")
    print("Render with e.g.:")
    print(f"  dot -Tpng {args.out} -o graph.png")


if __name__ == "__main__":
    main()