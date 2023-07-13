import json
import os
from pathlib import Path
from timeit import Timer

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from instances import grid_from_graph
from solving import solve


def save_and_display_benchmark(node_counts, runtimes, graph_type, dirname):
    results = {int(node_counts[i]): runtimes[i] for i, n in enumerate(node_counts)}
    qcoeffs = np.polyfit(x=np.array(node_counts), y=np.array(runtimes), deg=2)
    quadratic_fit = np.poly1d(qcoeffs)
    results["quadratic_fit"] = f"({qcoeffs[0]:.4}) x^2 + ({qcoeffs[1]:.4}) x + ({qcoeffs[2]:.4})"
    out_dir = Path(dirname)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / f"{graph_type}.json", "w") as file:
        json.dump(results, file, indent=4)
    plt.plot(node_counts, runtimes)
    polyline = np.linspace(node_counts[0], node_counts[-1], 100)
    plt.plot(polyline, quadratic_fit(polyline), label=results["quadratic_fit"])
    plt.title("Total runtime (transforming graph, building QCQP, solving QCQP,\n"
              f"merging anti-parallel flows) for {graph_type} graph of "
              f"{'2n' if graph_type == 'circular ladder' else 'n'} nodes")
    plt.xlabel("n")
    plt.ylabel("s")
    plt.legend()
    plt.savefig(out_dir / f"{graph_type}.pdf", bbox_inches='tight')
    plt.show()


def benchmark(graph_type, largest_n, repeats=5, num_unique_n=30, dirname="benchmarks"):
    node_counts = np.linspace(1, largest_n, num_unique_n, dtype=int)
    running_order = np.random.default_rng().permutation(np.repeat(range(len(node_counts)), repeats))
    recorded_runtimes = {n: [] for n in node_counts}
    for i in tqdm(running_order):
        recorded_runtimes[node_counts[i]].append(Timer(
            lambda: solve(grid_from_graph(
                (nx.cycle_graph if graph_type == "cycle" else
                 nx.circular_ladder_graph if graph_type == "circular ladder" else
                 nx.complete_graph if graph_type == "complete" else None)(node_counts[i])), verbosity=0)
        ).timeit(number=1))
    runtimes = [np.median(recorded_runtimes[n]) for n in node_counts]
    save_and_display_benchmark(node_counts, runtimes, graph_type, dirname)


def load_benchmark(graph_type, dirname="benchmarks"):
    with open(f"benchmarks/{graph_type}.json", "r") as file:
        results = json.load(file)
        del results["quadratic_fit"]
    node_counts = [int(n) for n in results.keys()]
    runtimes = [float(v) for v in results.values()]
    save_and_display_benchmark(node_counts, runtimes, graph_type, dirname)
