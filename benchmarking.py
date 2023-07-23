import json
import os
from math import ceil
from pathlib import Path
from timeit import Timer

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats
from tqdm import tqdm

from instances import grid_from_graph
from solving import solve


def save_and_display_benchmark(node_counts, runtimes, graph_type, dirname):
    results = {int(node_counts[i]): runtimes[i] for i, n in enumerate(node_counts)}
    qcoeffs = np.polyfit(x=np.array(node_counts), y=np.array(runtimes), deg=2)
    quadratic_fit = np.poly1d(qcoeffs)
    results["quadratic_fit"] = f"({qcoeffs[0]:.3}) x^2 + ({qcoeffs[1]:.3}) x + ({qcoeffs[2]:.3})"
    out_dir = Path(dirname)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / f"{graph_type}.json", "w") as file:
        json.dump(results, file, indent=4)
    plt.plot(node_counts, runtimes, "s", label="Measurements", zorder=3)
    polyline = np.linspace(node_counts[0], node_counts[-1], 100)
    plt.plot(polyline, quadratic_fit(polyline), "r", label=results["quadratic_fit"], zorder=1)
    # cubic_fit = np.poly1d(np.polyfit(node_counts, runtimes, deg=3))
    # d, c, b, a = cubic_fit
    # plt.plot(polyline, cubic_fit(polyline), "k", label=f"({d:.2}) x^3 + ({c:.2}) x^2 + ({b:.2}) x + {a:.2}", zorder=2)
    linear_fit = np.poly1d(np.polyfit(node_counts, runtimes, deg=1))
    s, i = linear_fit
    plt.plot(polyline, linear_fit(polyline), "k", label=f"({s:.4}) x + {i:.4}", zorder=2)
    plt.title("Total execution time (transforming graph, building QCQP,\n"
              f" solving QCQP, merging anti-parallel flows) for {graph_type} graphs")
    plt.xlabel("Node count")
    plt.ylabel("Execution time (seconds)")
    plt.legend()
    plt.savefig(out_dir / f"{graph_type}.pdf", bbox_inches='tight')
    plt.show()


def benchmark(graph_type, max_nodes, repeats=5, num_unique_n=30, dirname="benchmarks"):
    nodes_per_n = 2 if graph_type == 'circular ladder' else 1
    n_counts = np.linspace(1, max_nodes // nodes_per_n, num_unique_n, dtype=int)
    node_counts = n_counts * nodes_per_n
    running_order = np.random.default_rng().permutation(np.repeat(range(len(n_counts)), repeats))
    recorded_runtimes = {node_count: [] for node_count in node_counts}
    for i in tqdm(running_order):
        recorded_runtimes[node_counts[i]].append(record_runtime(graph_type, n_counts[i]))
    runtimes = [np.median(recorded_runtimes[node_count]) for node_count in node_counts]
    save_and_display_benchmark(node_counts, runtimes, graph_type, dirname)


def load_benchmark(graph_type, dirname="benchmarks"):
    with open(f"benchmarks/{graph_type}.json", "r") as file:
        results = json.load(file)
        del results["quadratic_fit"]
    node_counts = [int(n) for n in results.keys()]
    runtimes = [float(v) for v in results.values()]
    save_and_display_benchmark(node_counts, runtimes, graph_type, dirname)


def record_runtime(graph_type, n):
    return Timer(
        lambda: solve(grid_from_graph(
            (nx.cycle_graph if graph_type == "cycle" else
             nx.circular_ladder_graph if graph_type == "circular ladder" else
             nx.complete_graph if graph_type == "complete" else None)(n)
        ), verbosity=0)
    ).timeit(number=1)


def sample_runtime(graph_type, max_nodes, nodes_per_n):
    n = np.random.randint(1, max_nodes // nodes_per_n)
    exec_time = Timer(
        lambda: solve(grid_from_graph(
            (nx.cycle_graph if graph_type == "cycle" else
             nx.circular_ladder_graph if graph_type == "circular ladder" else
             nx.complete_graph if graph_type == "complete" else None)(n)
        ), verbosity=0)
    ).timeit(number=1)
    return {"node_count": n * nodes_per_n, "execution_time": exec_time}


def sample_runtimes(graph_type, max_nodes, num_samples=ceil(120/0.95*10), dirname="output"):
    nodes_per_n = 2 if graph_type == 'circular ladder' else 1
    df = pd.DataFrame([sample_runtime(graph_type, max_nodes, nodes_per_n) for _ in tqdm(range(num_samples))])
    out_dir = Path(dirname)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_dir / "execution_times.csv")


def remove_outliers(df):
    # df = df.sort_values(by="node_count")
    df["bin"] = pd.cut(df.node_count, int(len(df) / 40))
    bin_percentiles = df.groupby("bin").execution_time.apply(lambda y: y.quantile(0.95))
    return df[df.execution_time <= df.bin.map(bin_percentiles).astype("float")].drop("bin", axis=1)


def confidence_intervals(dirname="output"):
    df = remove_outliers(pd.read_csv(Path(dirname) / "execution_times.csv", index_col=0))
    df.to_csv(Path(dirname) / "filtered_execution_times.csv")
    sampled_dfs = np.array_split(df.sample(frac=1), 120)
    sampled_qcoeffs = [
        np.poly1d(np.polyfit(x=np.array(sampled.node_count), y=np.array(sampled.execution_time), deg=2))[0] for sampled
        in sampled_dfs]
    print(f"Empirical 95% confidence interval of quadratic coefficient: "
          f"[{np.quantile(sampled_qcoeffs, 0.025):.4}, {np.quantile(sampled_qcoeffs, 0.975):.4}]")
    # plt.hist(sampled_qcoeffs, density=True, bins=40)
    # plt.show()
    # sampled_qcoeffs = [
    #    np.poly1d(np.polyfit(x=np.array(sampled.node_count), y=np.array(sampled.execution_time), deg=2))[1] for sampled
    #     in sampled_dfs]
    # plt.scatter(x=sampled_lcoeffs, y=sampled_qcoeffs)
    # plt.show()
    # df.plot(x="node_count", y="execution_time", kind="scatter")
    # plt.show()
    # df.execution_time.hist(bins=50)
    # plt.show()
