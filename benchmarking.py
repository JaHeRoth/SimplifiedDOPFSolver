import json
import os
from math import ceil
from pathlib import Path
from timeit import Timer
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm

import networkx as nx
import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from instances import grid_from_graph
from solving import solve


def save_and_display_benchmark(node_counts, runtimes, graph_type, dirname):
    results = {int(node_counts[i]): runtimes[i] for i, n in enumerate(node_counts)}
    qfit = Polynomial.fit(x=node_counts, y=runtimes, deg=2).convert()
    results["quadratic_fit"] = f"({qfit.coef[2]:.2}) x^2 + ({qfit.coef[1]:.2}) x + ({qfit.coef[0]:.2})"
    out_dir = Path(dirname)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / f"{graph_type}.json", "w") as file:
        json.dump(results, file, indent=4)
    plt.plot(node_counts, runtimes, "s", label="Measurements", zorder=4)
    plt.plot(node_counts, qfit(np.array(node_counts)), "r", label=results["quadratic_fit"], zorder=2)
    # cubic_fit = np.poly1d(np.polyfit(node_counts, runtimes, deg=3))
    # d, c, b, a = cubic_fit
    # plt.plot(polyline, cubic_fit(polyline), "k", label=f"({d:.2}) x^3 + ({c:.2}) x^2 + ({b:.2}) x + {a:.2}", zorder=2)
    if graph_type == "complete":
        pfit = Polynomial.fit(x=node_counts, y=runtimes, deg=4).convert()
        plt.plot(node_counts, pfit(np.array(node_counts)), "k",
                 label=f"({pfit.coef[4]:.2}) x^4 + ({pfit.coef[3]:.2}) x^3 + ({pfit.coef[2]:.2}) x^2 + "
                       f"({pfit.coef[1]:.2}) x + {pfit.coef[0]:.2}", zorder=1)
    else:
        lfit = Polynomial.fit(x=node_counts, y=runtimes, deg=1).convert()
        plt.plot(node_counts, lfit(np.array(node_counts)), "k",
                 label=f"({lfit.coef[1]:.2}) x + {lfit.coef[0]:.2}", zorder=3)
    plt.title("Total execution time (transforming graph, building QCQP,\n"
              f" solving QCQP, merging anti-parallel flows) for {graph_type} graphs")
    plt.xlabel("Node count")
    plt.ylabel("Execution time (seconds)")
    plt.legend()
    plt.savefig(out_dir / f"{graph_type}.pdf", bbox_inches='tight')
    plt.show()


def record_runtime(graph_type, n):
    return Timer(
        lambda: solve(grid_from_graph(
            (nx.cycle_graph if graph_type == "cycle" else
             nx.circular_ladder_graph if graph_type == "circular ladder" else
             nx.complete_graph if graph_type == "complete" else None)(n)
        ), verbosity=0)
    ).timeit(number=1)


def record_runtimes(graph_type, max_nodes, repeats=5, num_unique_n=30):
    nodes_per_n = 2 if graph_type == 'circular ladder' else 1
    n_counts = np.linspace(1, max_nodes // nodes_per_n, num_unique_n, dtype=int)
    node_counts = n_counts * nodes_per_n
    running_order = np.random.default_rng().permutation(np.repeat(range(len(n_counts)), repeats))
    recorded_runtimes = {node_count: [] for node_count in node_counts}
    for i in tqdm(running_order):
        recorded_runtimes[node_counts[i]].append(record_runtime(graph_type, n_counts[i]))
    outdir = Path("output/raw")
    os.makedirs(outdir, exist_ok=True)
    with open(outdir / f"{graph_type}_{max_nodes}_{repeats}_{num_unique_n}.json", "w") as file:
        json.dump({str(key): value for key, value in recorded_runtimes.items()}, file, indent=4)
    return node_counts, recorded_runtimes


def benchmark(graph_type, max_nodes, repeats=5, num_unique_n=30, dirname="benchmarks"):
    node_counts, recorded_runtimes = record_runtimes(graph_type, max_nodes, repeats, num_unique_n)
    runtimes = [np.median(recorded_runtimes[node_count]) for node_count in node_counts]
    save_and_display_benchmark(node_counts, runtimes, graph_type, dirname)


def load_benchmark(graph_type, dirname="benchmarks"):
    with open(f"{dirname}/{graph_type}.json", "r") as file:
        results = json.load(file)
        del results["quadratic_fit"]
    node_counts = [int(n) for n in results.keys()]
    runtimes = [float(v) for v in results.values()]
    save_and_display_benchmark(node_counts, runtimes, graph_type, dirname)


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


def record_structured_runtimes(graph_type, max_nodes, repeats_per_record=3, num_unique_n=10, num_qcoeffs=40, dirname="output"):
    benchmarks = []
    for i in range(num_qcoeffs):
        node_counts, recorded_runtimes = record_runtimes(graph_type, max_nodes, repeats_per_record, num_unique_n)
        runtimes = [float(np.median(recorded_runtimes[node_count])) for node_count in node_counts]
        benchmarks.append({"node_count": node_counts.tolist(), "execution_time": runtimes})
    out_dir = Path(dirname)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / f"execution_times.json", "w") as file:
        json.dump(benchmarks, file, indent=4)


def remove_top_percentile(df):
    # df = df.sort_values(by="node_count")
    df["bin"] = pd.cut(df.node_count, int(len(df) / 40))
    bin_percentiles = df.groupby("bin").execution_time.apply(lambda y: y.quantile(0.95))
    return df[df.execution_time <= df.bin.map(bin_percentiles).astype("float")].drop("bin", axis=1)


def remove_outliers(df):
    model = IsolationForest(contamination=0.05)
    model.fit(df)
    outliers = model.predict(df)
    return df[outliers == 1]


def sampled_confidence_intervals(dirname="output"):
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


def find_variances(graph_type, max_nodes, repeats_per_record=3, num_unique_n=25, reconds_per_n=100):
    benchmarks = []
    for i in range(reconds_per_n):
        node_counts, recorded_runtimes = record_runtimes(graph_type, max_nodes, repeats_per_record, num_unique_n)
        runtimes = [float(np.median(recorded_runtimes[node_count])) for node_count in node_counts]
        benchmarks.append({"node_count": node_counts.tolist(), "execution_time": runtimes})
    plot_variances(benchmarks)


def plot_variances(benchmarks=None, dirname="output"):
    if benchmarks is None:
        with open(f"{dirname}/execution_times.json", "r") as file:
            benchmarks = json.load(file)

    node_counts = benchmarks[0]["node_count"]
    execution_times_per_node_counts = [[b["execution_time"][i] for b in benchmarks] for i in range(len(node_counts))]
    std_per_node_count = [np.array(times).std() for times in execution_times_per_node_counts]
    plt.plot(node_counts, std_per_node_count)
    plt.show()

    all_node_counts = np.array([b["node_count"] for b in benchmarks]).flatten()
    all_execution_times = np.array([b["execution_time"] for b in benchmarks]).flatten()
    pd.DataFrame({"nodes": all_node_counts, "seconds": all_execution_times}).groupby("nodes").std().plot()
    plt.show()
    print("success")


def estimate_weights(benchmarks, steps=1):
    all_node_counts = np.array([b["node_count"] for b in benchmarks]).flatten()
    all_execution_times = np.array([b["execution_time"] for b in benchmarks]).flatten()
    df = pd.DataFrame({"nodes": all_node_counts, "seconds": all_execution_times})
    min_nodes, max_nodes = all_node_counts.min(), all_node_counts.max()

    weights = None
    for _ in range(steps):
        y_fit = Polynomial.fit(x=df.nodes, y=df.seconds, deg=2, w=weights, window=(min_nodes, max_nodes))
        squared_residuals = (df.seconds - y_fit(df.nodes))**2
        s_res_fit = Polynomial.fit(x=df.nodes, y=squared_residuals, deg=2, window=(min_nodes, max_nodes))
        weights = 1/np.sqrt(s_res_fit(df.nodes))
        print(weights)
        plt.scatter(df.nodes, weights)
        df.groupby("nodes").std().plot()
        plt.show()
    return weights


def structured_confidence_intervals(dirname="output"):
    with open(f"{dirname}/execution_times.json", "r") as file:
        benchmarks = json.load(file)
    dirname = "output/ols"
    os.makedirs(dirname, exist_ok=True)

    recorded_qcoeffs = [
        Polynomial.fit(
            x=b["node_count"], y=b["execution_time"], deg=2, w=None  # 1/np.maximum(2000, np.array(b["node_count"]))
        ).convert().coef[2] for b in benchmarks
    ]
    print(f"Empirical 95% confidence interval of quadratic coefficient: "
          f"[{np.quantile(recorded_qcoeffs, 0.025):.2}, "
          f"{np.quantile(recorded_qcoeffs, 0.975):.2}]")
    plt.hist(recorded_qcoeffs, bins=10, density=True)
    plt.xlabel("Quadratic coefficient in best quadratic fit")
    plt.savefig(f"{dirname}/qcoeffs_hist.pdf", bbox_inches='tight')
    plt.show()
    print(np.sort(recorded_qcoeffs))

    recorded_lcoeffs = [
        Polynomial.fit(x=b["node_count"], y=b["execution_time"], deg=2, w=None).convert().coef[1] for b in benchmarks
    ]
    print(f"Empirical 95% confidence interval of linear coefficient: "
          f"[{np.quantile(recorded_lcoeffs, 0.025):.2}, "
          f"{np.quantile(recorded_lcoeffs, 0.975):.2}]")
    plt.hist(recorded_lcoeffs, bins=10, density=True)
    plt.xlabel("Linear coefficient in best quadratic fit")
    plt.savefig(f"{dirname}/lcoeffs_hist.pdf", bbox_inches='tight')
    plt.show()

    print(f"{sum(np.array(recorded_qcoeffs) > 0)}/{len(recorded_qcoeffs)} quadratic coefficients are positive.")
    all_node_counts = np.array([b["node_count"] for b in benchmarks]).flatten()
    all_execution_times = np.array([b["execution_time"] for b in benchmarks]).flatten()
    plt.scatter(all_node_counts, all_execution_times, marker=".", label="Measurements", zorder=3)
    for b in benchmarks:
        n_counts = b["node_count"]
        plt.plot(n_counts, Polynomial.fit(x=n_counts, y=b["execution_time"], deg=2, w=None)(np.array(n_counts)), zorder=1)
    plt.xlabel("Node count")
    plt.ylabel("Execution time (seconds)")
    plt.legend()
    plt.savefig(f"{dirname}/scatter_with_quadratics.pdf", bbox_inches='tight')
    plt.show()

    node_counts = benchmarks[0]["node_count"]
    execution_times = benchmarks[0]["execution_time"]
    qfit = Polynomial.fit(x=node_counts, y=execution_times, deg=2, w=None, window=(node_counts[0], node_counts[-1]))
    plt.plot(node_counts, execution_times, "s", label="Measurements", zorder=3)
    plt.plot(node_counts, qfit(np.array(node_counts)), "r", label=str(qfit), zorder=1)
    plt.legend()
    plt.show()
