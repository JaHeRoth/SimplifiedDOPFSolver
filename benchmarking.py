import json
import os
import time
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

from instances import from_graph
from solving import solve


def run_and_display(graph_type, max_nodes, repeats_per_record=3, num_unique_n=10, num_runs=40, overwrite=True):
    timestamp = time.time() if not overwrite else None
    record(graph_type, max_nodes, repeats_per_record, num_unique_n, num_runs, timestamp)
    load_and_display(graph_type, max_nodes, repeats_per_record, num_unique_n, num_runs, timestamp)


def load_and_display(graph_type, max_nodes, repeats_per_record, num_unique_n, num_runs, timestamp=None):
    raw, working_dir = retrieve(graph_type, max_nodes, repeats_per_record, num_unique_n, num_runs, timestamp)
    measurements, nodes, benchmarks = wrangle(raw)
    if len(raw) == 1:
        output_simple(measurements, benchmarks, graph_type, outdir=working_dir)
    else:
        output_comprehensive(measurements, nodes, benchmarks, outdir=working_dir)


def record(graph_type, max_nodes, repeats_per_record, num_unique_n, num_runs, timestamp=None):
    benchmarks = []
    pbar = tqdm(range(num_runs), desc="Benchmarks completed")
    for _ in pbar:
        node_counts, recorded_runtimes = record_runtimes(graph_type, max_nodes, pbar, repeats_per_record, num_unique_n)
        runtimes = [float(np.median(recorded_runtimes[node_count])) for node_count in node_counts]
        benchmarks.append({"node_count": node_counts.tolist(), "execution_time": runtimes})
    outdir = dir_from_params(graph_type, max_nodes, repeats_per_record, num_unique_n, num_runs, timestamp)
    with open(outdir / f"execution_times.json", "w") as file:
        json.dump(benchmarks, file, indent=4)


def dir_from_params(graph_type, max_nodes, repeats_per_record, num_unique_n, num_runs, timestamp=None):
    path = f"output/benchmarking/{num_runs}_{graph_type}_{max_nodes}_{num_unique_n}_{repeats_per_record}"
    if timestamp is not None:
        path += f"_{timestamp}"
    os.makedirs(path, exist_ok=True)
    return Path(path)


def record_runtimes(graph_type, max_nodes, pbar, repeats=5, num_unique_n=30):
    nodes_per_n = 2 if graph_type == 'circular ladder' else 1
    n_counts = np.linspace(1, max_nodes // nodes_per_n, num_unique_n, dtype=int)
    node_counts = n_counts * nodes_per_n
    running_order = np.random.default_rng().permutation(np.repeat(range(len(n_counts)), repeats))
    recorded_runtimes = {node_count: [] for node_count in node_counts}
    for i, j in enumerate(running_order):
        recorded_runtimes[node_counts[j]].append(record_runtime(graph_type, n_counts[j]))
        pbar.set_postfix({f"Progress on benchmark (out of {len(running_order)})": i})
    return node_counts, recorded_runtimes


def record_runtime(graph_type, n):
    graph = from_graph(
            (nx.cycle_graph if graph_type == "cycle" else
             nx.circular_ladder_graph if graph_type == "circular ladder" else
             nx.complete_graph if graph_type == "complete" else None)(n)
        )
    return Timer(lambda: solve(graph, verbosity=0)).timeit(number=1)


def retrieve(graph_type, max_nodes, repeats_per_record, num_unique_n, num_runs, timestamp=None):
    indir = dir_from_params(graph_type, max_nodes, repeats_per_record, num_unique_n, num_runs, timestamp)
    with open(indir / f"execution_times.json", "r") as file:
        return json.load(file), indir


def wrangle(benchmarks: list[dict[str, list]]):
    dfs = [pd.DataFrame({"nodes": b["node_count"],
                         "seconds": b["execution_time"],
                         "benchmark": np.repeat(i, len(b["node_count"]))})
           for i, b in enumerate(benchmarks)]
    df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    nodes = pd.DataFrame({"standard_deviation": df.groupby("nodes").seconds.std()})
    benchmarks = df.groupby("benchmark").apply(
        lambda x: pd.Series({"linear_fit": Polynomial.fit(x=x.nodes, y=x.seconds, deg=1).convert(),
                             "quadratic_fit": Polynomial.fit(x=x.nodes, y=x.seconds, deg=2).convert(),
                             "fourth_fit": Polynomial.fit(x=x.nodes, y=x.seconds, deg=4).convert()})
    )
    return df, nodes, benchmarks


def decorate_save_show(dst: Path, xlabel=None, ylabel=None, title=None, legend=False):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if legend:
        plt.legend()
    plt.savefig(dst, bbox_inches='tight')
    plt.show()


def output_comprehensive(measurements: pd.DataFrame, nodes: pd.DataFrame, benchmarks: pd.DataFrame, outdir: Path):
    recorded_qcoeffs = benchmarks.quadratic_fit.apply(lambda fit: fit.coef[2])
    print(f"Empirical, two-sided 95% confidence interval of quadratic coefficient in OLS quadratic fit: "
          f"[{np.quantile(recorded_qcoeffs, 0.025):.2}, "
          f"{np.quantile(recorded_qcoeffs, 0.975):.2}]")
    plt.hist(recorded_qcoeffs, bins=10, density=True)
    decorate_save_show(outdir/"qcoeffs_hist.pdf", xlabel="Quadratic coefficient in OLS quadratic fit")

    recorded_lcoeffs = benchmarks.quadratic_fit.apply(lambda fit: fit.coef[1])
    print(f"Empirical, two-sided 95% confidence interval of linear coefficient in OLS quadratic fit: "
          f"[{np.quantile(recorded_lcoeffs, 0.025):.2}, "
          f"{np.quantile(recorded_lcoeffs, 0.975):.2}]")
    plt.hist(recorded_lcoeffs, bins=10, density=True)
    decorate_save_show(outdir/"lcoeffs_hist.pdf", xlabel="Linear coefficient in OLS quadratic fit")

    plt.scatter(measurements.nodes, measurements.seconds, marker=".", label="Measurements", zorder=3)
    measurements.join(benchmarks, on="benchmark").groupby("benchmark").apply(
        lambda x: plt.plot(x.nodes, x.quadratic_fit.iloc[0](x.nodes), zorder=1)
    )
    decorate_save_show(dst=outdir/"scatter.pdf", xlabel="Node count", ylabel="Execution time (seconds)", title="All benchmarks", legend=True)

    plt.plot(nodes.index, nodes.standard_deviation)
    decorate_save_show(
        dst=outdir / "std.pdf",
        xlabel="Node count",
        ylabel="Standard deviation in execution time (seconds)",
        title="Testing homoscedasticity")


def print_fit(fit):
    res = f"{fit.coef[0]:.2}"
    for i, c in enumerate(fit.coef[1:]):
        res += f" + {c:.2} x^{i+1}" if c > 0 else f" - {abs(c):.2} x^{i+1}"
    return res


def output_simple(measurements: pd.DataFrame, benchmarks: pd.DataFrame, graph_type: str, outdir: Path):
    ncounts, seconds, qfit = measurements.nodes, measurements.seconds, benchmarks.quadratic_fit.iloc[0]
    plt.plot(ncounts, seconds, "s", label="Measurements", zorder=4)
    if graph_type == "complete":
        simplefit = benchmarks.quadratic_fit.iloc[0]
        complexfit = benchmarks.fourth_fit.iloc[0]
    else:
        simplefit = benchmarks.linear_fit.iloc[0]
        complexfit = benchmarks.quadratic_fit.iloc[0]
    plt.plot(ncounts, simplefit(ncounts), "r", label=print_fit(simplefit), zorder=2)
    plt.plot(ncounts, complexfit(ncounts), "k", label=print_fit(complexfit), zorder=1)
    decorate_save_show(
        dst=outdir/"scatter.pdf",
        xlabel="Node count",
        ylabel="Execution time (seconds)",
        title="Total execution time (transforming graph, building QCQP,\n"
              f" solving QCQP, merging anti-parallel flows) for {graph_type} graphs",
        legend=True)
