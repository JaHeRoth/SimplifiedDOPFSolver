import os
import time
from itertools import islice
from pathlib import Path

import gurobipy
import networkx as nx
from matplotlib import pyplot as plt


def print_flow(cost: float, flow: dict, merged: bool, max_values_printed=20):
    print("---------------------------------------------")
    # Merging antiparallel flows will typically reduce the cost slightly. As `cost` is the cost
    #  of the unmerged flow it will then be a slight overapproximation of the cost of the merged flow.
    print(f"Obtained solution of value {'at most' if merged else ''} {cost}:")
    for name, value in islice(flow.items(), max_values_printed):
        print(f"\t{name}: {value}")
    if len(flow) > max_values_printed:
        print(f"\t... ({len(flow) - max_values_printed} more elements)")
    print("---------------------------------------------")


def plot_flow(flow: dict, G: nx.Graph):
    for i in range(G.graph["k"]):
        Gi = nx.DiGraph()
        edge_intensity, edge_use = [], {}
        for u, v, attr in G.edges(data=True):
            ui, vi = f"{u}'{i}", f"{v}'{i}"
            if f"x_{(ui, vi)}" in flow:
                Gi.add_edge(u, v)
                edge_use[(u, v)] = flow[f"x_{(ui, vi)}"] / attr["u"]
                edge_intensity.append(edge_use[(u, v)])
            else:
                Gi.add_edge(v, u)
                edge_use[(v, u)] = flow[f"x_{(vi, ui)}"] / attr["u"]
                edge_intensity.append(edge_use[(v, u)])
        node_color = ["#008000" if "c" in G.nodes[node] or "cr" in G.nodes[node] else "#1f78b4" for node in Gi.nodes]
        edge_labels = {edge: f"{round(use * 100)}%" for edge, use in edge_use.items()}
        plt.title(f"Edge utilization (teal = 0%, pink = 100%) at time-step {i + 1},\n"
                  f"with source nodes marked in green")
        # G is used over Gi for positioning as this makes for a much more nicely and evenly spread-out graph
        pos = nx.spring_layout(G)
        if len(Gi.nodes) < 20:
            nx.draw_networkx(Gi, pos, node_color=node_color,
                             edge_cmap=plt.get_cmap("cool"), edge_color=edge_intensity, edge_vmin=0, edge_vmax=1)
            nx.draw_networkx_edge_labels(Gi, pos, edge_labels=edge_labels, font_size=8)
        else:
            nx.draw_networkx(Gi, pos, node_size=50, with_labels=False, node_color=node_color,
                             edge_cmap=plt.get_cmap("cool"), edge_color=edge_intensity, edge_vmin=0, edge_vmax=1)
        safe_savefig("output/algorithm", f"{time.time()}.pdf")
        plt.show()


def plot_graph(graph: nx.Graph | nx.DiGraph):
    nx.draw_networkx(graph)
    safe_savefig("output/algorithm", f"{graph.graph['name']}_{time.time()}.pdf")
    plt.show()


def safe_savefig(outdir: str, filename: str):
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(Path(outdir) / filename, bbox_inches='tight')


def safe_savemodel(outdir: str, model: gurobipy.Model):
    os.makedirs(outdir, exist_ok=True)
    model.write(f"{outdir}/model.lp")

