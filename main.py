import json
import os
from datetime import datetime
from timeit import Timer

import numpy as np
from matplotlib import pyplot as plt
from gurobipy import Model, quicksum, GRB
import time
import grid2op
import networkx as nx
from tqdm import tqdm


def plot_multigraph(G, filename):
    pos = nx.spring_layout(G)
    names = {name: name for name in G.nodes}
    nx.draw_networkx_nodes(G, pos, node_color='b', node_size=500, alpha=1)
    nx.draw_networkx_labels(G, pos, names, font_size=10, font_color='w')
    ax = plt.gca()
    for e in G.edges:
        ax.annotate("",
                    xy=pos[e[1]], xycoords='data',
                    xytext=pos[e[0]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0",
                                    shrinkA=10, shrinkB=10,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * e[2])
                                                                           ),
                                    ),
                    )
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


def find_optimal_flow(G, verbose=False):
    qcqp = Model(name="QCQP")
    x, y = {}, {}
    for a in G.edges:
        x[a] = qcqp.addVar(name=f"x_{a}", lb=0, obj=G.edges[a]["c"])
        y[a] = qcqp.addVar(name=f"y_{a}", lb=0)
        qcqp.addConstr(G.edges[a]["r"] * x[a] ** 2 - x[a] + y[a] <= 0)
        qcqp.addConstr(x[a] <= G.edges[a]["u"])
    for v in G.nodes:
        in_flows = [y[a] for a in G.in_edges(v, keys=True)]
        out_flows = [x[a] for a in G.out_edges(v, keys=True)]
        if v == "d":
            qcqp.addConstr(quicksum(in_flows) == G.nodes[v]["d"])
        elif v != "s":
            qcqp.addConstr(quicksum(in_flows) == quicksum(out_flows))
    qcqp.modelSense = GRB.MINIMIZE
    qcqp.Params.LogToConsole = int(verbose)
    # qcqp.Params.BarQCPConvTol = 0
    # qcqp.Params.FeasibilityTol = 1e-9
    qcqp.optimize()
    qcqp.write("qcqp.lp")
    variables = {var.VarName: var.X for var in qcqp.getVars()}
    return qcqp.objVal, variables


def print_flow(cost, flow, merged=False):
    print("---------------------------------------------")
    # Merging antiparallel flows will typically reduce the cost. As `cost` is the cost of the unmerged flow it
    # thus tends to be a slight overapproximation of the cost of the merged flow
    print(f"Obtained solution of value {'at most' if merged else ''} {cost}:")
    if len(flow) < 50:
        for name, value in flow.items():
            print(f"\t{name}: {value}")
    print("---------------------------------------------")


def algorithm2(G, T):
    sources, sinks, arcs = G.graph["sources"], G.graph["sinks"], G.edges
    Gp = nx.MultiDiGraph()
    Gp.add_node("d", d=sum(attr["d"] for _, attr in sinks))
    for u, v, attr in arcs(data=True):
        Gp.add_edge(u, v, c=0, r=attr["r"], u=attr["u"])
    for src, attr in sources:
        for length, cost in attr["c"]:
            Gp.add_edge("s", src, c=T * cost, r=0, u=length / T)
    for sink, attrs in sinks:
        Gp.add_edge(sink, "d", c=0, r=0, u=attrs["d"])
    return Gp


def algorithm3(G):
    # Steps 1 & 4
    src_connected = set(nx.multi_source_dijkstra_path_length(
        G, [v for v, _ in G.graph["sources"]]).keys())
    sink_connected = set(nx.multi_source_dijkstra_path_length(
        G, [v for v, _ in G.graph["sinks"]]).keys())
    relevant = src_connected.intersection(sink_connected)
    Gp = nx.MultiGraph(G).subgraph(relevant).to_directed()
    Gp.graph["sources"] = []
    Gp.graph["sinks"] = []
    # Step 2
    for src, attr in G.graph["sources"]:
        if "cr" in attr:
            # Step 2a
            for i, (length, cost) in enumerate(attr["cr"]):
                node = f"{src}_s{i}"
                Gp.graph["sources"].append((node, {"c": [(length, cost)]}))
                Gp.add_node(node, c=cost)
                Gp.add_edge(node, src, r=0, u=length)
            del Gp.nodes[src]["cr"]
        else:
            # Step 2b
            node = f"{src}_s"
            Gp.graph["sources"].append((node, attr))
            Gp.add_node(node, c=attr["c"])
            csupply = sum(l for l, _ in attr["c"])
            Gp.add_edge(node, src, r=0, u=csupply)
            del Gp.nodes[src]["c"]
    # Step 3
    for sink, attr in G.graph["sinks"]:
        node = f"{sink}_d"
        Gp.graph["sinks"].append((node, attr))
        Gp.add_node(node, d=attr["d"]) # TODO: Ensure this only adds attribute to node if it already exists
        Gp.add_edge(sink, node, r=0, u=attr["d"])
        del Gp.nodes[sink]["d"]
    return Gp


def to_original_graph_flow(flow, G):
    """First get rid of variables corresponding to arcs not present in the original graph.
    Then, for each pair of antiparallel arcs, subtract the smaller flow from the larger flow and remove
    the variables corresponding to the smaller flow (adds waste to that arc unless resistance is 0)."""
    flow = flow.copy()
    for arc in G.out_edges("s", keys=True):
        del flow[f"x_{arc}"]
        del flow[f"y_{arc}"]
        # Without this if-statement we'd try to remove out-arcs of arc[1] once per parallel (arc[0],arc[1]) arc
        if arc[2] == 0:
            for narc in G.out_edges(arc[1], keys=True):
                del flow[f"x_{narc}"]
                del flow[f"y_{narc}"]
    for arc in G.in_edges("d", keys=True):
        del flow[f"x_{arc}"]
        del flow[f"y_{arc}"]
        for parc in G.in_edges(arc[0], keys=True):
            del flow[f"x_{parc}"]
            del flow[f"y_{parc}"]
    for arc in G.edges:
        if f"x_{arc}" in flow:
            aarc = (arc[1], arc[0], arc[2])
            # Shortcut way to check if forward flow beats backwards flow, assuming the flows differ by orders of magnitude
            if flow[f"y_{arc}"] > flow[f"x_{aarc}"]:
                flow[f"y_{arc}"] -= flow[f"x_{aarc}"]
                flow[f"x_{arc}"] -= flow[f"y_{aarc}"]
                del flow[f"x_{aarc}"]
                del flow[f"y_{aarc}"]
    return flow


def plot_flow(flow, G):
    arc_capacities = G.graph['capacity'] if type(G.graph['capacity']) is dict else {
        edge: G.graph['capacity'] for edge in G.edges}
    edge_use = {(u, v): (flow.get(f"x_{(u, v, 0)}", 0) + flow.get(f"x_{(v, u, 0)}", 0))
                        / arc_capacities[(u,v)] for u, v in G.edges}
    node_intensity = [max(0, sum(flow.get(f"x_{(node, neigh, 0)}", 0) - flow.get(f"y_{(neigh, node, 0)}", 0)
                          for neigh in G.neighbors(node)) / G.graph["supplies"][node])
                      if node in G.graph["supplies"].keys() else 1
                      for node in G.nodes]
    edge_intensity = [edge_use[edge] for edge in G.edges]
    edge_labels = {edge: f"{round(edge_use[edge]*100)}" for edge in G.edges}
    if len(G.nodes) < 20:
        plt.title("Edge utilization in percent, with generator and edge utilization heat maps")
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos, cmap=plt.get_cmap("cool"), node_color=node_intensity, vmin=0, vmax=1,
                         edge_cmap=plt.get_cmap("cool"), edge_color=edge_intensity, edge_vmin=0, edge_vmax=1)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    else:
        plt.title("Generator and edge utilization heat maps (teal=0, pink=1)")
        nx.draw_networkx(G, node_size=50, with_labels=False, cmap=plt.get_cmap("cool"), node_color=node_intensity,
                         vmin=0, vmax=1,
                         edge_cmap=plt.get_cmap("cool"), edge_color=edge_intensity, edge_vmin=0, edge_vmax=1)
    plt.savefig(f"plots/{time.time()}.pdf", bbox_inches='tight')
    plt.show()


def solve(G, verbosity=1):
    sstart = datetime.now()
    Gp = algorithm3(G)
    Gpp = algorithm2(Gp, 1)
    if verbosity > 0:
        print(f"Graph modifications algorithm ran in {(datetime.now()-sstart).total_seconds():.2} seconds.")
    if verbosity > 2:
        nx.draw_networkx(G)
        plt.savefig(f"plots/G_{time.time()}.pdf", bbox_inches='tight')
        plt.show()
        plot_multigraph(Gp, f"plots/Gp_{time.time()}.pdf")
        plot_multigraph(Gpp, f"plots/Gpp_{time.time()}.pdf")
    start = datetime.now()
    cost, flow = find_optimal_flow(Gpp, verbose=(verbosity > 1))
    if verbosity > 0:
        print(f"Defined and solved Gurobi program in {(datetime.now()-start).total_seconds():.2} seconds.")
    merged_flow = to_original_graph_flow(flow, Gpp)
    total_runtime = (datetime.now()-sstart).total_seconds()
    if verbosity > 0:
        print(f"Obtained merged flow. Total runtime: {total_runtime:.3f} seconds.")
        if verbosity > 1:
            print_flow(cost, merged_flow, merged=True)
        plot_flow(merged_flow, G)
    return cost, merged_flow, total_runtime


def fetch_l2rpn_grid(name):
    env = grid2op.make(name)
    obs = env.reset()
    rawG = obs.get_energy_graph()

    G = nx.MultiGraph(sources=[], sinks=[])
    for node, attr in rawG.nodes(data=True):
        net_out = attr["p"] / 75 * 1e3
        if net_out > 0:
            node_w_data = (node, {"c": [(net_out / 2, 1), (net_out / 2, 2)]})
            G.graph["sources"].append(node_w_data)
            G.add_nodes_from([node_w_data])
        if net_out < 0:
            node_w_data = (node, {"d": -net_out / 2})
            G.graph["sinks"].append(node_w_data)
            G.add_nodes_from([node_w_data])
    for u, v, attr in rawG.edges(data=True):
        arc_w_data = (u, v, {"r": 1e-4, "u": attr["thermal_limit"] * 1e3})
        G.add_edges_from([arc_w_data])
    return G


def make_grid(capacity, resistance, supplies, demands, prodcpus, graph):
    G = nx.Graph(sources=[], sinks=[], capacity=capacity, supplies=supplies)
    for node, attr in graph.nodes(data=True):
        node_w_data = (node, {"d": demands[node]})
        if node in supplies.keys():
            node_w_data[1]["c"] = prodcpus[node]
            G.graph["sources"].append(node_w_data)
        G.graph["sinks"].append(node_w_data)
        G.add_nodes_from([node_w_data])
    for u, v, attr in graph.edges(data=True):
        arc_resistance = resistance[(u, v)] if type(resistance) is dict else resistance
        arc_capacity = capacity[(u, v)] if type(capacity) is dict else capacity
        arc_w_data = (u, v, {"r": arc_resistance, "u": arc_capacity})
        G.add_edges_from([arc_w_data])
    return G


def ieee(capacity, resistance, supplies, demands, prodcpus, env_name="l2rpn_case14_sandbox"):
    env = grid2op.make(env_name)
    obs = env.reset()
    rawG = obs.get_energy_graph()
    return make_grid(capacity, resistance, supplies, demands, prodcpus, rawG)


def trivial_instance():
    G = nx.Graph(sources=[("a", {"c": [(2, 1)]})], sinks=[("b", {"d": 1})], capacity=2, supplies={"a": 2})
    G.add_nodes_from(G.graph["sources"])
    G.add_nodes_from(G.graph["sinks"])
    G.add_edge("a", "v1", r=1e-1, u=2)
    G.add_edge("a", "v2", r=1e-1, u=2)
    G.add_edge("v1", "b", r=1e-1, u=2)
    G.add_edge("v2", "b", r=1e-1, u=2)
    # G.graph["capacity"] = {(u,v): attr["u"] for u, v, attr in G.edges(data=True)}
    return G


def trivial_instance2():
    G = nx.Graph(sources=[("a", {"c": [(2, 1), (2, 2)]})], sinks=[("b1", {"d": 1}), ("b2", {"d": 1})],
                 capacity=2, supplies={"a": 4})
    G.add_nodes_from(G.graph["sources"])
    G.add_nodes_from(G.graph["sinks"])
    G.add_edge("a", "b1", r=1e-1, u=2)
    G.add_edge("a", "b2", r=1e-1, u=2)
    return G


def realistic_instance(kV=-2):
    # Numbers copied or estimated from "Optimal Power Systems Planning for IEEE-14 Bus Test System Application"
    # and Transmission Facts (by AMERICAN ELECTRIC POWER)
    # Note how everything is in MW instead of in current, as these are anyway just a multiple when voltage is constant
    # kV=-1 represents the easiest case, while kV=-2 represents a barely feasible (rather: barely Gurobi-solvable) case
    if kV == 345:
        capacity = 400
        resistance = 41.9 * 1e-6
    elif kV == 500:
        capacity = 900
        resistance = 11 * 1e-6
    elif kV == 765:
        capacity = 2200
        resistance = 3.4 * 1e-6
    elif kV == -1:
        capacity = 1e6
        resistance = 0
    elif kV == -2:
        capacity = 175
        resistance = 95.93999999899998698 * 1e-6
    else:
        raise ValueError
    supplies = {1: 300, 2: 500, 3: 55, 6: 300, 8: 700}
    demands = {node: 1200 / 14 for node in range(14)}
    cost_coeff = {1: (16.91, 0.00048), 2: (17.26, 0.00031), 3: (0, 0), 6: (16.6, 0.002), 8: (16.5, 0.00211)}
    prodcpus = {key: [(supplies[key] / 2, cost_coeff[key][0]),
                (supplies[key] / 2, cost_coeff[key][0] + cost_coeff[key][1] * (supplies[key] / 2) ** 2)]
          for key in supplies.keys()}
    return ieee(capacity, resistance, supplies, demands, prodcpus)


def funky_instance():
    capacity = 200
    resistance = 5e-3
    demands = {node: 50 for node in range(14)}
    supplies = {1: 500, 6: 500, 10: 500, 11: 500}
    prodcpus = {node: [(supplies[node], node+1)] for node in supplies.keys()}
    return ieee(capacity, resistance, supplies, demands, prodcpus)


def funky_instance2():
    node_count = 14
    capacity = 40
    resistance = 1e-2
    demands = {node: 50 for node in range(node_count)}
    supplies = {node: 100 for node in range(node_count)}
    prodcpus = {node: [(supplies[node], node+1)] for node in range(node_count)}
    return ieee(capacity, resistance, supplies, demands, prodcpus)


def big_funky_instance():
    node_count = 118
    capacity = 200
    resistance = 1e-3
    demands = {node: 50 for node in range(node_count)}
    supplies = {node: 100 for node in range(node_count)}
    prodcpus = {node: [(supplies[node], node+1)] for node in range(node_count)}
    return ieee(capacity, resistance, supplies, demands, prodcpus, env_name="l2rpn_wcci_2022")


def grid_from_graph(graph):
    """graph: networkx graph with numbers as node names"""
    capacity = {arc: np.random.randint(0, 200) for arc in graph.edges}
    resistance = {arc: 10 ** -(2 + 2 * np.random.rand()) for arc in graph.edges}
    demands = {node: np.random.randint(0, 50) for node in graph.nodes}
    supplies = {node: np.random.randint(50, 100) for node in graph.nodes}
    prodcpus = {node: [(supplies[node], node + 1)] for node in graph.nodes}
    return make_grid(capacity, resistance, supplies, demands, prodcpus, graph)


def save_and_display_benchmark(node_counts, runtimes, graph_type):
    results = {node_counts[i]: runtimes[i] for i, n in enumerate(node_counts)}
    qcoeffs = np.polyfit(x=np.array(node_counts), y=np.array(runtimes), deg=2)
    quadratic_fit = np.poly1d(qcoeffs)
    results["quadratic_fit"] = f"({qcoeffs[0]:.4}) x^2 + ({qcoeffs[1]:.4}) x + ({qcoeffs[2]:.4})"
    os.makedirs("benchmarks", exist_ok=True)
    with open(f"benchmarks/{graph_type}.json", "w") as file:
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
    plt.savefig(f"benchmarks/{graph_type}.pdf", bbox_inches='tight')
    plt.show()


def benchmark(graph_type):
    repeats = 5
    num_unique_n = 30
    largest_n = int(2e4)
    step_size = int(np.ceil((largest_n - 1) / num_unique_n))
    node_counts = [n for n in range(1, largest_n + 1, step_size)]
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
    save_and_display_benchmark(node_counts, runtimes, graph_type)


def load_benchmark(graph_type):
    with open(f"benchmarks/{graph_type}.json", "r") as file:
        results = json.load(file)
        del results["quadratic_fit"]
    node_counts = [int(n) for n in results.keys()]
    runtimes = [float(v) for v in results.values()]
    save_and_display_benchmark(node_counts, runtimes, graph_type)


#G = fetch_l2rpn_graph("l2rpn_case14_sandbox")
# solve(grid_from_graph(nx.complete_graph(5)), verbosity=3)
# solve(grid_from_graph(nx.cycle_graph(25)), verbosity=0)
# solve(grid_from_graph(nx.circular_ladder_graph(10)), verbosity=3)
# solve(big_funky_instance(), verbosity=2)
# solve(realistic_instance(kV=500))
# solve(trivial_instance2(), verbosity=3)
benchmark(graph_type="cycle")
# load_benchmark(graph_type="circular ladder")
