import grid2op
import networkx as nx
import numpy as np


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


def make_grid(capacity, resistance, supplies, demands, prodcpus, graph, step_durations=(1,)):
    G = nx.Graph(sources={}, sinks={}, capacity=capacity, supplies=supplies,
                 step_durations=step_durations, k=len(step_durations))
    for node, attr in graph.nodes(data=True):
        G.graph["sinks"][node] = demands[node]
        node_w_data = (node, {"d": demands[node]})
        if node in supplies.keys():
            G.graph["sources"][node] = {"c": prodcpus[node]}
            node_w_data[1]["c"] = prodcpus[node]
        G.add_nodes_from([node_w_data])
    for edge in graph.edges:
        arc_resistance = resistance[edge] if type(resistance) is dict else resistance
        arc_capacity = capacity[edge] if type(capacity) is dict else capacity
        G.add_edge(edge[0], edge[1], r=arc_resistance, u=arc_capacity)
    return G


def ieee(capacity, resistance, supplies, demands, prodcpus, step_durations, env_name="l2rpn_case14_sandbox"):
    env = grid2op.make(env_name)
    obs = env.reset()
    rawG = obs.get_energy_graph()
    return make_grid(capacity, resistance, supplies, demands, prodcpus, rawG)


def attach_derived_attr(G):
    # TODO: Very unsure about this for cr and for c with anything but k=T=1. Look into plot_flow function
    #  (and sensibility of heatmaps on nodes in the first place now that we have multiple time-steps)
    G.graph["supplies"] = {name: sum(length for length, _ in (attr["c"] if "c" in attr else attr["cr"]))
                           for name, attr in G.graph["sources"].items()}
    G.graph["capacity"] = {(u, v): attr["u"] for u, v, attr in G.edges(data=True)}
    G.graph["k"] = len(G.graph["step_durations"])


def to_graph(sources, sinks, step_durations, edges, directed=False):
    G = (nx.DiGraph if directed else nx.Graph)(edges, sources=sources, sinks=sinks, step_durations=step_durations)
    for src, attr in sources.items():
        G.add_nodes_from([(src, attr)])
    for sink, d in sinks.items():
        G.add_node(sink, d=d)
    attach_derived_attr(G)
    return G


def trivial_instance(directed=False):
    # Exact feasibility requires cumulative supply of: (5-sqrt(15)) + (5-sqrt(5) + 2*(5-sqrt(15)) = 20-sqrt(5)-3sqrt(15)
    # = 6.1449819838779596480530301319215249320606165245137017959663814562, so this instance should be barely infeasible
    return to_graph(sources={"s": {"c": [(4, 1), (2.1449009999999999, 2)]}},
                    sinks={"d": (1, 2, 1)},
                    step_durations=(1, 1, 2),
                    edges=[("s", "d", {"r": 1e-1, "u": 3})],
                    directed=directed)


def alg3_instance():
    return to_graph(sources={"a": {"cr": [(1, 1), (1, 1)]}, "b": {"c": [(1, 1)]}},
                    sinks={"c": (1,)},
                    step_durations=(1,),
                    edges=[("a", "c", {"r": 1e-1, "u": 2}), ("b", "c", {"r": 1e-1, "u": 2})])


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
    nx.relabel_nodes(graph, {node: str(node) for node in graph.nodes}, copy=False)
    capacity = {arc: np.random.rand() * 25 for arc in graph.edges}
    resistance = {arc: 10 ** -(2 + 3 * np.random.rand()) for arc in graph.edges}
    demands = {node: [np.random.rand() * 10 for _ in range(4)] for node in graph.nodes}
    supplies = {node: 70 + np.random.rand() * 70 for node in graph.nodes}
    prodcpus = {node: [(supplies[node]*2/3, np.random.rand()), (supplies[node]/3, 1)] for node in graph.nodes}
    return make_grid(capacity, resistance, supplies, demands, prodcpus, graph, step_durations=[1, 2, 3, 1])
