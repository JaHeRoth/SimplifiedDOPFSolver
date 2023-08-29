import grid2op
import networkx as nx
import numpy as np


def problem5_instance():
    # Exact feasibility requires cumulative supply of: (5-sqrt(15)) + (5-sqrt(5) + 2*(5-sqrt(15)) = 20-sqrt(5)-3sqrt(15)
    # = 6.1449819838779596480530301319215249320606165245137017959663814562, so this instance should be barely infeasible
    return to_digraph(sources={"s": {"c": [(4, 1), (2.1449009999999999, 2)]}},
                      sinks={"d": (1, 2)},
                      step_durations=(1, 1),
                      arcs=[("s", "d", {"r": 1e-1, "u": 3})])


def to_digraph(sources, sinks, step_durations, arcs):
    return nx.DiGraph(nodes={**sources,**sinks},
                      edges=arcs,
                      sources=sources,
                      sinks=sinks,
                      step_durations=step_durations,
                      capacity={(u, v): attr["u"] for u, v, attr in arcs},
                      k=len(step_durations))


def basic_instance():
    return from_attributes(
        rcpus={"v": [(1, 1), (2, 2)]},
        ccpus={"u": [(1, 1), (2, 2)]},
        demands={"u": (1,), "v": (2,)},
        resistances={("u", "v"): 1e-1},
        capacities={("u", "v"): 2})


def ieee14(kilo_volts=-2):
    # Numbers copied or estimated from "Optimal Power Systems Planning for IEEE-14 Bus Test System Application"
    # and "Transmission Facts" (by AMERICAN ELECTRIC POWER)
    # Note how everything is in MW instead of in current, as these are anyway just a multiple when voltage is constant
    # kV=-1 represents the easiest case, while kV=-2 represents a barely feasible (rather: barely Gurobi-solvable) case
    if kilo_volts == 345:
        capacity = 400
        resistance = 41.9 * 1e-6
    elif kilo_volts == 500:
        capacity = 900
        resistance = 11 * 1e-6
    elif kilo_volts == 765:
        capacity = 2200
        resistance = 3.4 * 1e-6
    elif kilo_volts == -1:
        capacity = 1e6
        resistance = 0
    elif kilo_volts == -2:
        capacity = 175
        resistance = 95.93999999899998698 * 1e-6
    else:
        raise ValueError
    supplies = {"1": 300, "2": 500, "3": 55, "6": 300, "8": 700}
    cost_coeff = {"1": (16.91, 0.00048), "2": (17.26, 0.00031), "3": (0, 0), "6": (16.6, 0.002), "8": (16.5, 0.00211)}
    G = fetch_l2rpn_graph()
    return from_attributes(
        capacities={edge: capacity for edge in G.edges},
        resistances={edge: resistance for edge in G.edges},
        demands={str(node): [1200 / 14] for node in range(14)},
        ccpus={key: [(supply / 2, cost_coeff[key][0]),
                     (supply / 2, cost_coeff[key][0] + cost_coeff[key][1] * (supply / 2) ** 2)]
               for key, supply in supplies.items()}
    )


def ieee118():
    return from_graph(fetch_l2rpn_graph(env_name="l2rpn_wcci_2022"))


def fetch_l2rpn_graph(env_name="l2rpn_case14_sandbox"):
    env = grid2op.make(env_name)
    obs = env.reset()
    l2rpn_graph = obs.get_energy_graph()
    return nx.relabel_nodes(nx.Graph(l2rpn_graph), lambda node: str(node))


def from_graph(G):
    """G: networkx graph with numbers as node names"""
    nx.relabel_nodes(G, lambda node: str(node), copy=False)
    supplies = {node: 70 + np.random.rand() * 70 for node in G.nodes}
    return from_attributes(
        capacities={arc: np.random.rand() * 25 for arc in G.edges},
        resistances={arc: 10 ** -(2 + 3 * np.random.rand()) for arc in G.edges},
        demands={node: [np.random.rand() * 10 for _ in range(4)] for node in G.nodes},
        ccpus={node: [(supplies[node] * 2 / 3, np.random.rand()),
                      (supplies[node] / 3, 1)] for node in G.nodes},
        step_lengths=[1, 2, 3, 1]
    )


def from_attributes(capacities: dict, resistances: dict, demands: dict, rcpus=None, ccpus=None, step_lengths=None):
    k = len(iter(demands.values()).__next__())
    if rcpus is None:
        rcpus = {}
    if ccpus is None:
        ccpus = {}
    if step_lengths is None:
        step_lengths = tuple(np.repeat(1, k))

    for d in demands.values():
        assert len(d) == k, "All demands must be defined on all timesteps"
    assert rcpus.keys().isdisjoint(ccpus.keys()), "A source should base its marginal costs on rate xor cumulative"
    assert resistances.keys() == capacities.keys(), "Resistance and capacity must be provided for every edge"

    no_demand = tuple(np.repeat(0, k))
    G = nx.Graph(capacities=capacities, k=k, step_lengths=step_lengths, name="G")
    for name, costs in rcpus.items():
        G.add_node(name, cr=costs, d=no_demand)
    for name, costs in ccpus.items():
        G.add_node(name, c=costs, d=no_demand)
    for name, demand in demands.items():
        G.add_node(name, d=demand)
    for name, resistance in resistances.items():
        G.add_edge(name[0], name[1], r=resistance, u=capacities[name])
    return G
