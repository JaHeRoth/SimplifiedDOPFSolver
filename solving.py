import os
import re
from datetime import datetime

import networkx as nx
from gurobipy import Model, GRB, quicksum

from outputting import print_flow, plot_flow, plot_graph


def find_optimal_flow(G, verbose=False):
    qcqp = Model(name="QCQP")
    x, y = {}, {}
    # obj = 0
    for a in G.edges:
        x[a] = qcqp.addVar(name=f"x_{a}", lb=0, obj=G.edges[a]["c"])
        y[a] = qcqp.addVar(name=f"y_{a}", lb=0)
        qcqp.addConstr(G.edges[a]["r"] * x[a] ** 2 - G.edges[a]["mu"] * x[a] + y[a] <= 0)
        qcqp.addConstr(x[a] <= G.edges[a]["u"])
        # obj += G.edges[a]["c"] * 0.5 * x[a] + np.random.rand() * 0.001 * G.edges[a]["c"] * x[a]**2
    for v in G.nodes:
        in_flows = [y[a] for a in G.in_edges(v)]
        out_flows = [x[a] for a in G.out_edges(v)]
        if v in G.graph["sinks"]:
            qcqp.addConstr(quicksum(in_flows) == G.nodes[v]["d"])
        elif v != "s*":
            qcqp.addConstr(quicksum(in_flows) == quicksum(out_flows))
    qcqp.modelSense = GRB.MINIMIZE
    qcqp.Params.LogToConsole = int(verbose)
    # qcqp.setObjective(obj)
    # qcqp.Params.BarQCPConvTol = 0
    # qcqp.Params.FeasibilityTol = 1e-9
    qcqp.optimize()
    if verbose:
        os.makedirs("output", exist_ok=True)
        qcqp.write("output/qcqp.lp")
    variables = {var.VarName: var.X for var in qcqp.getVars()}
    return qcqp.objVal, variables


def algorithm1(G):
    sources, sinks, arcs, k, step_durations =\
        G.graph["sources"], G.graph["sinks"], G.edges, G.graph["k"], G.graph["step_durations"]
    Gp = nx.DiGraph(sinks=set())
    for i in range(k):
        for u, v, attr in arcs(data=True):
            Gp.add_edge(f"{u}'{i}", f"{v}'{i}", c=0, mu=1, r=attr["r"], u=attr["u"])
        for sink in sinks:
            Gp.add_node(f"{sink}'{i}", d=G.nodes[sink]["d"][i])
            Gp.graph["sinks"].add(f"{sink}'{i}")
    for src in sources:
        for j, (length, cost) in enumerate(G.nodes[src]["c"]):
            Gp.add_edge("s*", f"{src}|{j}", c=cost, mu=1, r=0, u=length)
            for i, step_duration in enumerate(step_durations):
                Gp.add_edge(f"{src}|{j}", f"{src}'{i}", c=cost, mu=1/step_duration, r=0, u=length)
    return Gp


def algorithm3(G):
    # Steps 1 & 4
    src_connected = set(nx.multi_source_dijkstra_path_length(
        G, G.graph["sources"].keys()).keys())
    sink_connected = set(nx.multi_source_dijkstra_path_length(
        G, G.graph["sinks"].keys()).keys())
    relevant = src_connected.intersection(sink_connected)
    Gp = G.subgraph(relevant).to_directed()
    Gp.graph["sources"] = set()
    Gp.graph["sinks"] = set()
    # Step 2
    for src, attr in G.graph["sources"].items():
        if "cr" in attr:
            # Step 2a
            for i, (length, cost) in enumerate(attr["cr"]):
                node = f"{src}s{i}"
                Gp.graph["sources"].add(node)
                Gp.add_node(node, c=[(length, cost)])
                Gp.add_edge(node, src, r=0, u=length)
            del Gp.nodes[src]["cr"]
        else:
            # Step 2b
            node = f"{src}s"
            Gp.graph["sources"].add(node)
            Gp.add_node(node, c=attr["c"])
            csupply = sum(l for l, _ in attr["c"])
            Gp.add_edge(node, src, r=0, u=csupply)
            del Gp.nodes[src]["c"]
    # Step 3
    for sink, d in G.graph["sinks"].items():
        node = f"{sink}d"
        Gp.graph["sinks"].add(node)
        Gp.add_node(node, d=d) # TODO: Ensure this only adds attribute to node if it already exists
        # Setting this arc capacity to the demand of the sink would break strict feasibility, so we make it larger
        Gp.add_edge(sink, node, r=0, u=float("inf"))
        del Gp.nodes[sink]["d"]
    return Gp


def to_original_graph_flow(full_flow, G):
    """First get rid of variables corresponding to arcs not present in the original graph.
    Then, for each pair of antiparallel arcs, subtract the smaller flow from the larger flow and remove
    the variables corresponding to the smaller flow (adds waste to that arc unless resistance is 0)."""
    # Example: In "x_(\"s\'0", \"d\'0\")" we match with "s" and "d", giving ("s", "d"), which is in G.edges
    flow = {var: val for var, val in full_flow.items() if re.match(
        "[xy]_\\(['\"]([^'\"]+).+, ['\"]([^'\"]+).+", var).groups() in G.edges}
    # for arc in G.out_edges("s*", keys=True):
    #     del flow[f"x_{arc}"]
    #     del flow[f"y_{arc}"]
    #     # Without this if-statement we'd try to remove out-arcs of arc[1] once per parallel (arc[0],arc[1]) arc
    #     if arc[2] == 0:
    #         for narc in G.out_edges(arc[1], keys=True):
    #             del flow[f"x_{narc}"]
    #             del flow[f"y_{narc}"]
    # for arc in G.in_edges("d", keys=True):
    #     del flow[f"x_{arc}"]
    #     del flow[f"y_{arc}"]
    #     for parc in G.in_edges(arc[0], keys=True):
    #         del flow[f"x_{parc}"]
    #         del flow[f"y_{parc}"]
    for u, v in G.edges:
        for i in range(G.graph["k"]):
            ui, vi = f"{u}'{i}", f"{v}'{i}"
            arc, aarc = str((ui, vi)), str((vi, ui))
            # Shortcut way to check if forward flow beats backwards flow, assuming large differences
            if flow[f"y_{arc}"] > flow[f"x_{aarc}"]:
                flow[f"y_{arc}"] -= flow[f"x_{aarc}"]
                flow[f"x_{arc}"] -= flow[f"y_{aarc}"]
                del flow[f"x_{aarc}"]
                del flow[f"y_{aarc}"]
    return flow


def solve(G, verbosity=1):
    sstart = datetime.now()
    Gp = algorithm3(G)
    Gpp = algorithm1(Gp)
    if verbosity > 0:
        print(f"Graph modifications algorithm ran in {(datetime.now()-sstart).total_seconds():.2} seconds.")
    if verbosity > 2:
        plot_graph(G, "G")
        plot_graph(Gp, "Gp")
        plot_graph(Gpp, "Gpp")
    start = datetime.now()
    cost, flow = find_optimal_flow(Gpp, verbose=(verbosity > 1))
    if verbosity > 0:
        print(f"Defined and solved Gurobi program in {(datetime.now()-start).total_seconds():.2} seconds.")
    merged_flow = to_original_graph_flow(flow, G)
    total_runtime = (datetime.now()-sstart).total_seconds()
    if verbosity > 0:
        print(f"Obtained merged flow. Total runtime: {total_runtime:.3f} seconds.")
        if verbosity > 1:
            print_flow(cost, merged_flow, merged=True)
        plot_flow(merged_flow, G)
    return cost, merged_flow, total_runtime
