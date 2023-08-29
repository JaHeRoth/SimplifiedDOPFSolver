import math
import os
import re
from datetime import datetime

import networkx
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
            qcqp.addConstr(quicksum(in_flows) >= G.nodes[v]["d"])
        elif v != "s*":
            qcqp.addConstr(quicksum(in_flows) >= quicksum(out_flows))
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


def split_sources_and_time_expand(G):
    sources, sinks, arcs, k, step_lengths =\
        G.graph["sources"], G.graph["sinks"], G.edges, G.graph["k"], G.graph["step_lengths"]
    Gp = nx.DiGraph(sinks=set(), name="Gpp")
    for i in range(k):
        for u, v, attr in arcs(data=True):
            Gp.add_edge(f"{u}'{i}", f"{v}'{i}", c=0, mu=1, r=attr["r"], u=attr["u"])
        for sink in sinks:
            Gp.add_node(f"{sink}'{i}", d=G.nodes[sink]["d"][i])
            Gp.graph["sinks"].add(f"{sink}'{i}")
    for src in sources:
        for j, (length, cost) in enumerate(G.nodes[src]["c"]):
            Gp.add_edge("s*", f"{src}|{j}", c=cost, mu=1, r=0, u=length)
            for i, step_length in enumerate(step_lengths):
                Gp.add_edge(f"{src}|{j}", f"{src}'{i}", c=cost, mu=1/step_length, r=0, u=length)
    return Gp


def split_nodes_and_direct_arcs(G: networkx.Graph):
    # Step 1
    Gp: networkx.DiGraph = G.to_directed()
    Gp.graph["name"] = "Gp"
    Gp.graph["sources"] = set()
    Gp.graph["sinks"] = set()
    # Step 2
    for v, attr in G.nodes(data=True):
        if "cr" in attr:
            # Step 2a
            for i, (length, cost) in enumerate(attr["cr"]):
                src = f"{v}s{i}"
                Gp.graph["sources"].add(src)
                Gp.add_node(src, c=[(length, cost)])
                Gp.add_edge(src, v, r=0, u=length)
            del Gp.nodes[v]["cr"]
        elif "c" in attr:
            # Step 2b
            src = f"{v}s"
            Gp.graph["sources"].add(src)
            Gp.add_node(src, c=attr["c"])
            csupply = sum(l for l, _ in attr["c"])
            Gp.add_edge(src, v, r=0, u=csupply)
            del Gp.nodes[v]["c"]
        sink = f"{v}d"
        Gp.graph["sinks"].add(sink)
        Gp.add_node(sink, d=attr["d"])
        # Setting this arc capacity to the demand of the sink would break strict feasibility, so we make it larger
        Gp.add_edge(v, sink, r=0, u=float("inf"))
        del Gp.nodes[v]["d"]
    # Step 3 (ish: here only requiring that node is part of S-D walk, instead of S-D path as in thesis)
    src_connected = nx.multi_source_dijkstra_path_length(Gp, Gp.graph["sources"]).keys()
    sink_connected = nx.multi_source_dijkstra_path_length(Gp.reverse(), Gp.graph["sinks"]).keys()
    return Gp.subgraph(src_connected & sink_connected)


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
    for u, v, attr in G.edges(data=True):
        for i in range(G.graph["k"]):
            ui, vi = f"{u}'{i}", f"{v}'{i}"
            arc, aarc = str((ui, vi)), str((vi, ui))
            if flow[f"y_{arc}"] > flow[f"y_{aarc}"]:
                small, large = aarc, arc
            else:
                small, large = arc, aarc
            flow[f"x_{large}"] -= flow[f"y_{small}"]
            # y = x - rx^2
            flow[f"y_{large}"] = flow[f"x_{large}"] - attr["r"] * flow[f"x_{large}"] ** 2
            del flow[f"x_{small}"]
            del flow[f"y_{small}"]
    return flow


def solve(G, verbosity=1):
    sstart = datetime.now()
    Gp = split_nodes_and_direct_arcs(G)
    Gpp = split_sources_and_time_expand(Gp)
    if verbosity > 0:
        print(f"Graph modifications algorithm ran in {(datetime.now()-sstart).total_seconds():.2} seconds.")
    if verbosity > 2:
        plot_graph(G)
        plot_graph(Gp)
        plot_graph(Gpp)
    start = datetime.now()
    cost, flow = find_optimal_flow(Gpp, verbose=(verbosity > 1))
    if verbosity > 0:
        print(f"Defined and solved Gurobi program in {(datetime.now()-start).total_seconds():.2} seconds.")
    start = datetime.now()
    merged_flow = to_original_graph_flow(flow, G)
    total_runtime = (datetime.now()-sstart).total_seconds()
    if verbosity > 0:
        print(f"Flow merging ran in {(datetime.now()-start).total_seconds():.2} seconds.")
        print(f"Obtained merged flow. Total runtime: {total_runtime:.3f} seconds.")
        if verbosity > 1:
            print_flow(cost, merged_flow, merged=True)
            plot_flow(merged_flow, G)
    return cost, merged_flow, total_runtime
