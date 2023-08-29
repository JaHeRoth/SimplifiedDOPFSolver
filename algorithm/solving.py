import re
from datetime import datetime

import networkx
import networkx as nx
from gurobipy import Model, GRB, quicksum

from algorithm.outputting import print_flow, plot_flow, plot_graph


def find_optimal_flow(Gpp, verbose=False):
    qcqp = Model(name="QCQP")
    x, y = {}, {}
    for a in Gpp.edges:
        x[a] = qcqp.addVar(name=f"x_{a}", lb=0, obj=Gpp.edges[a]["c"])
        y[a] = qcqp.addVar(name=f"y_{a}", lb=0)
        qcqp.addConstr(Gpp.edges[a]["r"] * x[a] ** 2 - Gpp.edges[a]["mu"] * x[a] + y[a] <= 0)
        qcqp.addConstr(x[a] <= Gpp.edges[a]["u"])
    for v in Gpp.nodes:
        in_flows = [y[a] for a in Gpp.in_edges(v)]
        out_flows = [x[a] for a in Gpp.out_edges(v)]
        if v in Gpp.graph["sinks"]:
            qcqp.addConstr(quicksum(in_flows) >= Gpp.nodes[v]["d"])
        elif v != "s*":
            qcqp.addConstr(quicksum(in_flows) >= quicksum(out_flows))
    qcqp.modelSense = GRB.MINIMIZE
    qcqp.Params.LogToConsole = int(verbose)
    qcqp.optimize()
    variables = {var.VarName: var.X for var in qcqp.getVars()}
    return qcqp.objVal, variables


def split_sources_and_time_expand(Gp):
    sources, sinks, arcs, k, step_lengths =\
        Gp.graph["sources"], Gp.graph["sinks"], Gp.edges, Gp.graph["k"], Gp.graph["step_lengths"]
    Gpp = nx.DiGraph(sinks=set(), name="Gpp")
    for i in range(k):
        for u, v, attr in arcs(data=True):
            Gpp.add_edge(f"{u}'{i}", f"{v}'{i}", c=0, mu=1, r=attr["r"], u=attr["u"])
        for sink in sinks:
            Gpp.add_node(f"{sink}'{i}", d=Gp.nodes[sink]["d"][i])
            Gpp.graph["sinks"].add(f"{sink}'{i}")
    for src in sources:
        for j, (length, cost) in enumerate(Gp.nodes[src]["c"]):
            Gpp.add_edge("s*", f"{src}|{j}", c=cost, mu=1, r=0, u=length)
            for i, step_length in enumerate(step_lengths):
                Gpp.add_edge(f"{src}|{j}", f"{src}'{i}", c=cost, mu=1/step_length, r=0, u=length)
    return Gpp


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


def to_original_graph_flow(full_flow: dict[str, float], G: nx.Graph):
    """First get rid of variables corresponding to arcs not present in the original graph.
    Then, for each pair of antiparallel arcs, subtract the smaller flow from the larger flow and remove
    the variables corresponding to the smaller flow. Update costs correspondingly."""
    # Example: In "x_(\"s\'0", \"d\'0\")" we match with "s" and "d", giving ("s", "d"), which is in G.edges
    flow = {var: val for var, val in full_flow.items() if re.match(
        "[xy]_\\(['\"]([^'\"]+).+, ['\"]([^'\"]+).+", var).groups() in G.edges}
    for u, v, attr in G.edges(data=True):
        for i in range(G.graph["k"]):
            ui, vi = f"{u}'{i}", f"{v}'{i}"
            arc, rarc = str((ui, vi)), str((vi, ui))
            if flow[f"y_{arc}"] > flow[f"y_{rarc}"]:
                small, large = rarc, arc
            else:
                small, large = arc, rarc
            flow[f"x_{large}"] -= flow[f"y_{small}"]
            flow[f"y_{large}"] = flow[f"x_{large}"] - attr["r"] * flow[f"x_{large}"] ** 2
            del flow[f"x_{small}"], flow[f"y_{small}"]
    return flow


def print_with_seconds_elapsed_since(text_before: str, start_time: datetime):
    print(f"{text_before}{(datetime.now()-start_time).total_seconds():.2} seconds.")


def solve(G, verbosity=1):
    sstart = datetime.now()
    Gp = split_nodes_and_direct_arcs(G)
    Gpp = split_sources_and_time_expand(Gp)
    if verbosity > 0:
        print_with_seconds_elapsed_since("Graph modifications algorithm ran in ", sstart)
    if verbosity > 2:
        plot_graph(G)
        plot_graph(Gp)
        plot_graph(Gpp)
    start = datetime.now()
    cost, flow = find_optimal_flow(Gpp, verbose=(verbosity > 1))
    if verbosity > 0:
        print_with_seconds_elapsed_since("Defined and solved Gurobi program in ", start)
    start = datetime.now()
    merged_flow = to_original_graph_flow(flow, G)
    if verbosity > 0:
        print_with_seconds_elapsed_since("Flow merging ran in ", start)
        print_with_seconds_elapsed_since("Obtained merged flow. Total runtime: ", sstart)
    if verbosity > 1:
        print_flow(cost, merged_flow, merged=True)
        plot_flow(merged_flow, G)
    return cost, merged_flow
