import os
import time
from datetime import datetime

import networkx as nx
from gurobipy import Model, GRB, quicksum
from matplotlib import pyplot as plt

from outputting import plot_multigraph, print_flow, plot_flow


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
    if verbose:
        os.makedirs("ouptut", exist_ok=True)
        qcqp.write("output/qcqp.lp")
    variables = {var.VarName: var.X for var in qcqp.getVars()}
    return qcqp.objVal, variables


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
