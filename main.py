from matplotlib import pyplot as plt
from gurobipy import Model, quicksum, GRB
import grid2op
import networkx as nx


def plot_multigraph(G):
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
    # qcqp.Params.BarQCPConvTol = 1e-12
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
    Gp = G.subgraph(relevant).to_directed()
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
    Gp = algorithm3(G)
    Gpp = algorithm2(Gp, 1)
    if verbosity > 2:
        nx.draw_networkx(G)
        plt.show()
        plot_multigraph(Gp)
        plot_multigraph(Gpp)
    cost, flow = find_optimal_flow(Gpp, verbose=(verbosity > 1))
    merged_flow = to_original_graph_flow(flow)
    if verbosity > 0:
        print_flow(cost, merged_flow, merged=True)
    return merged_flow


def fetch_l2rpn_graph(name):
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


def ieee14():
    # Numbers copied or estimated from "Optimal Power Systems Planning for IEEE-14 Bus Test System Application"
    # and Transmission Facts (by AMERICAN ELECTRIC POWER)
    env = grid2op.make("l2rpn_case14_sandbox")
    obs = env.reset()
    rawG = obs.get_energy_graph()
    voltage = 500 * 1e3
    MW_capacity = 900
    resistance = 6.3 * 1e-9
    MW_supplies = {1: 300, 2: 500, 3: 55, 6: 300, 8: 700}
    MV_demand = 1200
    supplies = {key: value * 1e6 / voltage for key, value in MW_supplies.items()}
    prodcpus = {1: (16.91, 0.00048), 2: (17.26, 0.00031), 3: (0, 0), 6: (16.6, 0.002), 8: (16.5, 0.00211)}
    cs = {key: [(supplies[key]/2, prodcpus[key][0] * voltage / 1e6),
                (supplies[key]/2, prodcpus[key][0] * voltage / 1e6 + prodcpus[key][1] * (MW_supplies[key]/2)**2)]
          for key in supplies.keys()}
    avg_demand = MV_demand * 1e6 / len(rawG.nodes)

    G = nx.MultiGraph(sources=[], sinks=[])
    for node, attr in rawG.nodes(data=True):
        node_w_data = (node, {"d": avg_demand})
        if node in supplies.keys():
            node_w_data[1]["c"] = cs[node]
            G.graph["sources"].append(node_w_data)
        G.graph["sinks"].append(node_w_data)
        G.add_nodes_from([node_w_data])
    for u, v, attr in rawG.edges(data=True):
        arc_w_data = (u, v, {"r": resistance, "u": MW_capacity * 1e6 / voltage})
        G.add_edges_from([arc_w_data])
    return G


#env_name = "l2rpn_case14_sandbox"
#G = fetch_l2rpn_graph(env_name)
flow = solve(ieee14(), verbosity=3)