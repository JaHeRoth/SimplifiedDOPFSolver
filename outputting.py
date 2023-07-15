import time

import networkx as nx
from matplotlib import pyplot as plt


# def plot_multigraph(G, filename):
#     pos = nx.spring_layout(G)
#     names = {name: name for name in G.nodes}
#     nx.draw_networkx_nodes(G, pos, node_color='b', node_size=500, alpha=1)
#     nx.draw_networkx_labels(G, pos, names, font_size=10, font_color='w')
#     ax = plt.gca()
#     for e in G.edges:
#         ax.annotate("",
#                     xy=pos[e[1]], xycoords='data',
#                     xytext=pos[e[0]], textcoords='data',
#                     arrowprops=dict(arrowstyle="->", color="0",
#                                     shrinkA=10, shrinkB=10,
#                                     patchA=None, patchB=None,
#                                     connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * e[2])
#                                                                            ),
#                                     ),
#                     )
#     plt.axis('off')
#     plt.savefig(filename, bbox_inches='tight')
#     plt.show()


def print_flow(cost, flow, merged=False):
    print("---------------------------------------------")
    # Merging antiparallel flows will typically reduce the cost. As `cost` is the cost of the unmerged flow it
    # thus tends to be a slight overapproximation of the cost of the merged flow
    print(f"Obtained solution of value {'at most' if merged else ''} {cost}:")
    if len(flow) < 50:
        for name, value in flow.items():
            print(f"\t{name}: {value}")
    print("---------------------------------------------")


def plot_flow(flow, G):
    for i in range(G.graph["k"]):
        edge_intensity, edge_use, node_intensity = [], {}, []
        for u, v in G.edges:
            edge_capacity = G.graph['capacity'][(u, v)]
            ui, vi = f"{u}'{i}", f"{v}'{i}"
            edge_use[(u, v)] = (flow.get(f"x_{(ui, vi)}", 0) + flow.get(f"x_{(vi, ui)}", 0)) / edge_capacity
            edge_intensity.append(edge_use[(u, v)])
        for node in G.nodes:
            if node in G.graph["supplies"].keys():
                nodei = f"{node}'{i}"
                net_out_flows = []
                for neigh in G.neighbors(node):
                    neighi = f"{neigh}'{i}"
                    net_out_flows.append(flow.get(f"x_{(nodei, neighi)}", 0) - flow.get(f"y_{(neighi, nodei)}", 0))
                node_intensity.append(sum(net_out_flows) / G.graph["supplies"][node])
            else:
                node_intensity.append(1)
        edge_labels = {edge: f"{round(use * 100)}" for edge, use in edge_use.items()}
        if len(G.nodes) < 20:
            plt.title(f"Generator and edge utilization heat maps (teal=0, pink=1)\n"
                      f"and edge utilization in percent, for time-step {i}")
            pos = nx.spring_layout(G)
            nx.draw_networkx(G, pos, cmap=plt.get_cmap("cool"), node_color=node_intensity, vmin=0, vmax=1,
                             edge_cmap=plt.get_cmap("cool"), edge_color=edge_intensity, edge_vmin=0, edge_vmax=1)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        else:
            plt.title(f"Generator and edge utilization heat maps (teal=0, pink=1), for time-step {i}")
            nx.draw_networkx(G, node_size=50, with_labels=False, cmap=plt.get_cmap("cool"), node_color=node_intensity,
                             vmin=0, vmax=1,
                             edge_cmap=plt.get_cmap("cool"), edge_color=edge_intensity, edge_vmin=0, edge_vmax=1)
        plt.savefig(f"plots/{time.time()}.pdf", bbox_inches='tight')
        plt.show()


def plot_graph(graph, name):
    nx.draw_networkx(graph)
    plt.savefig(f"plots/{name}_{time.time()}.pdf", bbox_inches='tight')
    plt.show()
