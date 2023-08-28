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
        edge_intensity, edge_use = [], {}
        for u, v, attr in G.edges(data=True):
            ui, vi = f"{u}'{i}", f"{v}'{i}"
            edge_use[(u, v)] = (flow.get(f"x_{(ui, vi)}", 0) + flow.get(f"x_{(vi, ui)}", 0)) / attr["u"]
            edge_intensity.append(edge_use[(u, v)])
        node_color = ["#008000" if node in G.graph["supplies"].keys() else "#1f78b4" for node in G.nodes]
        edge_labels = {edge: f"{round(use * 100)}%" for edge, use in edge_use.items()}
        plt.title(f"Edge utilization (teal = 0%, pink = 100%) at time-step {i + 1},\n"
                  f"with source nodes marked in green")
        if len(G.nodes) < 20:
            pos = nx.spring_layout(G)
            nx.draw_networkx(G, pos, node_color=node_color,
                             edge_cmap=plt.get_cmap("cool"), edge_color=edge_intensity, edge_vmin=0, edge_vmax=1)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        else:
            nx.draw_networkx(G, node_size=50, with_labels=False, node_color=node_color,
                             edge_cmap=plt.get_cmap("cool"), edge_color=edge_intensity, edge_vmin=0, edge_vmax=1)
        plt.savefig(f"plots/{time.time()}.pdf", bbox_inches='tight')
        plt.show()


def plot_graph(graph, name):
    # TODO: Instead have name be a graph property
    nx.draw_networkx(graph)
    plt.savefig(f"plots/{name}_{time.time()}.pdf", bbox_inches='tight')
    plt.show()
