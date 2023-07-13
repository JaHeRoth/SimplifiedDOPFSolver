import time

import networkx as nx
from matplotlib import pyplot as plt


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
