import json
import os
import pathlib
from datetime import datetime
from timeit import Timer

import numpy as np
from matplotlib import pyplot as plt
from gurobipy import Model, quicksum, GRB
import time
import grid2op
import networkx as nx
from tqdm import tqdm

from benchmarking import benchmark
from instances import grid_from_graph, trivial_instance, attach_derived_attr, alg3_instance
from solving import solve, algorithm1, find_optimal_flow, algorithm3
from outputting import plot_graph

# G = fetch_l2rpn_graph("l2rpn_case14_sandbox")
# solve(grid_from_graph(nx.complete_graph(5)), verbosity=3)
# nx.draw_networkx(alg2_instance())
# plt.savefig(f"plots/alg2_instance_{time.time()}.pdf", bbox_inches='tight')
# plt.show()
# plot_multigraph(algorithm2(alg2_instance(), T=1), f"plots/alg2_on_alg2_instance_{time.time()}.pdf")
G = alg3_instance()
plot_graph(G, "G")
Gp = algorithm3(G)
plot_graph(Gp, "Gp")
# solve(grid_from_graph(nx.wheel_graph(200)), verbosity=1)
# solve(big_funky_instance(), verbosity=1)
# solve(realistic_instance(kV=500))
# solve(trivial_instance(), verbosity=3)
# benchmark(graph_type="circular ladder", max_nodes=int(0.5e2), repeats=5, num_unique_n=30, dirname="output")
# load_benchmark(graph_type="circular ladder")
# solve(trivial_instance0(), verbosity=3)
# Gp = trivial_instance0(directed=True)
# plot_graph(Gp, "Gp")
# Gpp = algorithm1(Gp)
# plot_graph(Gpp, "Gppp")
# find_optimal_flow(Gpp, verbose=True)
