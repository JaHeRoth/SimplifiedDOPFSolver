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
from instances import grid_from_graph
from solving import solve

#G = fetch_l2rpn_graph("l2rpn_case14_sandbox")
# solve(grid_from_graph(nx.complete_graph(5)), verbosity=3)
# nx.draw_networkx(alg2_instance())
# plt.savefig(f"plots/alg2_instance_{time.time()}.pdf", bbox_inches='tight')
# plt.show()
# plot_multigraph(algorithm2(alg2_instance(), T=1), f"plots/alg2_on_alg2_instance_{time.time()}.pdf")
# nx.draw_networkx(alg3_instance())
# plt.savefig(f"plots/alg3_instance_{time.time()}.pdf", bbox_inches='tight')
# plt.show()
# plot_multigraph(algorithm3(alg3_instance()), f"plots/alg3_on_alg3_instance_{time.time()}.pdf")
# solve(grid_from_graph(nx.wheel_graph(200)), verbosity=1)
# solve(big_funky_instance(), verbosity=1)
# solve(realistic_instance(kV=500))
# solve(trivial_instance(), verbosity=3)
benchmark(graph_type="cycle", largest_n=30, dirname="output")
# load_benchmark(graph_type="circular ladder")
