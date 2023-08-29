import networkx as nx
from matplotlib import pyplot as plt

from algorithm.outputting import print_flow
from benchmarking import run_and_display
from instances import basic_instance, ieee14, ieee118, problem5_instance
from algorithm.solving import solve, split_sources_and_time_expand, find_optimal_flow

solve(ieee14(kV=345), verbosity=2)
solve(ieee118(), verbosity=2)
try:
    solve(ieee14(kV=-2), verbosity=2)
except AttributeError:
    pass
solve(basic_instance(), verbosity=3)
print(problem5_instance())
nx.draw_networkx(problem5_instance()); plt.show()
nx.draw_networkx(split_sources_and_time_expand(problem5_instance())); plt.show()
print_flow(*find_optimal_flow(split_sources_and_time_expand(problem5_instance())), merged=False)
run_and_display(graph_type="circular ladder", max_nodes=101, num_runs=1)
run_and_display(graph_type="circular ladder", max_nodes=22, overwrite=False)
