# SimplifiedDOPFSolver

This repository provides an implementation of the main algorithm of my master's thesis
_Solving min-cost concave generalized dynamic flows and approximating dynamic optimal power flows_.
It hence provides an efficient algorithm for solving the min-cost dynamic V-V flow problem with
quadratic losses. For more details and a deeper explanation of this problem and algorithm, please see
the thesis.

This repository also provides some example instances to run the algorithm on, helper functions for
creating more such instances, and a benchmarking suite. The algorithm implementation is what is
described in the _Implementation_ chapter of the thesis, while the benchmarking suite is what was used
to obtain all the findings of that chapter.

This repository is exclusively written in Python. The graph algorithms are implemented in
NetworkX, while the convex QCQP is formulated and solved with the gurobipy interface of Gurobi.

# Getting started

1. Download and install the [Gurobi optimizer](https://www.gurobi.com/solutions/gurobi-optimizer/).
2. Download and install [Python](https://www.python.org/downloads/).
3. Clone this repository, install its dependencies in a virtual environment (using _pipenv_), and
run `main.py` in that virtual environment:
```
git clone https://github.com/JaHeRoth/SimplifiedDOPFSolver.git
cd SimplifiedDOPFSolver
pip install pipenv
pipenv install
pipenv run python main.py
```

# Structure

`algorithm/` contains the two files `solving.py` and `outputting.py`. The former
is the actual implementation of the algorithm, while the latter contains support functions for plotting,
printing and persisting. `instances.py` contains some example instances to run the algorithm on, as well
as some helper functions for generating more such instances. `benchmarking.py` contains the benchmarking
suite, both in terms of measuring execution time and in terms of plotting, storing and analyzing the
results. `main.py` is the main entry point of the program, currently containing some code meant to
demonstrate the main functionalities of this repository.

# License

As this software was developed using Gurobi with an Academic License, it can only be used for
research and educational purposes. Within these bounds, feel free to use this software however you wish,
with the one condition that any derived works must credit this repository and its author.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.