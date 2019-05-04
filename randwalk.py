import networkx as nx
import numpy
from import_data import train_edges, test_edges, unique_nodes as nodes

# G = nx.gnp_random_graph(5, 0.5)
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(train_edges)

A = nx.adj_matrix(G)
A = A.todense()
A = numpy.array(A, dtype = numpy.float64)

# the degree matrix D
D = numpy.diag(numpy.sum(A, axis=0))
# the transition matrix T
T = numpy.dot(numpy.linalg.inv(D),A)

# Markovian random walk 
# the graph can be considered as a finite-state Markov chain
walkLength = 10
# the state vector - the i-th component indicates the probability of being at node i
# define the starting node
p = numpy.array([1, 0, 0, 0, 0]).reshape(-1,1)
visited = list()
for k in range(walkLength):
    # evaluate the next state vector
    p = numpy.dot(T,p)
    # choose the node with the highest probability as the next visited node
    visited.append(numpy.argmax(p))
   
accumulated = {}
for val in visited:
    accumulated[val] = accumulated[val] + 1 if val in accumulated else 1

# normalize
for key in accumulated:
    accumulated[key] /= walkLength

rank_sorted = sorted(accumulated.items(), reverse=True, key=lambda elem: elem[1])