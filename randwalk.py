import networkx as nx
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from import_data import train_edges, test_edges, unique_nodes as nodes

def draw_graph(G):
    nx.draw(G, with_labels=True)
    #plt.savefig(f"graph_{int(time.time())}.png")
    plt.show()

def get_transition_mat(adj_mat):
    degrees = numpy.sum(adj_mat, axis=1)
    trans_mat = adj_mat.copy()
    for i, degree in enumerate(degrees):
        if degree != 0:
            trans_mat[i] /= degree
    return trans_mat

def get_probs(arr):
    r, c = arr.shape
    if r != n_nodes:
        raise ValueError("Number of rows should equal number of nodes")
    normalized = normalize(arr.reshape(1, -1)).reshape(-1)
    if numpy.sum(normalized) == 0:
        restart_node = numpy.random.randint(r)
        normalized[restart_node] = 1
    return normalized

n_nodes = 5        
G = nx.gnp_random_graph(n_nodes, 0.2)
#G = nx.Graph()
#G.add_nodes_from(nodes)
#G.add_edges_from(train_edges)
draw_graph(G)

A = nx.adj_matrix(G)
A = A.todense()
A = numpy.array(A, dtype = numpy.float64)

# the degree matrix D
# D = numpy.diag(numpy.sum(A, axis=0))
# the transition matrix T
# T = numpy.dot(numpy.linalg.inv(D), A)
T = get_transition_mat(A)

# Markovian random walk 
# the graph can be considered as a finite-state Markov chain
walkLength = 10
# the state vector - the i-th component indicates the probability of being at node i
# define the starting node
p = numpy.zeros(n_nodes)
p[0] = 1
p = p.reshape(-1,1)
visited = list()
for k in range(walkLength):
    # evaluate the next state vector
    p = numpy.dot(T, p)
    # choose the node with the highest probability as the next visited node
    # visited.append(numpy.argmax(p))
    try:
        next_node = numpy.random.choice(n_nodes, p=get_probs(p))
    except ValueError as err:
        print(err)
        print("p =", get_probs(p))
        
    visited.append(next_node)
   
accumulated = {}
for val in visited:
    accumulated[val] = accumulated[val] + 1 if val in accumulated else 1

# normalize
for key in accumulated:
    accumulated[key] /= walkLength

rank_sorted = sorted(accumulated.items(), reverse=True, key=lambda elem: elem[1])
# ranked_nodes = [(nodes[i], rank) for (i, rank) in rank_sorted]