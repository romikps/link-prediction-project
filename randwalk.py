import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def draw_graph(G):
    nx.draw(G, with_labels=True)
    #plt.savefig(f"graph_{int(time.time())}.png")
    plt.show()

def get_transition_mat(adj_mat):
    degrees = np.sum(adj_mat, axis=1)
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
    if np.sum(normalized) == 0:
        restart_node = np.random.randint(r)
        normalized[restart_node] = 1
    return normalized

#n_nodes = 5        
#G = nx.gnp_random_graph(n_nodes, 0.2)
#G = nx.Graph()
#G.add_nodes_from(nodes)
#G.add_edges_from(train_edges)
#draw_graph(G)

#A = nx.adj_matrix(G)
#A = A.todense()
#A = np.array(A, dtype = np.float64)

# the degree matrix D
# D = np.diag(np.sum(A, axis=0))
# the transition matrix T
# T = np.dot(np.linalg.inv(D), A)
#T = get_transition_mat(A)

# Markovian random walk 
# the graph can be considered as a finite-state Markov chain
#walkLength = 10
# the state vector - the i-th component indicates the probability of being at node i
# define the starting node
#p = np.zeros(n_nodes)
#p[0] = 1
#p = p.reshape(-1,1)
#visited = list()
#for k in range(walkLength):
    # evaluate the next state vector
#    p = np.dot(T, p)
    # choose the node with the highest probability as the next visited node
    # visited.append(np.argmax(p))
#    try:
#        next_node = np.random.choice(n_nodes, p=get_probs(p))
#    except ValueError as err:
#        print(err)
#        print("p =", get_probs(p))
#        
#    visited.append(next_node)
#   
#accumulated = {}
#for val in visited:
#    accumulated[val] = accumulated[val] + 1 if val in accumulated else 1

# normalize
#for key in accumulated:
#    accumulated[key] /= walkLength
#
#rank_sorted = sorted(accumulated.items(), reverse=True, key=lambda elem: elem[1])
# ranked_nodes = [(nodes[i], rank) for (i, rank) in rank_sorted]


def convergence(p1, p2, epsilon=1e-3):
    diff = np.amax(np.abs(p1 - p2))
    print("p diff =", diff)
    return diff <= epsilon


def page_rank(Q):
    """
    Computes the stationary probabilities vector and its derivative
    :param Q: transition probability matrix
    :return: The vector of stationary distribution probabilities of Q
    Could be more memory efficient!!!
    """
    V = Q.shape[0]
#    p = np.array([np.repeat(1 / V, V)], dtype=np.float64)
#    t1 = 1
#    converged = False
#    while not converged:
#        p_new = np.empty([V])
#        for i in range(V):
#            p_new[i] = np.dot(p[t1 - 1], Q[:, i])
#        p = np.append(p, [p_new], axis=0)
#        converged = convergence(p_new, p[t1 - 1])
#        t1 = t1 + 1
#    return p[-1]
    p = np.repeat(1 / V, V).reshape(1, -1)
    converged = False
    while not converged:
        p_new = np.dot(p, Q)
        converged = convergence(p_new, p)
        p = p_new
    return p_new.reshape(-1)


def generate_transition_probability_matrix(edge_strength_matrix, alpha, start_node):
    """
    :param edge_strength_matrix: NxN matrix containing edge strengths between nodes
    :param alpha: restart parameter
    :param start_node: random walk start node index
    :return: NxN matrix of transition probabilities between nodes
    """
    N = edge_strength_matrix.shape[0]
    strength_row_sums = edge_strength_matrix.sum(axis=1)  # np.apply_along_axis(math.fsum, axis = 1, arr=edge_strength_matrix)
    transition_probability_matrix = np.empty_like(edge_strength_matrix, dtype=np.float64)
    for i in range(N):
        if (strength_row_sums[i] != 0):
            transition_probability_matrix[i] = edge_strength_matrix[i] / strength_row_sums[i]
        else:
            transition_probability_matrix[i] = np.zeros(N)
    transition_probability_matrix = transition_probability_matrix * (1 - alpha)
    transition_probability_matrix[:, start_node] = transition_probability_matrix[:, start_node] + alpha
    
    return transition_probability_matrix


def calculate_edge_strength(feature_vector, w):
    """
    Defines the edge strength function
    :param feature_vector: vector of features
    :param w: vector of feature weights
    :return: The edge strength of a given edge with respect to its feature vector.
    """
    #return np.exp(np.dot(feature_vector, w))
    return np.dot(feature_vector, w) + 1


def generate_edge_strength_matrix(w, feature_vector_matrix, adjacency_matrix):
    """
    Generates edge strength
    :param w: vector of feature weights
    :param feature_vector_matrix: NxNxW matrix of edge_feature_vectors between nodes
    :param adjacency_matrix: NxN adjacency matrix of the network
    :return: A NxN sized matrix containing the edge strengths of node pairs
    """
    edge_strength_matrix = np.empty([feature_vector_matrix.shape[0], feature_vector_matrix.shape[1]], dtype=np.float64)
    for i in range(feature_vector_matrix.shape[0]):
        for j in range(feature_vector_matrix.shape[1]):
            if adjacency_matrix[i][j] > 0:
                edge_strength_matrix[i][j] = calculate_edge_strength(feature_vector_matrix[i][j], w) * adjacency_matrix[i][j]
            
    return edge_strength_matrix