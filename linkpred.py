import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from import_data import train_edges, test_edges, unique_nodes as nodes
# from randwalk import draw_graph

def get_train_test(edge_list, test_part=0.3):
    edge_list = np.array(edge_list)
    n_edges = edge_list.shape[0]
    n_test_edges = int(n_edges * test_part)
    
    test_indices = np.random.choice(n_edges, n_test_edges, replace=False)
    train_indices = np.setdiff1d(np.arange(n_edges), test_indices)
    
    return edge_list[train_indices], edge_list[test_indices]   

def get_complement_edges(graph):
    return np.array([e for e in nx.complement(graph).edges])   

def sort_by_coef(preds):
    return np.array(sorted(preds, reverse=True, key=lambda elem: elem[2]))

def get_over_threshold(preds, threshold=0):
    return preds[preds[:, 2] > threshold]
    

n_nodes = 15     
#G = erdos_renyi_graph(n_nodes, 0.25)
# small world graph
G = nx.watts_strogatz_graph(n_nodes, k=4, p=0.1, seed=7)
nx.draw_circular(G, with_labels=True)
plt.show()

train_edges, test_edges = get_train_test(G.edges())
G_train = nx.Graph()
G_train.add_nodes_from(G.nodes)
G_train.add_edges_from(train_edges)
nx.draw_circular(G_train, with_labels=True)
plt.show()
# nx.draw_spectral(G_train, with_labels=True)

complement_edges = get_complement_edges(G_train)

# jaccard_coefficient
preds_jac = nx.jaccard_coefficient(G_train, complement_edges)
preds_jac = np.array([(u, v, p) for (u, v, p) in preds_jac])
preds_jac_sorted = sort_by_coef(preds_jac)

# adamic_adar_index
preds_adam = nx.adamic_adar_index(G_train, complement_edges)
preds_adam = np.array([pred for pred in preds_adam])
preds_adam_sorted = sort_by_coef(preds_adam)

# preferential_attachment
preds_pref = nx.preferential_attachment(G_train, complement_edges)
preds_pref = np.array([pred for pred in preds_pref])
preds_pref_sorted = sort_by_coef(preds_pref)

# within_inter_cluster
# nodes attribute name containing the community information
# G[u][community] identifies which community u belongs to. 
# Each node belongs to at most one community
# cn_soundarajan_hopcroft(G[, ebunch, community])