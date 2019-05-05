import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# from import_data import train_edges, test_edges, unique_nodes as nodes
# from randwalk import draw_graph

def get_train_test(edge_list, test_part=0.3):
    n_edges = len(edge_list)
    n_test_edges = int(n_edges * test_part)
    
    test_indices = np.random.choice(n_edges, n_test_edges, replace=False)
    train_indices = np.setdiff1d(np.arange(n_edges), test_indices)

    return ([edge_list[edge_index] for edge_index in train_indices],
           [edge_list[edge_index] for edge_index in test_indices]) 

def get_complement_edges(graph):
    return [e for e in nx.complement(graph).edges]  

def sort_by_coef(preds):
    return sorted(preds, reverse=True, key=lambda elem: elem[2])

def get_over_threshold(preds, threshold=0):
    return [pred for pred in preds if pred[2] > threshold]
    
def get_accuracy(pred_edges, true_edges, all_edges):
    tp, fp, tn, fn = 0, 0, 0, 0
    for pred_edge in pred_edges:
        if pred_edge in true_edges:
            tp += 1
        else:
            fp += 1
    fn = len(true_edges) - tp
    tn = len(all_edges) - len(true_edges) - fp
    return  float(tp + tn) / len(all_edges)

def get_edges_over_threshold(preds, threshold=0):
    return [(u, v) for u, v, p in get_over_threshold(preds, threshold)]
    

n_nodes = 1000     
#G = erdos_renyi_graph(n_nodes, 0.25)
# small world graph
G = nx.watts_strogatz_graph(n_nodes, k=4, p=0.1, seed=7)
nx.draw_circular(G, with_labels=True)
plt.show()

train_edges, test_edges = get_train_test([e for e in G.edges])
G_train = nx.Graph()
G_train.add_nodes_from(G.nodes)
G_train.add_edges_from(train_edges)
nx.draw_circular(G_train, with_labels=True)
plt.show()
# nx.draw_spectral(G_train, with_labels=True)

complement_edges = get_complement_edges(G_train)

# jaccard_coefficient
preds_jac = [pred for pred in nx.jaccard_coefficient(G_train, complement_edges)]
preds_jac_sorted = sort_by_coef(preds_jac)

pred_edges_jac = get_edges_over_threshold(preds_jac)
acc_jac = get_accuracy(pred_edges_jac, test_edges, complement_edges)

# adamic_adar_index
preds_adam = [pred for pred in nx.adamic_adar_index(G_train, complement_edges)]
preds_adam_sorted = sort_by_coef(preds_adam)

pred_edges_adam = get_edges_over_threshold(preds_adam)
acc_adam = get_accuracy(pred_edges_adam, test_edges, complement_edges)

# preferential_attachment
preds_pref = [pred for pred in nx.preferential_attachment(G_train, complement_edges)]
preds_pref_sorted = sort_by_coef(preds_pref)

pred_edges_pref = get_edges_over_threshold(preds_pref, 15)
acc_pref = get_accuracy(pred_edges_pref, test_edges, complement_edges)

# within_inter_cluster
# nodes attribute name containing the community information
# G[u][community] identifies which community u belongs to. 
# Each node belongs to at most one community
# cn_soundarajan_hopcroft(G[, ebunch, community])