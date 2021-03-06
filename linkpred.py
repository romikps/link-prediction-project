import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
# from randwalk import draw_graph
from import_data import hepph_graph, test_edges

def get_train_test(edge_list, test_part=0.3):
    n_edges = len(edge_list)
    n_test_edges = int(n_edges * test_part)
    
    test_indices = np.random.choice(n_edges, n_test_edges, replace=False)
    train_indices = np.setdiff1d(np.arange(n_edges), test_indices)

    return ([edge_list[edge_index] for edge_index in train_indices],
           [edge_list[edge_index] for edge_index in test_indices]) 

def get_complement_edges(graph, num):
    # return [e for e in nx.complement(graph).edges]
    comp_edges = []
    nodes = [node for node in graph.nodes]
    from_nodes = random.choices(nodes, k=2*num)
    to_nodes = random.choices(nodes, k=2*num)
    for from_node, to_node in zip(from_nodes, to_nodes):
        if len(comp_edges) > num:
            break
        
        if to_node in graph[from_node]:
            continue
        comp_edges.append((from_node, to_node))
    if len(comp_edges) < num: 
        raise Exception("Insufficient length of complement edges!",
                        f"{len(comp_edges)} != {num}")
    return comp_edges
            
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
 
def get_confusion_matrix(pred_edges, true_edges, all_edges):
    tp, fp, tn, fn = 0, 0, 0, 0
    for pred_edge in pred_edges:
        if pred_edge in true_edges:
            tp += 1
        else:
            fp += 1
    fn = len(true_edges) - tp
    tn = len(all_edges) - len(true_edges) - fp
    return  [[tn, fp], [fn, tp]]

#n_nodes = 1000     
#G = erdos_renyi_graph(n_nodes, 0.25)
# small world graph
#G = nx.watts_strogatz_graph(n_nodes, k=4, p=0.1, seed=7)
#nx.draw_circular(G, with_labels=True)
#plt.show()
    
G = hepph_graph
#train_edges, test_edges = get_train_test([e for e in G.edges])
#G_train = nx.Graph()
#G_train.add_nodes_from(G.nodes)
#G_train.add_edges_from(train_edges)
# nx.draw_circular(G_train, with_labels=True)
# plt.show()
# nx.draw_spectral(G_train, with_labels=True)

#complement_edges = get_complement_edges(G)

#all_test_edges = test_edges + random.sample(complement_edges, k=len(test_edges))
complement_edges = get_complement_edges(hepph_graph, 9 * len(test_edges))
all_test_edges = test_edges + complement_edges
print("all test edges ready!")

#complement_edges = get_complement_edges(G)
#neg_edges = np.random.choice(complement_edges, size=len(test_edges), replace=False)


# jaccard_coefficient
preds_jac = [pred for pred in nx.jaccard_coefficient(hepph_graph, all_test_edges)]
preds_jac_sorted = sort_by_coef(preds_jac)
pred_edges_jac = [jac[:2] for jac in preds_jac_sorted[:len(test_edges)]]
#pred_edges_jac = get_edges_over_threshold(preds_jac)
acc_jac = get_accuracy(pred_edges_jac, test_edges, all_test_edges)
print("jaccard accuracy:", acc_jac)
print("jaccard confusion matrix:", get_confusion_matrix(pred_edges_jac, test_edges, all_test_edges))

# adamic_adar_index
preds_adam = [pred for pred in nx.adamic_adar_index(hepph_graph, all_test_edges)]
preds_adam_sorted = sort_by_coef(preds_adam)

pred_edges_adam = get_edges_over_threshold(preds_adam)
acc_adam = get_accuracy(pred_edges_adam, test_edges, all_test_edges)
print("adamic adar accuracy:", acc_adam)

# preferential_attachment
preds_pref = [pred for pred in nx.preferential_attachment(hepph_graph, all_test_edges)]
preds_pref_sorted = sort_by_coef(preds_pref)

pred_edges_pref = [(u, v) for u, v, p in preds_pref_sorted[:len(test_edges)]]
# pred_edges_pref = get_edges_over_threshold(preds_pref, 1000)
acc_pref = get_accuracy(pred_edges_pref, test_edges, all_test_edges)
print("preferential attachment accuracy:", acc_pref)
print("pref attach confusion matrix:", get_confusion_matrix(pred_edges_pref, 
                                                            test_edges, 
                                                            all_test_edges))

# random
pred_edges_rand = random.sample(all_test_edges, k=len(test_edges))
acc_rand = get_accuracy(pred_edges_rand, test_edges, all_test_edges)
print("random attachment accuracy:", acc_rand)
print("random attachment confusion matrix:", get_confusion_matrix(pred_edges_rand, 
                                                            test_edges, 
                                                            all_test_edges))

# within_inter_cluster
# nodes attribute name containing the community information
# G[u][community] identifies which community u belongs to. 
# Each node belongs to at most one community
# cn_soundarajan_hopcroft(G[, ebunch, community])