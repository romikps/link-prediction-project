import community
import pickle
import networkx as nx
import numpy as np
import randwalk
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from import_data import hepph_graph, num_edges_used
from louvain_community_detection import best_partition, dendo

def plt_pr_rec(pr_rec, title="Precision-Recall curve"):
    precision, recall, thresholds = pr_rec
    print('auc=%.3f' % metrics.auc(recall, precision))
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.plot(recall, precision, marker='.')
    plt.show()
    
for level in range(len(dendo)):
    level_partition = community.partition_at_level(dendo, level)
    for node, com in level_partition.items():
        hepph_graph.nodes[node][level] = com  
    
for from_node, to_node in hepph_graph.edges:
    for level in hepph_graph.nodes[from_node]:
        if hepph_graph.nodes[from_node][level] == hepph_graph.nodes[to_node][level]:
            hepph_graph.edges[from_node, to_node][level] = 1
        else:
            hepph_graph.edges[from_node, to_node][level] = 0
            
feature_num = level + 1
nodes_n = len(hepph_graph)

adjacency_matrix = np.zeros((nodes_n, nodes_n))
feature_matrix = np.zeros((nodes_n, nodes_n, feature_num), dtype=np.float64)

index_to_node = [n for n in hepph_graph.nodes]
node_to_index = {}
for i, node in enumerate(hepph_graph.nodes):
    node_to_index[node] = i

for from_node, to_node, features in hepph_graph.edges.data():
    i = node_to_index[from_node]
    j = node_to_index[to_node]
    adjacency_matrix[i, j] = 1
    feature_matrix[i, j] = np.array(list(features.values()))
print("Adjacency and feature matrices ready!")
    
alpha = 0.25
w = np.ones(feature_num)
all_nodes = np.arange(nodes_n)


edge_strengths = randwalk.generate_edge_strength_matrix(w, feature_matrix, adjacency_matrix)
print("Edge strength matrix ready!")

#pickle.dump(edge_strengths, open(f"edgstr_{int(time.time())}.p", "wb"))
# opt_w = pickle.load(open("save.p", "rb"))
stats = []
for i in range(30):
    s = np.random.randint(nodes_n)
    d_nodes = np.array([node_to_index[node] for node in hepph_graph[index_to_node[s]].keys()])
    l_nodes = np.setdiff1d(all_nodes, d_nodes)
    Q = randwalk.generate_transition_probability_matrix(edge_strengths, alpha, s)
    print("Transition probability matrix ready!")
    
    p = randwalk.page_rank(Q)  
    print(f"Random walk converged for node #{i}!")

    #p_sorted = sorted([(i, prob) for i, prob in enumerate(p)], reverse=True, key=lambda elem: elem[1])
    stats.append({'s': s, 'd': d_nodes, 'p': p})
    pickle.dump(stats, open(f"stats_{int(time.time())}.p", "wb"))


conf_sum = np.zeros((2, 2))
for i, stat in enumerate(stats):
    true = np.zeros(nodes_n)
    true[stat['d']] = 1
    
    pr_rec = metrics.precision_recall_curve(true, stat['p'])    
    plt_pr_rec(pr_rec, title=f"Precision-Recall for node #{i}")
    
    pred = np.array([1 if prob > 0 else 0 for prob in p])
    conf_matrix = metrics.confusion_matrix(true, pred)  
    conf_sum += conf_matrix
pickle.dump(conf_sum, open(f"conf_sum_{int(time.time())}.p", "wb"))    
