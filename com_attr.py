from louvain_community_detection import best_partition, dendo
from import_data import hepph_graph
import community
from minimization_problem import goal_function_single_node, goal_function_derivative_single_node, PageRank
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix, roc_curve, auc
import networkx as nx
import transition_probabilities as tp
import matplotlib.pyplot as plt
import pickle
import time
    
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
            
        
'''
    feature_matrix = args[0]
    alpha = args[1]
    start_node = args[2]
    pl = args[3]
    pd = args[4]
    loss_func = args[5]
    adjacency_matrix = args[6]
'''
feature_num = level + 1
nodes_n = len(hepph_graph)

adjacency_matrix = np.zeros((nodes_n, nodes_n))
feature_matrix = np.zeros((nodes_n, nodes_n, feature_num), dtype=np.float64)

node_to_index = {}
for i, node in enumerate(hepph_graph.nodes):
    node_to_index[node] = i

for from_node, to_node, features in hepph_graph.edges.data():
    i = node_to_index[from_node]
    j = node_to_index[to_node]
    adjacency_matrix[i, j] = 1
    feature_matrix[i, j] = np.array(list(features.values()))
    
alpha = 0.05
s = node_to_index[17010]
loss_func = "wmw"

index_to_node = [n for n in hepph_graph.nodes]

#all_nodes = np.array([n for n in hepph_graph.nodes])
#d_nodes = np.array(list(hepph_graph[s].keys()))
#l_nodes = np.setdiff1d(all_nodes, d_nodes)

all_nodes = np.array([node_to_index[n] for n in hepph_graph.nodes])
d_nodes = np.array([node_to_index[node] for node in hepph_graph[index_to_node[s]].keys()])
l_nodes = np.setdiff1d(all_nodes, d_nodes)

w_init = np.random.rand(feature_num)
goal_func = goal_function_single_node(w_init, feature_matrix, alpha, s, l_nodes, d_nodes, loss_func, adjacency_matrix)
goal_func_d = goal_function_derivative_single_node(w_init, feature_matrix, alpha, s, l_nodes, d_nodes, loss_func, adjacency_matrix)


opt_res = minimize(goal_function_single_node, w_init, 
                   (feature_matrix, alpha, s, l_nodes, d_nodes, loss_func, adjacency_matrix),
                   "L-BFGS-B",
                   jac=goal_function_derivative_single_node,
                   options={'disp': True})

opt_w = opt_res.x
pickle.dump(opt_w, open(f"w_{int(time.time())}.p", "wb"))
# opt_w = pickle.load(open( "save.p", "rb" ))

edge_strengths, edge_strength_derivatives = tp.generate_edge_strength_matrices(opt_w, feature_matrix, adjacency_matrix)
Q, dQ = tp.generate_transition_probability_matrices(edge_strengths, edge_strength_derivatives, alpha, s)
p, dp = PageRank(Q, dQ)  


p_sorted = sorted([(i, prob) for i, prob in enumerate(p)], reverse=True, key=lambda elem: elem[1])

true = np.zeros(nodes_n)
true[d_nodes] = 1

roc = roc_curve(true, p, pos_label=1)
    
def plt_roc(true, pred):
    fpr, tpr, thresholds = roc_curve(true, pred, pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

plt_roc(true, p)

pred = np.array([1 if prob > 3.28932e-14 else 0 for prob in p])
conf_matrix = confusion_matrix(true, pred)

