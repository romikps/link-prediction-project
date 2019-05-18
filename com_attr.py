from louvain_community_detection import best_partition, dendo
from import_data import hepph_graph
import community
from minimization_problem import goal_function_single_node, goal_function_derivative_single_node
from scipy.optimize import minimize
    
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
    layer_transitions = args[7]
'''


    