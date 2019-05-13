from louvain_community_detection import best_partition, dendo
from import_data import hepph_graph
import community
    
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
            
        
    
    