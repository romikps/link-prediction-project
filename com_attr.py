from louvain_community_detection import best_partition
from import_data import hepph_graph

for node, com in best_partition.items():
    hepph_graph.nodes[node]["com"] = com    