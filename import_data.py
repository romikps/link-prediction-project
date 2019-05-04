import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

# Collaboration network of Arxiv High Energy Physics.
# https://snap.stanford.edu/data/ca-HepPh.html
hepph_data = pd.read_csv("CA-HepPh.txt", sep='\t', header=3)

num_nodes_used = 10 #hepph_data.shape[0]
tuple_edge_list = list(hepph_data.itertuples(index=False, name=None))[:num_nodes_used]
unique_nodes = np.unique([node for edge in tuple_edge_list for node in edge])

# Create graph.
# hepph_graph = nx.Graph()
# hepph_graph.add_nodes_from(unique_nodes)
# hepph_graph.add_edges_from(tuple_edge_list)

# Draw graph.
# --Separately--
#pos = nx.spring_layout(hepph_graph)
#nx.draw_networkx_nodes(hepph_graph, pos, node_size=10, node_color='B')
#nx.draw_networkx_edges(hepph_graph, pos, alpha=0.5)

# --Together--
#nx.draw(hepph_graph)
#plt.savefig(f"graph_{int(time.time())}.png")
#plt.show()

# Other datasets
# https://snap.stanford.edu/data/ca-AstroPh.html
# https://snap.stanford.edu/data/ca-CondMat.html
  
train_part = 0.7
break_point = int(num_nodes_used * train_part)
train_edges = tuple_edge_list[:break_point]   
test_edges =  tuple_edge_list[break_point:]