import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

# Collaboration network of Arxiv High Energy Physics.
# https://snap.stanford.edu/data/ca-HepPh.html
hepph_data = pd.read_csv("CA-HepPh.txt", sep='\t', header=3)

num_edges_used = 50 #hepph_data.shape[0]
tuple_edge_list = np.array(list(hepph_data.itertuples(index=False, name=None)))[:num_edges_used]
unique_nodes = np.unique([node for edge in tuple_edge_list for node in edge])

# Create graph.
hepph_graph = nx.Graph()
hepph_graph.add_nodes_from(unique_nodes)
hepph_graph.add_edges_from(tuple_edge_list)

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
  
test_part = 0.3
n_test_edges = int(num_edges_used * test_part)

test_indices = np.random.choice(num_edges_used, n_test_edges, replace=False)
train_indices = np.setdiff1d(np.arange(num_edges_used), test_indices)

train_edges = tuple_edge_list[train_indices]   
test_edges =  tuple_edge_list[test_indices]