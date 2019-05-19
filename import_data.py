import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

# Collaboration network of Arxiv High Energy Physics.
# https://snap.stanford.edu/data/ca-HepPh.html
np.random.seed(seed=7)
hepph_data = pd.read_csv("CA-HepPh.txt", sep='\t', header=3)

num_edges_used = hepph_data.shape[0]
#test_part = 0.5

tuple_edge_list = list(hepph_data.itertuples(index=False, name=None))
#n_test_edges = int(num_edges_used * test_part)
#test_indices = np.random.choice(num_edges_used, n_test_edges, replace=False)
#train_indices = np.setdiff1d(np.arange(num_edges_used), test_indices)

#unique_nodes = np.unique([node for edge in tuple_edge_list for node in edge])
#test_edges =  tuple_edge_list[test_indices]

# Create graph.
hepph_graph = nx.Graph()
#hepph_graph.add_nodes_from(unique_nodes)
hepph_graph.add_edges_from(tuple_edge_list)

node_degrees = [len(hepph_graph[node]) for node in hepph_graph.nodes()] 
avg_deg = np.average(node_degrees)

# take 300 nodes with degree > 40
used_nodes = np.random.choice([node for node in hepph_graph.nodes() if len(hepph_graph[node]) > 40],
                               300, replace=False)

# randomly remove 50% of their edges
test_edges = []
for node in used_nodes:
    neighbors = list(hepph_graph[node].keys())
    test_neighbors = np.random.choice(neighbors, int(len(neighbors)/2), replace=False)
    test_edges.extend([(node, neighbor) for neighbor in test_neighbors])
    
pickle.dump(test_edges, open(f"testedges_{int(time.time())}.p", "wb"))
# test_edges = pickle.load(open("save.p", "rb"))

print("test edges", len(test_edges) / len(tuple_edge_list)) 
#train_edges = [edge for edge in tuple_edge_list if edge not in test_edges]
train_edges = list(set(tuple_edge_list) - set(test_edges)) 

pickle.dump(test_edges, open(f"testedges_{int(time.time())}.p", "wb"))

hepph_graph = nx.Graph()
#hepph_graph.add_nodes_from(unique_nodes)
hepph_graph.add_edges_from(train_edges)

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
  
#test_part = 0.3
#n_test_edges = int(num_edges_used * test_part)

#test_indices = np.random.choice(num_edges_used, n_test_edges, replace=False)
#train_indices = np.setdiff1d(np.arange(num_edges_used), test_indices)

#train_edges = tuple_edge_list[train_indices]   
#test_edges =  tuple_edge_list[test_indices]