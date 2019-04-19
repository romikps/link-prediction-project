import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

# Collaboration network of Arxiv High Energy Physics.
# https://snap.stanford.edu/data/ca-HepPh.html
hepph_data = pd.read_csv("CA-HepPh.txt", sep='\t', header=3)

hepph_graph = nx.Graph()
unique_from_nodes = np.unique(hepph_data['FromNodeId'])
unique_to_nodes = np.unique(hepph_data['ToNodeId'])
if np.array_equal(unique_from_nodes, unique_to_nodes):
    print("From and to nodes are the same")

hepph_graph.add_nodes_from(unique_from_nodes)

tuple_list = list(hepph_data.itertuples(index=False, name=None))
hepph_graph.add_edges_from(tuple_list)

nx.draw(hepph_graph)
plt.show()
plt.savefig(f"graph_{int(time.time())}.png")

# Other datasets
# https://snap.stanford.edu/data/ca-AstroPh.html
# https://snap.stanford.edu/data/ca-CondMat.html

example_g = nx.Graph()
for edge in tuple_list[:50]:
    from_n, to_n = edge
    example_g.add_node(from_n)
    example_g.add_node(to_n)
    example_g.add_edge(from_n, to_n)
    
nx.draw(example_g)
plt.show()    
    