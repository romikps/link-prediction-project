import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

# Collaboration network of Arxiv High Energy Physics.
# https://snap.stanford.edu/data/ca-HepPh.html
hepph_data = pd.read_csv("CA-HepPh.txt", sep='\t', header=3)

num_nodes_used = 30
tuple_edge_list = list(hepph_data.itertuples(index=False, name=None))[:num_nodes_used]
unique_nodes = np.unique([node for edge in tuple_edge_list for node in edge])

# Create graph.
hepph_graph = nx.Graph()
hepph_graph.add_nodes_from(unique_nodes)
hepph_graph.add_edges_from(tuple_edge_list)

# Draw graph.
nx.draw(hepph_graph)
plt.savefig(f"graph_{int(time.time())}.png")
plt.show()


# nx.draw_graphviz(hepph_graph)
# nx.write_dot(hepph_graph, f"graph_{int(time.time())}.dot")

# Other datasets
# https://snap.stanford.edu/data/ca-AstroPh.html
# https://snap.stanford.edu/data/ca-CondMat.html
  
    