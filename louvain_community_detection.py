from import_data import hepph_graph

import community
import networkx as nx
import matplotlib.pyplot as plt

# G = nx.erdos_renyi_graph(30, 0.05)
# G = nx.karate_club_graph()
G = hepph_graph

# first compute the best partition
partition = community.best_partition(G)
print(partition)

# drawing
colormap = plt.get_cmap('Reds')
size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0.
for com in set(partition.values()):
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                  if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size=30,
                           node_color=str(count / size))

    # nx.draw_networkx_nodes(G, pos, list_nodes, node_size=30,
    #                       node_color=colormap(count / size))


nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()
