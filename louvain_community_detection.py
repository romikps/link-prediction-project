from import_data import hepph_graph

import community
import networkx as nx
import matplotlib.pyplot as plt

# G = nx.erdos_renyi_graph(30, 0.05)
# G = nx.karate_club_graph()
G = hepph_graph

# top modularity communities
partition = community.best_partition(G)
dendogram = community.generate_dendrogram(G)
#print(partition)

# per level
dendo = community.generate_dendrogram(G)
for level in range(len(dendo) - 1):
    print("partition at level", level,
    "is", community.partition_at_level(dendo, level))
    leveldendo = community.partition_at_level(dendo, level) #loops to end

'''
A dendrogram is a tree and each level is a partition of the graph nodes. Level 0 is the first partition, which
contains the smallest communities, and the best is len(dendrogram) - 1.
'''
dict_best = partition
count_best = list(set([i for i in dict_best.values()]))
print('Communities best partition: ', count_best)
print(len(count_best)-1)

level = 0
dict_levels = community.partition_at_level(dendo, level)
count_level = list(set([i for i in dict_levels.values()]))
print('Communities ' + level.__str__() + ' partition: ', count_level)
print(len(count_level)-1)

part = dict_levels # dictionary of nodes and belonging
for node in G.nodes():
    part[node] = node % 2
    ind = community.induced_graph(part, G)
    goal = nx.Graph() # graph where communities in levels are the nodes
    plt.show()
    #goal.add_weighted_edges_from([(0, 1, n*n), (0, 0, n*(n-1)/2), (1, 1, n*(n-1)/2)])
    #nx.is_isomorphic(int, goal)


colormap = plt.get_cmap('Reds')
size = float(len(set(dict_levels.values())))
pos = nx.spring_layout(G)
count = 0.
for com in set(dict_levels.values()):
    count = count + 1.
    list_nodes = [nodes for nodes in dict_levels.keys()
                  if dict_levels[nodes] == com]
    #nx.draw_networkx_nodes(G, pos, list_nodes, node_size=30,
    #                      node_color=str(count / size))

    nx.draw_networkx_nodes(G, pos, list_nodes, node_size=30,
                          node_color=colormap(count / size))


nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()

'''
leveldendo = community.partition_at_level(dendo, 1)
bestpartitionLevelDendo = len(leveldendo) - 1   # level 0
print(bestpartitionLevelDendo)'''


'''
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
'''
