import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import powerlaw
from string import ascii_uppercase

n_district = 15
np.random.seed(5789)

powerlaw_distances = np.rint(25*powerlaw.rvs(0.25, size=(((n_district-1)*n_district)//2)) ).astype(np.int64)
edges = np.where(np.random.rand(((n_district-1)*n_district)//2) < 0.25, 1, 0)
wieghted_edges = edges * powerlaw_distances
adj_matrix = np.zeros((n_district, n_district), dtype=np.int64)
w = 0
for i in range(n_district):
    for j in range(i):
        adj_matrix[i, j] = wieghted_edges[w]
        adj_matrix[j, i] = wieghted_edges[w]
        w+=1

# np.save("adjacency_matrix", adj_matrix, allow_pickle=True)

np.random.seed(None)
print(adj_matrix)


G = nx.from_numpy_array(adj_matrix)
mapping = {i: ascii_uppercase[i] for i in range(15)}
G = nx.relabel_nodes(G, mapping)


plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=4567)
nx.draw_networkx_nodes(G, pos, node_size=200)
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
nx.draw_networkx_edges(G, pos, width=2)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
plt.title("Imaginary City Road Network (15 Districts)", fontsize=10)
plt.show()