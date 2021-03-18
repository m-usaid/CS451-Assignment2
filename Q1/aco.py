import networkx as nx
from collections import defaultdict
from networkx.convert_matrix import to_numpy_array
import numpy as np

from networkx.linalg.graphmatrix import adjacency_matrix

def read_data(filename: str) -> list:
    listOfEdges = []
    with open (filename, 'r') as file:
        for l in file:
            tokens = l.split()
            if tokens[0] == 'e':
                edge = (tokens[1], tokens[2])
                listOfEdges.append(edge)
    return listOfEdges

def create_graph(filename: str) -> nx.Graph:
    graph = nx.Graph()
    edgeList = read_data(filename)
    graph.add_edges_from(edgeList)
    return graph    

def ant_dsatur():
    pass 

graph = create_graph('data/gcol1.txt')
# n = graph.number_of_nodes()
# M = to_numpy_array(graph)
# change_M = np.zeros((n, n))

def init_M(M: np.array):
    for i in range(len(M)):
        for j in range(len(M[i])):
            if int(M[i][j]) == 0:
                M[i][j] = 1
            else: M[i][j] = 0
    return M

# M = init_M(M)

def ant_col(graph: nx.Graph, n_cycles: int, n_ants: int):
    n = graph.number_of_nodes()
    M = to_numpy_array(graph)
    M = init_M(M)
    f_opt = float('inf')
    for i in range(n_cycles):
        change_M = np.zeros((n,n))
        for j in range(n_ants):
            # color graph using dsatur 
            pass
