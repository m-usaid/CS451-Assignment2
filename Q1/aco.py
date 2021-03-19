import networkx as nx
from networkx.classes.function import subgraph
from networkx.convert_matrix import to_numpy_array
import numpy as np

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


def init_M(M: np.array):
    for i in range(len(M)):
        for j in range(len(M[i])):
            if int(M[i][j]) == 0:
                M[i][j] = 1
            else: M[i][j] = 0
    return M

def find_first_color(color_list: list):
    color_set = set(color_list)
    count = 0
    while True:
        if count not in color_set:
            return count
        count += 1    

def ant_dsatur(graph: nx.Graph):
    vertices = list(graph.nodes())
    c_min = {}
    d_sat = {}
    for vertex in vertices:
        c_min[vertex] = 0
        d_sat[vertex] = 0
    V = {i: [] for i in range(len(vertices))}
    A = vertices
    sub_A = subgraph(graph, A)        # uncolored vertices 
    degrees = list(sub_A.degrees(A))
    degrees.sort(key=lambda x:x[1], reverse=True)
    v = degrees[0]
    V[0].append(v)
    q = 1       # number of colors used 
    for i in range(1, len(vertices)):
        for v_prime in sub_A.neighbors(v):
            used_colors = set()
            for item in V.items():
                if v_prime == item[1]: used_colors.add(item[0])
            c_min[v_prime] = find_first_color(used_colors) 
            if v_prime in V.values(): d_sat[v_prime] += 1
        A.remove(v)
        sub_A.remove_node(v)   
        partial_sol = {key: V.get(key) for key in range(q)}
        # apply selection strategy


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




dic = {0:5, 1:6, 2:7, 3:6, 4:9, 5:11}
mdic = {key: dic.get(key) for key in range(3)}
print(mdic)



graph = create_graph('data/gcol1.txt')

