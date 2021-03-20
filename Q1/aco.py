import networkx as nx
from networkx.classes.function import subgraph
from networkx.convert_matrix import to_numpy_array
import numpy as np
import random

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


# TODO: Check if ok, else do M[i+1][j+1]
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

def find_tau(partial_sol, M, v, c):
    tau2 = 0
    sum = 0
    if partial_sol[c] == []: tau2 = 1
    else:
        for x in partial_sol[c]:
            sum += M[int(x)-1][int(v)-1]
        tau2 = sum / len(partial_sol[c])
    return tau2

def find_uncolored(graph, partial_sol):
    set_nodes = set(graph.nodes())
    arr = []
    for color in partial_sol:
        arr += list(partial_sol[color])
    set_colored = set(arr)
    return set_nodes - set_colored

def p_item(partial_sol: dict, M: np.array, v, graph: nx.Graph, c_min: dict, d_sat: dict):
    # find tau_2 for v 
    tau2 = find_tau(partial_sol, M, v, c_min[v]) + 1
    #print(tau2) 
    # find eta for v 
    eta = d_sat[v] + 1
    #print(eta)
    # find numerator 
    num = tau2 * eta 
    # find denominator
    uncolored = find_uncolored(graph, partial_sol)
    denom = 0
    for vrtx in uncolored:
        t2 = find_tau(partial_sol, M, vrtx, c_min[vrtx])
        m_eta = d_sat[vrtx] + 1
        denom = denom + (t2 * m_eta)
    # do some calculation shit 
    p_it = num / denom
    return p_it

def fps(partial_sol, graph, M, A, c_min, d_sat):
    probs = []
    prob = 0
    for v in A:
        prob = p_item(partial_sol, M, v, graph, c_min, d_sat)
        probs.append(prob)
    print(sum(probs))
    new_v = np.random.choice(A, 1, p=probs)
    return new_v
    # for i in range(len(probs)):
        # if r < probs[i]:
        #     continue
        

def ant_dsatur(graph: nx.Graph, M):
    vertices = list(graph.nodes())
    c_min = {}
    d_sat = {}
    for vertex in vertices:
        c_min[vertex] = 0
        d_sat[vertex] = 0
    V = {i: [] for i in range(len(vertices))}
    A = vertices
    sub_A = subgraph(graph, A) 
    sub_A = nx.Graph(sub_A)   # uncolored vertices 
    # print(list(sub_A.nodes()))
    degrees = list(sub_A.degree(A))
    degrees.sort(key=lambda x:x[1], reverse=True)
    v = degrees[0][0]
    V[0].append(v)
    q = 1       # number of colors used 
    for i in range(1, len(vertices)):
        for v_prime in list(sub_A[v]):
            # print("what")
            used_colors = set()
            for item in V.items():
                if v_prime == item[1]: used_colors.add(item[0])
            c_min[v_prime] = find_first_color(used_colors) 
            if v_prime in V.values(): d_sat[v_prime] += 1
            # print(v_prime)
        A.remove(v)
        # print(c_min)
        sub_A.remove_node(v)   
        partial_sol = {key: V.get(key) for key in range(q)}
        v = fps(partial_sol, graph, M, A, c_min, d_sat)
        v = v[0]
        # v = random.choice(A)
        # c = c_min[v]
        # V[c].append(v)

        # apply selection strategy to find next v 
        # update dictionaries and shit
         

# TODO: check matrix indexing ka scene 
def ant_col(graph: nx.Graph, n_cycles: int, n_ants: int):
    n = graph.number_of_nodes()
    M = to_numpy_array(graph)
    M = init_M(M)
    f_opt = float('inf')
    for i in range(n_cycles):
        change_M = np.zeros((n,n))
        for j in range(n_ants):
            ant_dsatur(graph, M)
            pass


dic = {0:[5,3], 1:[6,3]}
x = [1,5,3,8,9]
pr = [0.05,0.05,0.05, 0.05, 0.8]
# print(np.random.choice(x, 1, p=pr))
# vertices = set([1,2,3,4,5])
# colored = set([1,2,3])
# print(vertices - colored)

graph = create_graph('data/gcol1.txt')
ant_col(graph, 3, 5)

# print(list(graph['7']))
