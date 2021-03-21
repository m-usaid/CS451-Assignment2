import networkx as nx
import numpy as np 
import random 
import matplotlib.pyplot as plt

class ant:
    def __init__(self, alpha = 1, beta=5) -> None:
        self.graph = None 
        self.coloring = {}
        self.start_vtx = None 
        self.visited_vtx = []
        self.unvisited_vtx = []
        self.alpha = alpha 
        self.beta = beta 
        self.avail_colors = []
        self.assign_colors = {}
        self.n_colors = 0

    def init_sol(self, G:nx.Graph, colorings, startvtx = None):
        self.avail_colors = sorted(colorings.copy()) #.copy()
        keys = list(G.nodes())
        keys = convert_int_lst(keys)
        self.assign_colors = {key: None for key in keys} # Make a dictionary with graph's nodes as keys storing their colour
        if startvtx == None : # if no start vertex given as input
            self.start_vtx = random.choice(keys) # Randomly assign a graph vertex to start vertex
        else: 
            self.start_vtx = startvtx
        # print("yo")
        # self.start_vtx = startvtx
        self.visited_vtx = []
        self.unvisited_vtx = keys.copy()
        # print(self.unvisited_vtx)
        if (len(self.visited_vtx)==0): #Abbas: Won't this always be true?
            # assign color to node 
            # print(self.start_vtx)
            self.assign_color(self.start_vtx, self.avail_colors[0])
        return self 
    
    def assign_color(self, vtx, color):
        self.assign_colors[vtx] = color 
        self.visited_vtx.append(vtx)
        # print(vtx)
        self.unvisited_vtx.remove(vtx)


    def construct_coloring(self):
        n_unvisited = len(self.unvisited_vtx)
        tabu_lst = []
        for i in range(n_unvisited):
            nextvtx = self.next_vertex()
            # print(nextvtx)
            for j in range(n_nodes):
                if adj_mat[nextvtx-1, j-1] == 1 :
                    tabu_lst.append((self.assign_colors[j+1]))
            for k in self.avail_colors:
                if (k not in tabu_lst):
                    self.assign_color(nextvtx, k)
                    break 
        self.n_colors = len(set(self.assign_colors.values()))


    def find_dsat(self, num_nodes: int, startvtx=None):
        if startvtx == None: 
            startvtx = self.start_vtx
        colored_nbrs = []
        for i in range(num_nodes):
            if adj_mat[startvtx-1, i-1] == 1:
                colored_nbrs.append((self.assign_colors[i+1]))
        return len(set(colored_nbrs))

    def M_ij(self, vtx1, vtx2):
        return M[vtx1-1, vtx2-1]

    def next_vertex(self):
        if (len(self.unvisited_vtx) == 0):
            nextvtx = None 
        elif len(self.unvisited_vtx) == 1:
            nextvtx = self.unvisited_vtx[0]
        else:
            maxval = 0
            h_vals = []
            potential_cands = []
            available_cands = []
            for i in self.unvisited_vtx:
                h_vals.append((self.M_ij(self.start_vtx, i)**self.alpha) * (self.find_dsat(i)**self.beta))
                potential_cands.append(i)
            maxval = max(h_vals)
            for j in range(len(potential_cands)):
                if h_vals[j] >= maxval:
                    available_cands.append(potential_cands[j])
            nextvtx = random.choice(available_cands)
        self.start_vtx = nextvtx
        return nextvtx
   
    def trail(self, G: nx.Graph):
        m_trail = np.zeros((n_nodes, n_nodes), float)
        nodes = convert_int_lst(list(G.nodes()))
        for v in nodes:
            for u in nodes:
                if self.assign_colors[v] == self.assign_colors[u]:
                    m_trail[v-1][u-1] = 1 
        return m_trail


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

def draw_graph(G: nx.Graph):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()

def init_M(G: nx.Graph):
    n = G.number_of_nodes()
    M = np.ones((n,n), float)
    for vtx in G:
        for nbr in G.neighbors(vtx):
            M[int(vtx)-1, int(nbr)-1] = 0
    return M

def create_adj_mat(G: nx.Graph):
    n = G.number_of_nodes()
    adj_mat = np.zeros((n,n), int)
    nodes = list(map(int, G.nodes()))
    for node in list(nodes):
        for nbr in G.neighbors(str(node)):
            adj_mat[node-1, int(nbr)-1] = 1
    return adj_mat

def create_ants(G: nx.Graph, n_ants: int, colors: list):
    ants = []
    ants.extend([ant().init_sol(G, colors) for i in range(n_ants)])
    return ants 
    
def initialize_coloring(G: nx.Graph):
    colors = []
    grundy_num = len(nx.degree_histogram(G))
    for color in range(grundy_num):
        colors.append(color)
    return colors 

def evap_trail(G: nx.Graph, M, rho: float):
    nodes = convert_int_lst(list(G.nodes()))
    for v in nodes:
        for u in nodes:
            M[v-1, u-1] = M[v-1, u-1] * (1 - rho)

def select_best(M, ants: list, G: nx.Graph):
    best = 0
    elite_ant = None 
    for ant in ants:
        if best == 0:
            best = ant.n_colors
            elite_ant = ant 
        elif ant.n_colors < best:
            best = ant.n_colors 
            elite_ant = ant 
    elite_matrix = elite_ant.trail(G)
    M = M + elite_matrix
    return elite_ant.n_colors, elite_ant.assign_colors

def convert_int_lst(lst: list):
    lst = list(map(int, lst))
    return sorted(lst)

def solve(G: nx.Graph, n_ants, n_iters, a, b, evap):
    global M 
    global adj_mat
    global alpha 
    global beta 
    global n_nodes 
    alpha = a
    beta = b
    final_sol = {}
    final_cost = 0
    n_nodes = G.number_of_nodes()
    nodes_int = convert_int_lst(list(G.nodes()))
    adj_mat = create_adj_mat(G)
    colors = initialize_coloring(G)
    M = init_M(G)
    for i in range(n_iters):
        ants = create_ants(G, n_ants, colors)
        for ant in ants:
            ant.construct_coloring()   
        evap_trail(G, M, evap)
        best_dist, best_sol = select_best(M, ants, G)
        if final_cost == 0:
            final_cost = best_dist
            final_sol = best_sol
        elif best_dist < final_cost:
            final_cost = best_dist
            final_sol = best_sol
        print(final_cost)
    return final_sol, final_cost
        
    

    

graph = create_graph('data/gcol3.txt')
# lst = list(graph.nodes())
# lst = list(map(int, lst))
# print(sorted(lst))
# cols = initialize_coloring(graph)
# draw_graph(graph)
solve(graph, 20, 100, 0.6, 0.6, 0.5)