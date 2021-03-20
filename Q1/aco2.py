import networkx as nx
import numpy as np 
import random 
import matplotlib.pyplot as plt

class ant:
    def __init__(self, alpha: int, beta: int) -> None:
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
        self.avail_colors = sorted(colorings) #.copy()
        keys = list(G.nodes())
        self.assign_colors = {key: None for key in keys}
        if startvtx == None : self.start_vtx = random.choice(list(G.nodes()))
        else: 
            self.start_vtx = startvtx
        if (len(self.visited_vtx)==0):
            # assign color to node 
            pass 
    
    def assign_color(self, vtx, color):
        self.assign_colors[vtx] = color 
        self.visited_vtx.append(vtx)
        self.unvisited_vtx.remove(vtx)
    

    def construct_coloring(self):
        pass 

    def find_dsat(self, startvtx=None):
        pass

    def M_ij(self, vtx1, vtx2):
        pass 

    def next_vertex(self):
        pass

    def trail(self):
        pass

    def check_equal(self):
        pass

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
            M[vtx, nbr] = 0
    return M

def create_adj_mat(G: nx.Graph):
    n = G.number_of_nodes()
    adj_mat = np.zeros((n,n), int)
    for node in list(G.nodes()):
        for nbr in G.neighbors(node):
            adj_mat[node, nbr] = 1
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
    for v in G.nodes():
        for u in G.nodes():
            M[v, u] = M[v, u] * (1 - rho)

def select_best(M, ants):
    best = 0
    elite_ant = None 
    for ant in ants:
        if best == 0:
            best = ant.n_colors
            elite_ant = ant 
        elif ant.n_colors < best:
            best = ant.n_colors 
            elite_ant = ant 
    elite_matrix = elite_ant.trail()
    M = M + elite_matrix
    return elite_ant.best, elite_ant.assign_colors

def solve(G: nx.Graph, n_ants, n_iters, alpha, beta, evap):
    pass 

graph = create_graph('data/gcol1.txt')

# cols = initialize_coloring(graph)
# draw_graph(graph)
