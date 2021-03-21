import networkx as nx
import numpy as np 
import random 
import matplotlib.pyplot as plt

class ant:
    def __init__(self) -> None:
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
        self.visited_vtx = []
        self.unvisited_vtx = keys.copy()
        # assign color to first node 
        if (len(self.visited_vtx) == 0):
           self.assign_color(self.start_vtx, self.avail_colors[0])
        return self 
    
    def assign_color(self, vtx, color):
        # assign color to node and add it to visited list 
        self.assign_colors[vtx] = color 
        self.visited_vtx.append(vtx)
        self.unvisited_vtx.remove(vtx)


    def construct_coloring(self):
        # construct a coloring using modified DSATUR algorithm
        n_unvisited = len(self.unvisited_vtx)
        tabu_lst = []
        # find a color for each unvisited node 
        for i in range(n_unvisited):
            nextvtx = self.next_vertex()
            tabu_lst = []
            for j in range(n_nodes):
                if adj_mat[nextvtx-1, j-1] == 1 :
                    # add colors of neighbors to tabu list 
                    tabu_lst.append((self.assign_colors[j+1]))
            for k in self.avail_colors:
                # assign smallest color that is not tabu 
                if (k not in tabu_lst):
                    self.assign_color(nextvtx, k)
                    break 
        # update number of colors used 
        self.n_colors = len(set(self.assign_colors.values()))


    def find_dsat(self, startvtx=None):
        # find d_sat value of vertex 
        if startvtx == None: 
            startvtx = self.start_vtx
        colored_nbrs = []
        for i in range(n_nodes):
            if adj_mat[startvtx-1, i-1] == 1:
                colored_nbrs.append((self.assign_colors[i+1]))
        return len(set(colored_nbrs))

    def M_ij(self, vtx1, vtx2):
        # return pheromone trail between two vertices 
        return M[vtx1-1, vtx2-1]

    def next_vertex(self):
        # select next vertex to color based on transition rules 
        if (len(self.unvisited_vtx) == 0):
            nextvtx = None 
        elif len(self.unvisited_vtx) == 1:
            nextvtx = self.unvisited_vtx[0]
        else:
            maxval = 0
            h_vals = [] # heuristic values for comparison and selection     
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
        # return ant's pheromone trail based on coloring 
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
    # construct networkx graph from file 
    graph = nx.Graph()
    edgeList = read_data(filename)
    graph.add_edges_from(edgeList)
    return graph    

def draw_graph(G: nx.Graph):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.savefig('graph.png')

def init_M(G: nx.Graph):
    # initialize pheromone matrix M 
    n = G.number_of_nodes()
    M = np.ones((n,n), float)
    for vtx in G:
        for nbr in G.neighbors(vtx):
            M[int(vtx)-1, int(nbr)-1] = 0
    return M

def create_adj_mat(G: nx.Graph):
    # create adjacency matrix 
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
    # create initial coloring based on grundy approx. number 
    colors = []
    grundy_num = len(nx.degree_histogram(G))
    for color in range(grundy_num):
        colors.append(color)
    return colors 

def evap_trail(G: nx.Graph, M, rho: float):
    # evaporate pheromones on all trains given a evaporation rate rho 
    nodes = convert_int_lst(list(G.nodes()))
    for v in nodes:
        for u in nodes:
            M[v-1, u-1] = M[v-1, u-1] * (1 - rho)

def select_best(M, ants: list, G: nx.Graph):
    # choose best ants to implement an elitism strategy 
    best = 0
    elite_ant = None 
    for ant in ants:
        if best == 0:
            best = ant.n_colors
            elite_ant = ant 
        elif ant.n_colors < best:
            best = ant.n_colors 
            elite_ant = ant 
    # update global matrix of pheromones 
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
    # final_sol = {}
    final_cost = 0
    n_nodes = G.number_of_nodes()
    adj_mat = create_adj_mat(G)
    colors = initialize_coloring(G)
    M = init_M(G)
    avg_fit = 0
    avgs = []
    best_fit = []
    for i in range(n_iters):
        ants = create_ants(G, n_ants, colors)
        for ant in ants:
            ant.construct_coloring()
            avg_fit += ant.n_colors   
        evap_trail(G, M, evap)
        best_dist, best_sol = select_best(M, ants, G)
        if final_cost == 0:
            final_cost = best_dist
            # final_sol = best_sol
        elif best_dist < final_cost:
            final_cost = best_dist
            # final_sol = best_sol
        avg_fit = avg_fit / n_ants
        avgs.append(avg_fit)
        best_fit.append(best_dist)
        print(avgs[-1], best_fit[-1])
    return avgs, best_fit 

# graph = create_graph('data/gcol1.txt')
# solve(graph, 20, 5, 3, 5, 0.5)

