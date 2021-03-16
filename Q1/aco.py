import networkx as nx

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

graph = create_graph('data/gcol1.txt')