from aco2 import * 
import pandas as pd
import numpy as np

# TODO:
# make analysis function 
##  plot data 

param_list = [(0.5, 0.5), (0.5, 0.8), (1, 1.5), (1, 3), (2, 5), (2, 6), (3, 8), (4, 8)]

graph = create_graph('data/gcol1.txt')
# draw_graph(graph)




def analyse_data():

    avg, best = solve(graph, 20, 10, 1, 3, 0.5)
    d = {"Best Fitness": best, "Average Fitness": avg}
    frame = pd.DataFrame(d)
    print(frame)