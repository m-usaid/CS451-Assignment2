from aco2 import * 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO:
# make analysis function 
##  plot data 

def analyse_data(graph,param,n_iters=10):
    iters = [i for i in range(n_iters)]

    for a,b in param:
        avg, best = solve(graph, 20, n_iters, a, b, 0.5)
        d1 = {"Iterations": iters, "Average Fitness": avg, "Best Fitness": best } 
        frame = pd.DataFrame(d1,columns=["Iterations","Average Fitness","Best Fitness"])
        frame.plot(x ='Iterations', y=['Best Fitness', 'Average Fitness'], kind = 'line')
        plt.show()
        print(frame)


param_list = [(0.5, 0.5), (0.5, 0.8), (1, 1.5), (1, 3), (2, 5), (2, 6), (3, 8), (4, 8)]
param_list = [(0.5, 0.5)]

graph = create_graph('data/gcol1.txt')
# draw_graph(graph)


analyse_data(graph,param_list,50)