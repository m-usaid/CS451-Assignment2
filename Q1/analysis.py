from aco2 import * 
import pandas as pd
import numpy as np

# TODO:
# make analysis function 
##  plot data 

graph = create_graph('data/gcol1.txt')
# draw_graph(graph)

# iters = [_ for _ in range(10)]
# frame = pd.DataFrame(np.array(iters), columns=['iteration'])
avg, best = solve(graph, 20, 10, 1, 3, 0.5)
# print(avg)
d = {"Best Fitness": best, "Average Fitness": avg}
frame = pd.DataFrame(d)
print(frame)