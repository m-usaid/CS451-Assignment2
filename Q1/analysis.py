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
        plt.title('ACO for alpha = '+str(a)+', beta = '+str(b))
        plt.savefig('plot-'+str(a)+'-'+str(b)+'.png')
        plt.show()
        print(frame)

def analyse_data_param_a(graph,aList,b,n_iters=15):
    iters = [i for i in range(n_iters)]

    avgList = []
    bestList = []
    for a in aList:
        avg, best = solve(graph, 20, n_iters, a, b, 0.5)
        avgList.append(min(avg))
        bestList.append(min(best))
    

    d1 = {"Alpha": aList, "Average Fitness": avgList, "Best Fitness": bestList } 
    frame = pd.DataFrame(d1,columns=["Alpha","Average Fitness","Best Fitness"])
    frame.plot(x ='Alpha', y=['Best Fitness', 'Average Fitness'], style='o')
    plt.title('Maximum fitness of ACO against alpha')
    plt.savefig('plot-Varying-Alpha.png')
    plt.show()
    print(frame)

def analyse_data_param_b(graph,bList,a,n_iters=15):
    iters = [i for i in range(n_iters)]

    avgList = []
    bestList = []
    for b in bList:
        avg, best = solve(graph, 20, n_iters, a, b, 0.5)
        avgList.append(min(avg))
        bestList.append(min(best))
    

    d1 = {"Beta": bList, "Average Fitness": avgList, "Best Fitness": bestList } 
    frame = pd.DataFrame(d1,columns=["Beta","Average Fitness","Best Fitness"])
    frame.plot(x ='Alpha', y=['Best Fitness', 'Average Fitness'], style='o')
    plt.title('Maximum fitness of ACO against Beta')
    plt.savefig('plot-Varying-Beta.png')
    plt.show()
    print(frame)


param_list = [(0.5, 0.5), (0.5, 0.8), (1, 1.5), (1, 3), (2, 5), (2, 6), (3, 8), (3, 5)]
#param_list = [(0.8, 0.8)]

graph = create_graph('data/gcol1.txt')
# draw_graph(graph)


# analyse_data(graph,param_list,10)


bList = [0.5,0.7,0.8,0.9,1,2]
a = 0.8
analyse_data_param_b(graph,bList,a)