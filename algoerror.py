import numpy as np
from copy import deepcopy
import math
import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})
plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.weight"] = "bold"
from lp_solve import  *
starttime = time.time()

#np.random.seed(100)
file_numbers = [12,13,14]
lg = []
markers = ['*','o','x']
for i in range(len(file_numbers)):
    file_number = file_numbers[i]

    # horizons = 5
    # states = 100
    # actions = 5
    horizons = np.load('horizons{}.npy'.format(file_number))#horizons 0,1,2, .. ,horizons ,( horizons+1 stages in total)
    states = np.load('states{}.npy'.format(file_number))
    actions = np.load('actions{}.npy'.format(file_number))
    num_of_cells = (horizons + 1) * states * actions * actions
    lg.append('N = {}, |S|={}, |U|={}, |V|={}'.format(horizons, states, actions, actions,actions))

    discount = 1
    #max_iterations = 100000# 100000
    # np.save('horizons{}.npy'.format(file_number),horizons)
    # np.save('states{}.npy'.format(file_number), states)
    # np.save('actions{}.npy'.format(file_number), actions)
    error = np.load('error{}.npy'.format(file_number*10+3))

    total_avgs = 1
    avgError = np.sum(error, axis=0)/total_avgs
    x = np.arange(len(avgError))

    plot1 = plt.figure(1)
    #plt.plot(avgError)
    plt.plot(avgError / np.sqrt(num_of_cells))
    #plt.plot(np.sum(error, axis=0))
    plt.yscale("log")
    plt.yticks(ticks=[0.025,0.05,0.1,0.2,0.4,0.8,1.6,3.2,6.4], labels=[0.025,0.05,0.1,0.2,0.4,0.8,1.6,3.2,6.4])
    #plt.yticks([1,10,100,1000])
    #plt.title("Error over iterations for N={}, |S|={}, |A|={}".format(horizons,states,actions))
    #plt.title("Error over iterations for N={}, |S|={}, |A|={}".format(horizons, states, actions))
    #plt.xlabel("Number of iterations", fontweight='bold')
    plt.xlabel("Number of iterations")
    plt.ylabel("Error (log scale)")
    plt.legend(lg)
    #plt.savefig("algorithm{}.png".format(file_number))

plt.show()

