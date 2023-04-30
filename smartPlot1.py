import numpy as np
from lp_solve import *
#np.random.seed(100)

from copy import deepcopy
import math
import os
import time
import matplotlib.pyplot as plt
from lp_solve import *
starttime = time.time()
plt.rcParams.update({'font.size': 30})
plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.weight"] = "bold"








file_numbers = [243,223,213]
gridsizesarray = ["small","medium","large"]
max_iterations = 300000# 100000
lg = []
markers = ['*','o','x']
for i in range(len(file_numbers)):
    file_number = file_numbers[i]
    gridsize = gridsizesarray[i]


    smart_model = np.load('smart_model{}.npy'.format(file_number))
    horizons = smart_model[0]
    states = smart_model[1]
    actions = smart_model[2]
    num_of_cells = (horizons+1)*states*actions*actions

    # horizons = 5
    # states = 100
    # actions = 5
    # horizons = np.load('horizons{}.npy'.format(file_number))#horizons 0,1,2, .. ,horizons ,( horizons+1 stages in total)
    # states = np.load('states{}.npy'.format(file_number))
    # actions = np.load('actions{}.npy'.format(file_number))
    lg.append('N = {}, |S|={}, |U|={}, |V|={}'.format(horizons, states, actions, actions,actions))
    #lg.append(gridsize)
    discount = 1

    # np.save('horizons{}.npy'.format(file_number),horizons)
    # np.save('states{}.npy'.format(file_number), states)
    # np.save('actions{}.npy'.format(file_number), actions)
    error = np.load('errorSmart{}.npy'.format(file_number))


    total_avgs = 1
    avgError = np.sum(error, axis=0)/total_avgs
    x = np.arange(len(avgError))

    plot1 = plt.figure(1)

    plt.plot(avgError/np.sqrt(num_of_cells))
    #plt.plot(np.sum(error, axis=0))
    #plt.yscale("log")
    plt.yticks(ticks=[0.5,0.75,1,1.25,1.5,1.75,2], labels=[0.5,0.75,1,1.25,1.5,1.75,2])

    #plt.yticks(np.arange(min(avgError), max(avgError) + 1, 10))
    #plt.title("Error over iterations for N={}, |S|={}, |A|={}".format(horizons,states,actions))
    #plt.title("Error over iterations for N={}, |S|={}, |A|={}".format(horizons, states, actions))
    #plt.xlabel("Number of iterations", fontweight='bold')

    plt.xlabel("Number of iterations")
    plt.ylabel("Error")
    leg = plt.legend(lg)
    leg.set_draggable(state=True)

    #plt.savefig("algorithm{}.png".format(file_number))
#plt.tight_layout()
plt.show()