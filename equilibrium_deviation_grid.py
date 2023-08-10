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
plt.rcParams.update({'font.size': 32})
plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.weight"] = "bold"

file_number = 1 #2,3 for other cases
print("file_number = {}".format(file_number))
#np.random.seed(100)
#random.seed(100)
print(os.path.basename(__file__))
smart_model = np.load('smart_model{}.npy'.format(file_number))
completeActionSet1 = np.load("completeActionSet1_{}.npy".format(file_number))
completeActionSet2 = np.load("completeActionSet2_{}.npy".format(file_number))
completeStateSet = np.load("completeStateSet{}.npy".format(file_number))
P = np.load("ProbabilityMatrix{}.npy".format(file_number))
R = np.load("RewardMatrix{}.npy".format(file_number))
Q = np.load('optimalQMatrix{}.npy'.format(file_number))

horizons = smart_model[0]
states = smart_model[1]
actions = smart_model[2]


maxprice1 = 2
minprice1 = 1
maxprice2 = 2
minprice2 = 1
maxPurchase1 = 2
maxPurchase2 = 2


#print(completeActionSet1)



def getActionIndex1(purchase1, price1): # purchase1 starts from 1 and price1 from 0
    return purchase1 * maxprice1  + price1 - 1

def getActionIndex2(purchase2, price2): # purchase2 starts from 1 and price2 from 0
    return purchase2 * maxprice2 + price2 -1

def getDeterministicPolicy(index):
    pol = np.array([0]*actions)
    pol[index] = 1
    return pol

def getSmartBaseline1Max():
    pol = np.array([0] * actions)
    index = getActionIndex1(maxPurchase1, minprice1)
    pol[index] = 1
    return pol

def getSmartBaseline1Min():
    pol = np.array([0] * actions)
    index = getActionIndex2(maxPurchase2, minprice2)
    pol[index] = 1
    return pol

def getSmartBaseline2Max():
    pol = np.array([0] * actions)
    index = getActionIndex1(maxPurchase1, maxprice1)
    pol[index] = 1
    return pol

def getSmartBaseline2Min():
    pol = np.array([0] * actions)
    index = getActionIndex2(maxPurchase2, maxprice2)
    pol[index] = 1
    return pol

def getRandomSingletonPolicy():
    pol = np.array([0]*actions)
    index = np.random.randint(0,actions)
    pol[index] = 1
    return pol


def getmixedPolicy():
    arr = np.random.rand(actions)
    arr = arr / arr.sum()
    return arr

def getRandomBinaryPolicy():
    pol = np.array([0] * actions)
    index1, index2 = np.random.randint(low=0, high=actions, size=2)
    pol[index1] = 0.5
    pol[index2] = 0.5


#np.load('errorSmart{}'.format(file_number))

print(Q.shape)
print(horizons)

#np.save('optimalMinimaxValue{}.npy'.format(file_number), sor_minimax_Q)

policyMatrixLearnedMaxPlayer = [np.arange(actions), np.arange(actions), np.arange(actions)]
policyMatrixLearnedMaxPlayer.clear()
policyMatrixLearnedMinPlayer = [np.arange(actions), np.arange(actions), np.arange(actions)]
policyMatrixLearnedMinPlayer.clear()
valueMatrixLearned = np.zeros((horizons+1, states))


for h in range(horizons + 1):
    for s in range(states):

        policy_for_maxPlayer, minmax_for_maxPlayer = solve_zero_sum_game(Q[h, s, :, :])
        policy_for_minPlayer, minmax_for_minPlayer = solve_zero_sum_game(-Q[h, s, :, :].transpose())

        valueMatrixLearned[h,s] = minmax_for_maxPlayer

        policyMatrixLearnedMaxPlayer.append(policy_for_maxPlayer)

        policyMatrixLearnedMinPlayer.append(policy_for_minPlayer)



print("AVERAGE COST CALCULATION USING Learned Policy ")

total_Cost = np.zeros([horizons+1, states])
hit_Count = np.zeros([horizons+1, states])

max_iterations = 100000
#print(horizons,states)
for n in range(max_iterations):
    #change of episode
    h = np.random.randint(0, horizons+1)  #exclusive of upperbound
    state = np.random.randint(0, states)  #exclusive of upperbound

    episode_sum = 0
    episode_h = h
    episode_state = state

    hit_Count[h , state] += 1

    while(h<=horizons):
        polMax = policyMatrixLearnedMaxPlayer[h*states + state]
        polMin = policyMatrixLearnedMinPlayer[h*states + state]

        act1 = int(np.random.choice(np.arange(actions), 1, p=polMax))
        act2 = int(np.random.choice(np.arange(actions), 1, p=polMin))

        s_new = int(np.random.choice(np.arange(states), 1, p=P[h, state, act1, act2, :]))

        r = R[h, state, act1, act2]

        episode_sum += r

        state = s_new

        h = h + 1

    total_Cost[episode_h, episode_state] += episode_sum


avgCost_Qlearning = np.zeros([horizons+1, states])
#print("average Cost Calculation")
for h in range(horizons+1):
    for s in range(states):
        avgCost_Qlearning[h,s] = total_Cost[h,s]/hit_Count[h,s]

print(np.mean(avgCost_Qlearning))

np.save("avgCostQLearn{}.npy".format(file_number), avgCost_Qlearning)


print("AVERAGE COST CALCULATION USING Baseline Policy when MAX player deviates to a random mixed policy")

total_Cost = np.zeros([horizons + 1, states])
hit_Count = np.zeros([horizons + 1, states])

max_iterations = 100000
# print(horizons,states)
for n in range(max_iterations):
    # change of episode
    h = np.random.randint(0, horizons + 1)  # exclusive of upperbound
    state = np.random.randint(0, states)  # exclusive of upperbound
    d,b1,b2,price_curr = completeStateSet[state]
    episode_sum = 0
    episode_h = h
    episode_state = state

    hit_Count[h, state] += 1

    while (h <= horizons):
        #polMax = policyMatrixLearnedMaxPlayer[h * states + state]
        polMin = policyMatrixLearnedMinPlayer[h * states + state]
        polMax = getmixedPolicy()

        act1 = int(np.random.choice(np.arange(actions), 1, p=polMax))
        act2 = int(np.random.choice(np.arange(actions), 1, p=polMin))

        s_new = int(np.random.choice(np.arange(states), 1, p=P[h, state, act1, act2, :]))

        r = R[h, state, act1, act2]

        episode_sum += r

        state = s_new

        h = h + 1

    total_Cost[episode_h, episode_state] += episode_sum



avgCost_Max_deviates = np.zeros([horizons + 1, states])
#print("average Cost Calculation")
for h in range(horizons + 1):
    for s in range(states):
        avgCost_Max_deviates[h, s] = total_Cost[h, s] / hit_Count[h, s]

print(np.mean(avgCost_Max_deviates))

np.save("max_dev_mixed{}.npy".format(file_number), avgCost_Max_deviates)


print("AVERAGE COST CALCULATION USING Baseline Policy when MIN player deviates to a random mixed policy")

total_Cost = np.zeros([horizons+1 ,states])
hit_Count = np.zeros([horizons+1 , states])

max_iterations = 100000
#print(horizons,states)
for n in range(max_iterations):
    #change of episode
    h = np.random.randint(0, horizons+1)  #exclusive of upperbound
    state = np.random.randint(0, states)  #exclusive of upperbound

    d, b1, b2, price_curr = completeStateSet[state]
    episode_sum = 0
    episode_h = h
    episode_state = state

    hit_Count[h , state] += 1

    while(h<=horizons):
        polMax = policyMatrixLearnedMaxPlayer[h*states + state]
        polMin = getmixedPolicy()
        #polMin = policyMatrixLearnedMinPlayer[h*states + state]

        #pos = np.random.randint(0, actions)
        #pos = state % 3
        # pos = getActionIndex2(d, minprice2)
        # polMin = getDeterministicPolicy(pos)

        #polMin = [(1/actions)]*actions
        #polMin = getmixedPolicy()

        act1 = int(np.random.choice(np.arange(actions), 1, p=polMax))
        act2 = int(np.random.choice(np.arange(actions), 1, p=polMin))

        s_new = int(np.random.choice(np.arange(states), 1, p=P[h, state, act1, act2, :]))

        r = R[h, state, act1, act2]

        episode_sum += r

        state = s_new

        h = h + 1

    total_Cost[episode_h, episode_state] += episode_sum



avgCost_Min_Deviates = np.zeros([horizons+1 ,states])
#print("average Cost Calculation")
for h in range(horizons+1):
    for s in range(states):
        avgCost_Min_Deviates[h,s] = total_Cost[h,s]/hit_Count[h,s]

print(np.mean(avgCost_Min_Deviates))
np.save("min_dev_mixed{}.npy".format(file_number), avgCost_Min_Deviates)

# print("AVERAGE COST CALCULATION USING Baseline Policy when MAX player deviates to a random singleton policy")
#
# total_Cost = np.zeros([horizons + 1, states])
# hit_Count = np.zeros([horizons + 1, states])
#
# max_iterations = 100000
# # print(horizons,states)
# for n in range(max_iterations):
#     # change of episode
#     h = np.random.randint(0, horizons + 1)  # exclusive of upperbound
#     state = np.random.randint(0, states)  # exclusive of upperbound
#     d,b1,b2,price_curr = completeStateSet[state]
#     episode_sum = 0
#     episode_h = h
#     episode_state = state
#
#     hit_Count[h, state] += 1
#
#     while (h <= horizons):
#         #polMax = policyMatrixLearnedMaxPlayer[h * states + state]
#         polMin = policyMatrixLearnedMinPlayer[h * states + state]
#         polMax = getRandomSingletonPolicy()
#         #polMax = [(1 / actions)] * actions
#         #pos = np.random.randint(0, actions)
#         #pos = state%3
#         # pos = getActionIndex1(d,minprice1)
#         # polMax = getDeterministicPolicy(pos)
#         #polMax = getmixedPolicy()
#
#         act1 = int(np.random.choice(np.arange(actions), 1, p=polMax))
#         act2 = int(np.random.choice(np.arange(actions), 1, p=polMin))
#
#         s_new = int(np.random.choice(np.arange(states), 1, p=P[h, state, act1, act2, :]))
#
#         r = R[h, state, act1, act2]
#
#         episode_sum += r
#
#         state = s_new
#
#         h = h + 1
#
#     total_Cost[episode_h, episode_state] += episode_sum
#
#
#
# avgCost_Max_deviates = np.zeros([horizons + 1, states])
# print("average Cost Calculation")
# for h in range(horizons + 1):
#     for s in range(states):
#         avgCost_Max_deviates[h, s] = total_Cost[h, s] / hit_Count[h, s]
#
# print(np.mean(avgCost_Max_deviates))
# np.save("max_dev_singleton{}.npy".format(file_number), avgCost_Max_deviates)
#
# print("AVERAGE COST CALCULATION USING Baseline Policy when MIN player deviates to a random singleton policy")
#
# total_Cost = np.zeros([horizons+1 ,states])
# hit_Count = np.zeros([horizons+1 , states])
#
# max_iterations = 100000
# #print(horizons,states)
# for n in range(max_iterations):
#     #change of episode
#     h = np.random.randint(0, horizons+1)  #exclusive of upperbound
#     state = np.random.randint(0, states)  #exclusive of upperbound
#
#     d, b1, b2, price_curr = completeStateSet[state]
#     episode_sum = 0
#     episode_h = h
#     episode_state = state
#
#     hit_Count[h , state] += 1
#
#     while(h<=horizons):
#         polMax = policyMatrixLearnedMaxPlayer[h*states + state]
#         polMin = getRandomSingletonPolicy()
#         #polMin = policyMatrixLearnedMinPlayer[h*states + state]
#
#         #pos = np.random.randint(0, actions)
#         #pos = state % 3
#         # pos = getActionIndex2(d, minprice2)
#         # polMin = getDeterministicPolicy(pos)
#
#         #polMin = [(1/actions)]*actions
#         #polMin = getmixedPolicy()
#
#         act1 = int(np.random.choice(np.arange(actions), 1, p=polMax))
#         act2 = int(np.random.choice(np.arange(actions), 1, p=polMin))
#
#         s_new = int(np.random.choice(np.arange(states), 1, p=P[h, state, act1, act2, :]))
#
#         r = R[h, state, act1, act2]
#
#         episode_sum += r
#
#         state = s_new
#
#         h = h + 1
#
#     total_Cost[episode_h, episode_state] += episode_sum
#
#
#
# avgCost_Min_Deviates = np.zeros([horizons+1 ,states])
# print("average Cost Calculation")
# for h in range(horizons+1):
#     for s in range(states):
#         avgCost_Min_Deviates[h,s] = total_Cost[h,s]/hit_Count[h,s]
#
# print(np.mean(avgCost_Min_Deviates))
# np.save("min_dev_singleton{}.npy".format(file_number), avgCost_Min_Deviates)
#
#
#
# print("AVERAGE COST CALCULATION USING Baseline Policy when MAX player deviates to a random binary policy")
#
# total_Cost = np.zeros([horizons + 1, states])
# hit_Count = np.zeros([horizons + 1, states])
#
# max_iterations = 100000
# # print(horizons,states)
# for n in range(max_iterations):
#     # change of episode
#     h = np.random.randint(0, horizons + 1)  # exclusive of upperbound
#     state = np.random.randint(0, states)  # exclusive of upperbound
#     d,b1,b2,price_curr = completeStateSet[state]
#     episode_sum = 0
#     episode_h = h
#     episode_state = state
#
#     hit_Count[h, state] += 1
#
#     while (h <= horizons):
#         #polMax = policyMatrixLearnedMaxPlayer[h * states + state]
#         polMin = policyMatrixLearnedMinPlayer[h * states + state]
#         polMax = getRandomBinaryPolicy()
#         #polMax = [(1 / actions)] * actions
#         #pos = np.random.randint(0, actions)
#         #pos = state%3
#         # pos = getActionIndex1(d,minprice1)
#         # polMax = getDeterministicPolicy(pos)
#         #polMax = getmixedPolicy()
#
#         act1 = int(np.random.choice(np.arange(actions), 1, p=polMax))
#         act2 = int(np.random.choice(np.arange(actions), 1, p=polMin))
#
#         s_new = int(np.random.choice(np.arange(states), 1, p=P[h, state, act1, act2, :]))
#
#         r = R[h, state, act1, act2]
#
#         episode_sum += r
#
#         state = s_new
#
#         h = h + 1
#
#     total_Cost[episode_h, episode_state] += episode_sum
#
#
#
# avgCost_Max_deviates = np.zeros([horizons + 1, states])
# print("average Cost Calculation")
# for h in range(horizons + 1):
#     for s in range(states):
#         avgCost_Max_deviates[h, s] = total_Cost[h, s] / hit_Count[h, s]
#
# print(np.mean(avgCost_Max_deviates))
# np.save("max_dev_binary{}.npy".format(file_number), avgCost_Max_deviates)
#
# print("AVERAGE COST CALCULATION USING Baseline Policy when MIN player deviates to a random binary policy")
#
# total_Cost = np.zeros([horizons+1 ,states])
# hit_Count = np.zeros([horizons+1 , states])
#
# max_iterations = 100000
# #print(horizons,states)
# for n in range(max_iterations):
#     #change of episode
#     h = np.random.randint(0, horizons+1)  #exclusive of upperbound
#     state = np.random.randint(0, states)  #exclusive of upperbound
#
#     d, b1, b2, price_curr = completeStateSet[state]
#     episode_sum = 0
#     episode_h = h
#     episode_state = state
#
#     hit_Count[h , state] += 1
#
#     while(h<=horizons):
#         polMax = policyMatrixLearnedMaxPlayer[h*states + state]
#         polMin = getRandomBinaryPolicy()
#         #polMin = policyMatrixLearnedMinPlayer[h*states + state]
#
#         #pos = np.random.randint(0, actions)
#         #pos = state % 3
#         # pos = getActionIndex2(d, minprice2)
#         # polMin = getDeterministicPolicy(pos)
#
#         #polMin = [(1/actions)]*actions
#         #polMin = getmixedPolicy()
#
#         act1 = int(np.random.choice(np.arange(actions), 1, p=polMax))
#         act2 = int(np.random.choice(np.arange(actions), 1, p=polMin))
#
#         s_new = int(np.random.choice(np.arange(states), 1, p=P[h, state, act1, act2, :]))
#
#         r = R[h, state, act1, act2]
#
#         episode_sum += r
#
#         state = s_new
#
#         h = h + 1
#
#     total_Cost[episode_h, episode_state] += episode_sum
#
#
#
# avgCost_Min_Deviates = np.zeros([horizons+1 ,states])
# print("average Cost Calculation")
# for h in range(horizons+1):
#     for s in range(states):
#         avgCost_Min_Deviates[h,s] = total_Cost[h,s]/hit_Count[h,s]
#
# print(np.mean(avgCost_Min_Deviates))
# np.save("min_dev_binary{}.npy".format(file_number), avgCost_Min_Deviates)




print("AVERAGE COST CALCULATION USING Baseline Policy when MAX player deviates to smart baseline 1")

total_Cost = np.zeros([horizons + 1, states])
hit_Count = np.zeros([horizons + 1, states])

max_iterations = 100000
# print(horizons,states)
for n in range(max_iterations):
    # change of episode
    h = np.random.randint(0, horizons + 1)  # exclusive of upperbound
    state = np.random.randint(0, states)  # exclusive of upperbound
    d,b1,b2,price_curr = completeStateSet[state]
    episode_sum = 0
    episode_h = h
    episode_state = state

    hit_Count[h, state] += 1

    while (h <= horizons):
        #polMax = policyMatrixLearnedMaxPlayer[h * states + state]
        polMin = policyMatrixLearnedMinPlayer[h * states + state]
        polMax = getSmartBaseline1Max()
        #polMax = [(1 / actions)] * actions
        #pos = np.random.randint(0, actions)
        #pos = state%3
        # pos = getActionIndex1(d,minprice1)
        # polMax = getDeterministicPolicy(pos)
        #polMax = getmixedPolicy()

        act1 = int(np.random.choice(np.arange(actions), 1, p=polMax))
        act2 = int(np.random.choice(np.arange(actions), 1, p=polMin))

        s_new = int(np.random.choice(np.arange(states), 1, p=P[h, state, act1, act2, :]))

        r = R[h, state, act1, act2]

        episode_sum += r

        state = s_new

        h = h + 1

    total_Cost[episode_h, episode_state] += episode_sum



avgCost_Max_deviates = np.zeros([horizons + 1, states])
#print("average Cost Calculation")
for h in range(horizons + 1):
    for s in range(states):
        avgCost_Max_deviates[h, s] = total_Cost[h, s] / hit_Count[h, s]

print(np.mean(avgCost_Max_deviates))
np.save("max_dev_base1{}.npy".format(file_number), avgCost_Max_deviates)

print("AVERAGE COST CALCULATION USING Baseline Policy when MIN player deviates to smart baseline 1")

total_Cost = np.zeros([horizons+1 ,states])
hit_Count = np.zeros([horizons+1 , states])

max_iterations = 100000
#print(horizons,states)
for n in range(max_iterations):
    #change of episode
    h = np.random.randint(0, horizons+1)  #exclusive of upperbound
    state = np.random.randint(0, states)  #exclusive of upperbound

    d, b1, b2, price_curr = completeStateSet[state]
    episode_sum = 0
    episode_h = h
    episode_state = state

    hit_Count[h , state] += 1

    while(h<=horizons):
        polMax = policyMatrixLearnedMaxPlayer[h*states + state]
        polMin = getSmartBaseline1Min()
        #polMin = policyMatrixLearnedMinPlayer[h*states + state]

        #pos = np.random.randint(0, actions)
        #pos = state % 3
        # pos = getActionIndex2(d, minprice2)
        # polMin = getDeterministicPolicy(pos)

        #polMin = [(1/actions)]*actions
        #polMin = getmixedPolicy()

        act1 = int(np.random.choice(np.arange(actions), 1, p=polMax))
        act2 = int(np.random.choice(np.arange(actions), 1, p=polMin))

        s_new = int(np.random.choice(np.arange(states), 1, p=P[h, state, act1, act2, :]))

        r = R[h, state, act1, act2]

        episode_sum += r

        state = s_new

        h = h + 1

    total_Cost[episode_h, episode_state] += episode_sum



avgCost_Min_Deviates = np.zeros([horizons+1 ,states])
#print("average Cost Calculation")
for h in range(horizons+1):
    for s in range(states):
        avgCost_Min_Deviates[h,s] = total_Cost[h,s]/hit_Count[h,s]

print(np.mean(avgCost_Min_Deviates))
np.save("min_dev_base1{}.npy".format(file_number), avgCost_Min_Deviates)


print("AVERAGE COST CALCULATION USING Baseline Policy when MAX player deviates to smart baseline 2")

total_Cost = np.zeros([horizons + 1, states])
hit_Count = np.zeros([horizons + 1, states])

max_iterations = 100000
# print(horizons,states)
for n in range(max_iterations):
    # change of episode
    h = np.random.randint(0, horizons + 1)  # exclusive of upperbound
    state = np.random.randint(0, states)  # exclusive of upperbound
    d,b1,b2,price_curr = completeStateSet[state]
    episode_sum = 0
    episode_h = h
    episode_state = state

    hit_Count[h, state] += 1

    while (h <= horizons):
        #polMax = policyMatrixLearnedMaxPlayer[h * states + state]
        polMin = policyMatrixLearnedMinPlayer[h * states + state]
        polMax = getSmartBaseline2Max()
        #polMax = [(1 / actions)] * actions
        #pos = np.random.randint(0, actions)
        #pos = state%3
        # pos = getActionIndex1(d,minprice1)
        # polMax = getDeterministicPolicy(pos)
        #polMax = getmixedPolicy()

        act1 = int(np.random.choice(np.arange(actions), 1, p=polMax))
        act2 = int(np.random.choice(np.arange(actions), 1, p=polMin))

        s_new = int(np.random.choice(np.arange(states), 1, p=P[h, state, act1, act2, :]))

        r = R[h, state, act1, act2]

        episode_sum += r

        state = s_new

        h = h + 1

    total_Cost[episode_h, episode_state] += episode_sum



avgCost_Max_deviates = np.zeros([horizons + 1, states])
#print("average Cost Calculation")
for h in range(horizons + 1):
    for s in range(states):
        avgCost_Max_deviates[h, s] = total_Cost[h, s] / hit_Count[h, s]

print(np.mean(avgCost_Max_deviates))
np.save("max_dev_base2{}.npy".format(file_number), avgCost_Max_deviates)

print("AVERAGE COST CALCULATION USING Baseline Policy when MIN player deviates to smart baseline 2")

total_Cost = np.zeros([horizons+1 ,states])
hit_Count = np.zeros([horizons+1 , states])

max_iterations = 100000
#print(horizons,states)
for n in range(max_iterations):
    #change of episode
    h = np.random.randint(0, horizons+1)  #exclusive of upperbound
    state = np.random.randint(0, states)  #exclusive of upperbound

    d, b1, b2, price_curr = completeStateSet[state]
    episode_sum = 0
    episode_h = h
    episode_state = state

    hit_Count[h , state] += 1

    while(h<=horizons):
        polMax = policyMatrixLearnedMaxPlayer[h*states + state]
        polMin = getSmartBaseline2Min()
        #polMin = policyMatrixLearnedMinPlayer[h*states + state]

        #pos = np.random.randint(0, actions)
        #pos = state % 3
        # pos = getActionIndex2(d, minprice2)
        # polMin = getDeterministicPolicy(pos)

        #polMin = [(1/actions)]*actions
        #polMin = getmixedPolicy()

        act1 = int(np.random.choice(np.arange(actions), 1, p=polMax))
        act2 = int(np.random.choice(np.arange(actions), 1, p=polMin))

        s_new = int(np.random.choice(np.arange(states), 1, p=P[h, state, act1, act2, :]))

        r = R[h, state, act1, act2]

        episode_sum += r

        state = s_new

        h = h + 1

    total_Cost[episode_h, episode_state] += episode_sum



avgCost_Min_Deviates = np.zeros([horizons+1 ,states])
#print("average Cost Calculation")
for h in range(horizons+1):
    for s in range(states):
        avgCost_Min_Deviates[h,s] = total_Cost[h,s]/hit_Count[h,s]

print(np.mean(avgCost_Min_Deviates))
np.save("min_dev_base2{}.npy".format(file_number), avgCost_Min_Deviates)

file_numbers = [243,223,213]
lg = []
markers = ['*','o','x']
for i in range(len(file_numbers)):
    file_number = file_numbers[i]

    smart_model = np.load('smart_model{}.npy'.format(file_number))
    horizons = smart_model[0]
    states = smart_model[1]
    actions = smart_model[2]


    # horizons = 5
    # states = 100
    # actions = 5
    # horizons = np.load('horizons{}.npy'.format(file_number))#horizons 0,1,2, .. ,horizons ,( horizons+1 stages in total)
    # states = np.load('states{}.npy'.format(file_number))
    # actions = np.load('actions{}.npy'.format(file_number))
    lg.append('N = {}, |S|={}, |U|={}, |V|={}'.format(horizons, states, actions, actions,actions))

    discount = 1
    max_iterations = 300000# 100000
    # np.save('horizons{}.npy'.format(file_number),horizons)
    # np.save('states{}.npy'.format(file_number), states)
    # np.save('actions{}.npy'.format(file_number), actions)
    error = np.load('errorSmart{}.npy'.format(file_number))


    total_avgs = 1
    avgError = np.sum(error, axis=0)/total_avgs
    x = np.arange(len(avgError))

    plot1 = plt.figure(1)
    plt.plot(avgError)
    #plt.plot(np.sum(error, axis=0))
    plt.yscale("log")
    #plt.title("Error over iterations for N={}, |S|={}, |A|={}".format(horizons,states,actions))
    #plt.title("Error over iterations for N={}, |S|={}, |A|={}".format(horizons, states, actions))
    plt.xlabel("Number of iterations", fontweight='bold')
    plt.ylabel("Error (log scale)", fontweight='bold')
    leg = plt.legend(lg)
    leg.set_draggable(state=True)

    #plt.savefig("algorithm{}.png".format(file_number))
plt.tight_layout()
plt.show()
