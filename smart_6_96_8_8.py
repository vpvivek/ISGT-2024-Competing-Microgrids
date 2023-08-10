
import numpy as np
from lp_solve import *
#np.random.seed(100)

from copy import deepcopy
import math
import os
import time
import matplotlib.pyplot as plt
from lp_solve import *
file_number = 2
starttime = time.time()


#np.random.seed(100)
#random.seed(100)
print(os.path.basename(__file__))
print("Medium Set Up")
print("File number ", file_number)
print("STARTED")

maxPurchase1 = 2
maxPurchase2 = 2
#House parameters

demand_array = np.array([1, 2, 3]) # 3 possible demands valued 1,2,3
maximum_demand = max(demand_array)
demandTPM = np.array([[0.3, 0.25, 0.45], [0.25, 0.40, 0.35], [0.30, .15, .55]])

def getNextDemand(k):
    return int(np.random.choice(demand_array, 1, p=demandTPM[k-1, :]))


#Microgrid 1

renewal_array1 = np.array([1, 2, 3]) # 3 possible renewal energy valued 1,2,3
renewalProb1 = np.array([0.3, 0.45, 0.25])

def getNextRenewal1():
    return int(np.random.choice(renewal_array1, 1, p=renewalProb1))

price_array1 = np.array([1, 2]) # 3 possible prices valued 1,2,3
maxprice1 = max(price_array1)
minprice1 = min(price_array1)


maximum_battery1 = 3  #capacity of battery
current_battery1 = 0  #available power at a particular instant of time

#Microgrid 2
    #renewal
renewal_array2 = np.array([1,2,3])  #3 possible renewal energy valued 1,2,3
renewalProb2 = np.array([0.25, 0.45, 0.3])

def getNextRenewal2():
    return int(np.random.choice(renewal_array2, 1, p=renewalProb2))

    #price, please note these are going to be actions(decisions) and not evolve with system
price_array2 = np.array([1, 2])  #3 possible prices valued 1,2,3
maxprice2 = max(price_array2)
minprice2 = min(price_array2)

    #battery

maximum_battery2 = 3 #capacity of battery
current_battery2 = 0 #available power at a particular instant of time

#Main grid

price_array_Main = np.array([1, 2]) # 3 possible prices valued 1,2,3
maximum_Mainprice = max(price_array_Main)
priceTPM_Main = np.array([[0.33, 0.67], [0.60, 0.40]])
maximum_MainGrid = 3

def getNextPriceMain(k):
    return int(np.random.choice(price_array_Main, 1, p=priceTPM_Main[k-1, :]))



#MDP terms
horizons = 6
No_of_states = maximum_demand * (maximum_battery1 + 1) * (maximum_battery2 + 1) * max(price_array_Main)


#States, total states, number of states, state to state number mapping



#Actions, maximal set of actions, action to action number mapping, incorporate minimum price
completeActionSet1 = [(purchase1, priceQuote1) for purchase1 in range(maximum_MainGrid + 1) for priceQuote1 in range(1, maxprice1 + 1)]
completeActionSet2 = [(purchase2, priceQuote2) for purchase2 in range(maximum_MainGrid + 1) for priceQuote2 in range(1, maxprice2 + 1)]
completeStateSet = [(d, b1, b2, p) for d in demand_array for b1 in range(maximum_battery1+1) \
                    for b2 in range(maximum_battery2+1) for p in price_array_Main]

np.save("completeStateSet{}.npy".format(file_number), completeStateSet)
np.save("completeActionSet1_{}.npy".format(file_number), completeActionSet1)
np.save("completeActionSet2_{}.npy".format(file_number), completeActionSet2)


print("horizons", horizons)
print("completeStateSet ", len(completeStateSet))
print(completeStateSet)
print("completeActionSet1", len(completeActionSet1))
print(completeActionSet1)
print("completeActionSet2", len(completeActionSet2))
print(completeActionSet2)


smart_model = [0]*4

smart_model[0] = horizons
smart_model[1] = len(completeStateSet)
smart_model[2] = len(completeActionSet1)
smart_model[3] = len(completeActionSet2)

np.save('smart_model{}.npy'.format(file_number),smart_model)

print(smart_model)



def getActionIndex1(purchase1, price1): # purchase1 starts from 1 and price1 from 0
    return purchase1 * maxprice1  + price1 - 1

def getActionIndex2(purchase2, price2): # purchase2 starts from 1 and price2 from 0
    return purchase2 * maxprice2 + price2 -1




def getAction1fromIndex(ind):
    return completeActionSet1[ind]

def getAction2fromIndex(ind):
    return completeActionSet2[ind]


# d : from 1 to maximum_demand , b1 : from 0 to maximum_battery1
# b2 : from 0 to maximum_battery2,  priceMain: 1 to maximum_Mainprice
def getStateFromIndex(ind):
    return completeStateSet[ind]


def getStateIndex(d, b1, b2, priceMain):
    rank =  \
    (d-1)* (maximum_battery1 + 1) * (maximum_battery2 + 1) * (maximum_Mainprice) \
    + b1 * (maximum_battery2 + 1) * (maximum_Mainprice) \
    + b2 * maximum_Mainprice + priceMain - 1
    return rank




#probabilityMatrix = np.zeros(horizons+1, No_of_states, len(completeActionSet1), len(completeActionSet2), No_of_states)
#def getWinner(h,d,b1,b2,priceCurrent, purchase1,price_Quote1, purchase2, price_Quote2):
def getWinner(h, current_state_index, action1_index, action2_index):

    d,b1,b2,priceCurrent = getStateFromIndex(current_state_index)
    purchase1, price_Quote1 = getAction1fromIndex(action1_index)
    purchase2, price_Quote2 = getAction2fromIndex(action2_index)
    if ((b1 + maxPurchase1) >= d):
        haveSufficientBattery1 = True
    else:
        haveSufficientBattery1 = False

    if ((b2 + maxPurchase2) >= d):
        haveSufficientBattery2 = True
    else:
        haveSufficientBattery2 = False
    # Decide bid winner
    I1 = 0
    I2 = 0
    # case 1
    if (haveSufficientBattery1 and haveSufficientBattery2):
        # print("case1")
        if (price_Quote1 < price_Quote2):
            return 1
        elif (price_Quote1 > price_Quote2):
            return 2
        else:
            return 0


    # case 2
    elif (haveSufficientBattery1 and not (haveSufficientBattery2)):
        # print("case2")
        return 1

    # case 3
    elif (not (haveSufficientBattery1) and haveSufficientBattery2):
        # print("case3")
        return 2

    # case 4
    else:  # both players cannot win
        # print("case4")
        return 0
#def getNextBattery(h,d,b1,b2,priceCurrent, purchase1,price_Quote1, purchase2, price_Quote2)


#def getTransitionProb(h,d,b1,b2,priceCurrent, purchase1,price_Quote1, purchase2, price_Quote2): # horizon, state, action1, action2
def getTransitionProb(h, current_state_index, action1_index, action2_index):
    probOverStates = np.zeros(No_of_states)

    if h == horizons:
        for s in range(No_of_states):
            if s == current_state_index:
                probOverStates[s] = 1
                return probOverStates

    purchase1, price_Quote1 = getAction1fromIndex(action1_index)
    purchase2, price_Quote2 = getAction2fromIndex(action2_index)
    d, b1, b2, priceCurrent = getStateFromIndex(current_state_index)


    winner = getWinner(h, current_state_index, action1_index, action2_index)
    #print("winner",winner)

    if winner == 0:
        I1 = 0
        I2 = 0
    elif winner == 1:
        I1 = 1
        I2 = 0
    else: #winner == 2:
        I1 = 0
        I2 = 1

    #print("I1,I2",I1,I2)
    # r1 = 0
    # r2 = 0
    #print("battery1 Started")
    # Be careful about b1 and b1Next





    #print("batteries",b1Next,b2Next)
    #print(b1,b2)
    #print(I1, I2)
    for dNext in demand_array:
        for priceNext in price_array_Main:
            for renew1 in renewal_array1:
                for renew2 in renewal_array2:

                    b1Next = b1
                    b1Next = b1Next + renew1
                    b1Next = b1Next + purchase1
                    b1Next = min(b1Next, maximum_battery1)
                    b1Next = b1Next - d * I1  # see how price quote is influencing next state
                    b1Next = max(0, b1Next)



                    b2Next = b2
                    b2Next = b2Next + renew2
                    b2Next = b2Next + purchase2
                    b2Next = min(b2Next, maximum_battery2)
                    b2Next = b2Next - d * I2
                    b2Next = max(0, b2Next)


                    sNext = getStateIndex(dNext,b1Next, b2Next, priceNext)
                    demandProb = demandTPM[d-1, dNext -1]
                    priceProb = priceTPM_Main[priceCurrent-1, priceNext-1]
                    renProb1 = renewalProb1[renew1-1]
                    renProb2 = renewalProb2[renew2-1]
                    prob = demandProb * priceProb * renProb1 * renProb2
                    probOverStates[sNext] += prob

    #print("Probability sum ", np.sum(probOverStates))
    return probOverStates

def getCost(h, current_state_index, action1_index, action2_index):

    d, b1, b2, priceMain = getStateFromIndex(current_state_index)
    purchase1, price_Quote1 = getAction1fromIndex(action1_index)
    purchase2, price_Quote2 = getAction2fromIndex(action2_index)

    winner = getWinner(h,current_state_index, action1_index, action2_index)

    if winner == 1:
        return (d*price_Quote1 - purchase1*priceMain) * 10
    elif winner == 2:
        return -(d*price_Quote2 - purchase2*priceMain) * 10
    else:
        return 0


def getNextState(h, current_state_index, action1_index, action2_index):
    if h == horizons:
        return current_state_index
    purchase1, quote1 = getAction1fromIndex(action1_index)
    purchase2, quote2 = getAction2fromIndex(action2_index)
    d,b1,b2,price_main_current = getStateFromIndex(current_state_index)

    I1 = 0
    I2 = 0

    winner = getWinner(h,current_state_index, action1_index, action2_index)
    if winner == 1:
        I1 = 1
    elif winner == 2:
        I2 = 1
    else:
        I1 = 0
        I2 = 0


    dNext = getNextDemand(d)

    r1 = getNextRenewal1()
    r2 = getNextRenewal2()
    # r1 = 0
    # r2 = 0

    b1Next = b1
    b1Next = b1Next + r1
    b1Next = b1Next + purchase1
    b1Next = min(b1Next, maximum_battery1)
    b1Next = b1Next - d * I1   # see how price quote is influencing next state
    b1Next = max(0,b1Next)

    #r1Next = getNextRenewal1(r1)

    b2Next = b2
    b2Next = b2Next + r2
    b2Next = b2Next + purchase2
    b2Next = min(b2Next, maximum_battery2)
    b2Next = b2Next - d * I2
    b2Next = max(0, b2Next)

    #r2Next = getNextRenewal2(r2)

    priceMainNext = getNextPriceMain(price_main_current)

    nextStateIndex = getStateIndex(dNext, b1Next, b2Next, priceMainNext)

    return nextStateIndex



prob_Matrix_Smart_Grid = np.zeros([horizons+1,No_of_states, len(completeActionSet1), len(completeActionSet2), No_of_states])
costMatrix_Smart_Grid = np.zeros([horizons+1,No_of_states, len(completeActionSet1), len(completeActionSet2)])
for h in range(horizons + 1):
    for currentStateIndex in range(No_of_states):
        for actIndex1 in range(len(completeActionSet1)):
            for actIndex2 in range(len(completeActionSet2)):
                currentState = completeStateSet[currentStateIndex]
                action1 = completeActionSet1[actIndex1]
                action2 = completeActionSet2[actIndex2]
                prob_Matrix_Smart_Grid[h, currentStateIndex, actIndex1, actIndex2, :] = \
                    getTransitionProb(h, currentStateIndex, actIndex1, actIndex2)
                costMatrix_Smart_Grid[h, currentStateIndex, actIndex1, actIndex2] = \
                    getCost(h, currentStateIndex, actIndex1, actIndex2)
                #print(costMatrix_Smart_Grid[h, currentStateIndex, actIndex1, actIndex2])






smartGridHorizons = horizons

horizons = smartGridHorizons     #horizons 0,1,2, .. ,horizons ,( horizons+1 stages in total)
states = len(completeStateSet)
actions = len(completeActionSet1)
discount = 1
max_iterations = 300000 # 100000


# #Preparing Prob Matrix
# Probbase = np.random.rand(horizons+1, states, actions , actions, states)
# P = np.apply_along_axis(lambda x: x/np.sum(x), 4, Probbase)
# for sCurrent in range(states):
#     for a1 in range(actions):
#         for a2 in range(actions):
#             for sNext in range(states):
#                 if sCurrent == sNext:
#                     P[horizons, sCurrent, a1, a2, sNext] = 1
#                 else:
#                     P[horizons, sCurrent, a1, a2, sNext] = 0


#Preparing R
# R_Single = np.random.random((states, actions, actions)) * 10
# R = np.repeat(R_Single[np.newaxis, :, :, :], horizons+1, axis=0)
# R_N = np.ones([actions,actions])
#
# for s in range(states):
#     R[horizons, s] = (s+1)*R_N
# print("Probmatrix", prob_Matrix_Smart_Grid.shape)
# print("costMatrix", costMatrix_Smart_Grid.shape)
# print("R",R.shape)
# print("P",P.shape)

P = prob_Matrix_Smart_Grid
R = costMatrix_Smart_Grid

np.save("ProbabilityMatrix{}.npy".format(file_number),P)
np.save("RewardMatrix{}.npy".format(file_number),R)

print(np.sum(R))
print(np.sum(costMatrix_Smart_Grid))

#P = np.zeros([horizons+1, states, actions, actions, states])
#R = np.random.random((horizons+1, states, actions, actions))

def getDeterministicPolicy(index):
    pol = np.array([0]*actions)
    pol[index] = 1
    return pol

def getmixedPolicy():
    arr = np.random.rand(actions)
    arr = arr / arr.sum()
    return arr


value_DP = np.zeros((horizons+1,states))  # Initial value
while True:
    Q = np.zeros((horizons+1, states, actions, actions))
    for h in range(horizons+1):
        for a1 in range(actions):
            for a2 in range(actions):
                if(h == horizons): #terminal stage
                    Q[h, :, a1, a2] = R[h, :, a1, a2]
                else:
                    Q[h, :, a1, a2] = R[h, :, a1, a2] + discount * P[h, :,  a1, a2].dot(value_DP[h+1])



    v_prev = deepcopy(value_DP)
    #print(v_prev)

    for h in range(horizons+1):
        for s in range(states):
            pol, value_DP[h,s] = solve_zero_sum_game(Q[h, s, :, :])

    #print("Value Error", np.linalg.norm(V-v_prev))
    if np.linalg.norm(value_DP-v_prev) < 0.000001:
        break

#Value iteration done
for hh in range(horizons + 1):
    print(np.sum(Q[hh]), np.sum(R[hh]))

print(np.sum(Q[hh]))

#print("V= \n", V)




def stepSize(n):
    return 1 / math.ceil((n + 1) / 10)


Q = np.random.rand(horizons+1, states, actions, actions)
#Setting last stage to terminal cost.
Q[horizons] = R[horizons]



#print("Q=\n", Q)
state = np.random.randint(0, states)
tot_count = np.zeros((horizons + 1, states, actions, actions, states))
print("R",np.sum(R))
print("Q", np.sum(Q))

#for h in reversed(range(horizons + 1)):
total_avgs = 1
error = np.zeros([total_avgs,max_iterations])

for avgIndex in range(total_avgs):
    h = 0
    for n in range(max_iterations):
        if n % 10000 == 0:
            print("Iteration ", n, flush=True)
            #be careful with h=horizon case
        if(h>=horizons):
            h = np.random.randint(0, horizons+1)
            state = np.random.randint(0, states)

        act1 = np.random.randint(0, actions) # numpy has same function , don't confuse
        act2 = np.random.randint(0, actions)

        s_new = int(np.random.choice(np.arange(states), 1, p=P[h,state,act1,act2,:]))

        r = R[h][state][act1][act2]

                # print(Q[s_new,:,:])

        tot_count[h][state][act1][act2][s_new] += 1
        if (h!=horizons):

            pol, next_state_value = solve_zero_sum_game(Q[h+1, s_new, :, :])

                    #Q update
                    #print(np.sum(tot_count[h][state][act1][act2]))
            step = stepSize(np.sum(tot_count[h][state][act1][act2]))
            Q[h, state, act1, act2] = (1-step) * Q[h, state, act1, act2] + step * (r + next_state_value)
                    #print("hihi Q")
                    #print(np.sum(Q))

        h = h + 1
        state = s_new

        sor_minimax_Q = np.zeros([horizons+1, states])

        for h in range(horizons+1):
            for i in range(states):
                pol, sor_minimax_Q[h, i] = solve_zero_sum_game(Q[h, i, :, :])

        error[avgIndex, n] = np.linalg.norm(value_DP - sor_minimax_Q)

                #print("error for {} time".format(avgIndex) , error[avgIndex,n])
print("Q", np.sum(Q))
np.save('errorSmart{}'.format(file_number), error)
np.save('optimalQMatrix{}.npy'.format(file_number), Q )
np.save('optimalMinimaxValue{}.npy'.format(file_number), sor_minimax_Q)

plot1 = plt.figure(1)
plt.plot(np.sum(error, axis=0))
plt.yscale("log")
plt.title("Error over iterations for N={} S={} A={}".format(horizons,states,actions))
plt.xlabel("Number of iterations")
plt.ylabel("Error")
plt.savefig("SmartgridError{}.png".format(file_number))

#plt.show()


policyMatrixLearnedMaxPlayer = [np.arange(actions), np.arange(actions), np.arange(actions)]
policyMatrixLearnedMaxPlayer.clear()
policyMatrixLearnedMinPlayer = [np.arange(actions), np.arange(actions), np.arange(actions)]
policyMatrixLearnedMinPlayer.clear()
valueMatrixLearned = np.zeros((horizons+1, states))


for h in range(horizons + 1):
    for s in range(states):

        policy_for_maxPlayer , minmax_for_maxPlayer  = solve_zero_sum_game(Q[h, s, :, :])
        policy_for_minPlayer , minmax_for_minPlayer = solve_zero_sum_game(-Q[h, s, :, :].transpose())

        valueMatrixLearned[h,s] = minmax_for_maxPlayer

        policyMatrixLearnedMaxPlayer.append(policy_for_maxPlayer)

        policyMatrixLearnedMinPlayer.append(policy_for_minPlayer)

np.save('Policy_Matrix_Max{}.npy'.format(file_number),policyMatrixLearnedMaxPlayer)
np.save('Policy_Matrix_Min{}.npy'.format(file_number),policyMatrixLearnedMinPlayer)

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
print("average Cost Calculation")
for h in range(horizons+1):
    for s in range(states):
        avgCost_Qlearning[h,s] = total_Cost[h,s]/hit_Count[h,s]


print("AVERAGE COST CALCULATION USING when both player deviates ")

total_Cost = np.zeros([horizons+1, states])
hit_Count = np.zeros([horizons+1, states])

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
        #polMax = policyMatrixLearnedMaxPlayer[h*states + state]
        #polMin = policyMatrixLearnedMinPlayer[h*states + state]

        pos = getActionIndex1(d, minprice1)
        polMax = getDeterministicPolicy(pos)
        polMin = getDeterministicPolicy(pos)

        act1 = int(np.random.choice(np.arange(actions), 1, p=polMax))
        act2 = int(np.random.choice(np.arange(actions), 1, p=polMin))

        s_new = int(np.random.choice(np.arange(states), 1, p=P[h, state, act1, act2, :]))

        r = R[h, state, act1, act2]

        episode_sum += r

        state = s_new

        h = h + 1

    total_Cost[episode_h, episode_state] += episode_sum


avgCost_both_dev = np.zeros([horizons+1, states])
print("average Cost Calculation")
for h in range(horizons+1):
    for s in range(states):
        avgCost_both_dev[h,s] = total_Cost[h,s]/hit_Count[h,s]

print("AVERAGE COST CALCULATION USING Baseline Policy when MAX player deviates to a random policy")

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
print("average Cost Calculation")
for h in range(horizons + 1):
    for s in range(states):
        avgCost_Max_deviates[h, s] = total_Cost[h, s] / hit_Count[h, s]

print("AVERAGE COST CALCULATION USING Baseline Policy when MIN player deviates to a random policy")

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
        #polMin = policyMatrixLearnedMinPlayer[h*states + state]
        polMin = getmixedPolicy()
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
print("average Cost Calculation")
for h in range(horizons+1):
    for s in range(states):
        avgCost_Min_Deviates[h,s] = total_Cost[h,s]/hit_Count[h,s]



print("Value")
print(value_DP)
print("minimax Q")
print(sor_minimax_Q)
print("Average Cost Q learning ")
print(avgCost_Qlearning)
print("Average Cost MAX player deviates to discrete uniform ")
print(avgCost_Max_deviates)
print("Average Cost MIN player deviates to discrete uniform ")
print(avgCost_Min_Deviates)

difference_MAX = avgCost_Max_deviates - avgCost_Qlearning
difference_MIN = avgCost_Min_Deviates - avgCost_Qlearning
# for h in range(horizons+1):
#     for s in range(states):
#         print("{} {} ".format(h,s), "{:0.2f}".format(V[h,s]), "{:0.2f}".format(sor_minimax_Q[h,s]), "{:0.2f}".format(avgCost_Qlearning[h,s]),\
#               "{:0.2f}".format(avgCost_Max_deviates[h,s]), "{:0.2f}".format(difference_MAX[h,s]), \
#               "{:0.2f}".format(avgCost_Min_Deviates[h,s]), "{:0.2f}".format(difference_MIN[h,s]))

np.save('avg_Q_smart{}.npy'.format(file_number), avgCost_Qlearning)
np.save('minmax_Q_smart{}.npy'.format(file_number), sor_minimax_Q)
np.save('value_DP_smart{}.npy'.format(file_number), value_DP)
np.save('MAX_dev_smart{}.npy'.format(file_number), avgCost_Max_deviates)
np.save('MIN_dev_smart{}.npy'.format(file_number), avgCost_Min_Deviates)
np.save('both_dev{}.npy'.format(file_number), avgCost_both_dev)

print('That took {} seconds'.format(time.time() - starttime))



























