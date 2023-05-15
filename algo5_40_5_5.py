import numpy as np
from copy import deepcopy
import math
import time
import matplotlib.pyplot as plt
from lp_solve import  *
starttime = time.time()

#np.random.seed(100)

file_number = 14
horizons = 5 #horizons 0,1,2, .. ,horizons ,( horizons+1 stages in total)
states = 40
actions = 5
discount = 1
max_iterations = 100000# 100000


def getDeterministicPolicy(index):
    pol = np.array([0]*actions)
    pol[index] = 1
    return pol

def getmixedPolicy():
    arr = np.random.rand(actions)
    arr = arr / arr.sum()
    return arr


#Preparing Prob Matrix
Probbase = np.random.rand(horizons+1, states, actions , actions, states)
P = np.apply_along_axis(lambda x: x/np.sum(x), 4, Probbase)

for sCurrent in range(states):
    for a1 in range(actions):
        for a2 in range(actions):
            for sNext in range(states):
                if sCurrent == sNext:
                    P[horizons, sCurrent, a1, a2, sNext] = 1
                else:
                    P[horizons, sCurrent, a1, a2, sNext] = 0






#P = np.zeros([horizons+1, states, actions, actions, states])
#R = np.random.random((horizons+1, states, actions, actions))

R_Single = np.random.random((states, actions, actions)) * 10
R = np.repeat(R_Single[np.newaxis, :, :, :], horizons+1, axis=0)
R_N = np.ones([actions,actions])

for s in range(states):
    R[horizons, s] = (s+1)*R_N



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
            #print(n)
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

np.save('optimalQMatrix{}.npy'.format(file_number), Q )
avgError = np.sum(error, axis=0)/total_avgs

np.save('error{}'.format(file_number), error)

plot1 = plt.figure(1)
plt.plot(avgError)
#plt.plot(np.sum(error, axis=0))
plt.yscale("log")
plt.title("Error over iterations for H={} |S|={} |A|={}".format(horizons,states,actions))
plt.xlabel("Number of iterations")
plt.ylabel("Error")
plt.savefig("algorithm{}.png".format(file_number))

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



print("AVERAGE COST CALCULATION USING Baseline Policy when MAX player deviates to a deterministic policy")

total_Cost = np.zeros([horizons + 1, states])
hit_Count = np.zeros([horizons + 1, states])

max_iterations = 100000
# print(horizons,states)
for n in range(max_iterations):
    # change of episode
    h = np.random.randint(0, horizons + 1)  # exclusive of upperbound
    state = np.random.randint(0, states)  # exclusive of upperbound

    episode_sum = 0
    episode_h = h
    episode_state = state

    hit_Count[h, state] += 1

    while (h <= horizons):
        #polMax = policyMatrixLearnedMaxPlayer[h * states + state]
        polMin = policyMatrixLearnedMinPlayer[h * states + state]

        #polMax = [(1 / actions)] * actions
        #pos = np.random.randint(0, actions)
        pos = state%3
        polMax = getDeterministicPolicy(pos)
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

print("AVERAGE COST CALCULATION USING Baseline Policy when MIN player deviates to a deterministic policy")

total_Cost = np.zeros([horizons+1 ,states])
hit_Count = np.zeros([horizons+1 , states])

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
        #polMin = policyMatrixLearnedMinPlayer[h*states + state]

        #pos = np.random.randint(0, actions)
        pos = state % 3
        polMin = getDeterministicPolicy(pos)

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

np.save('avg_Q_algo{}.npy'.format(file_number), avgCost_Qlearning)
np.save('minmax_Q_algo{}.npy'.format(file_number), sor_minimax_Q)
np.save('value_algo{}.npy'.format(file_number), value_DP)
np.save('MAX_dev_algo{}.npy'.format(file_number), avgCost_Max_deviates)
np.save('MIN_dev_algo{}.npy'.format(file_number), avgCost_Min_Deviates)

print('That took {} seconds'.format(time.time() - starttime))
