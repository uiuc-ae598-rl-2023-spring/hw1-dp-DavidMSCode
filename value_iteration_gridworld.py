# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
import copy
from RLSharedFunctions import plot_gridworld, new_policy, getStatesAndActions, init_V, init_policy,plotMeanV

def value_iteration(env,discount,tol=1e-3):
    """
    Args:
        env
        discount
        tol
    Returns:
    """
    V = init_V(env)
    states, actions = getStatesAndActions(env)
    delta = np.Inf
    meanV=[0]
    while delta>tol:
        delta = 0
        for s in states:
            v = V[s]
            V_cands = np.zeros(len(actions),dtype=float)
            for a in actions:
                V_cands[a] = 0
                for s1 in states:
                    V_cands[a] += env.p(s1,s,a)*(env.r(s,a)+discount*V[s1])
            V[s] = max(V_cands)
            delta = max(delta,np.abs(V[s]-v))
        meanV.append(sum(V)/25)
    return (V,meanV)
                
def calcPolicy(env,V,discount):
    """
    Calculates the optimal policy based on the current state values
    Args:
    Returns:
        policy (numpy.array)
    """
    #Get all states
    states = getStatesAndActions(env)[0]
    #initialize policy
    policy = init_policy(env)
    for s in states:
        #calculate optimal policy for each state
        policy[s] = new_policy(env,V,s,discount)
    return policy

def VIGridworld(discount = 0.95,tol = 1e-3):
    #parameters
    discount = 0.95
    tol = 1e-3
    # Create environment
    env = gridworld.GridWorld(hard_version=False)
    V,meanV = value_iteration(env,discount,tol)
    policy = calcPolicy(env,V,discount)
    return (env,policy,V,meanV)

if __name__ == '__main__':
    discount = 0.95
    tol = 1e-3
    env, policy, V, meanV = VIGridworld(discount)
    plot_gridworld(V,policy,(5,-1),"Value Iteration")
    plotMeanV(meanV,"Value Iteration Learning Curve")
# %%
