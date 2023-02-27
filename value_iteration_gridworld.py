# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
import copy
from gridfuncs import plot_gridworld, new_policy, getStatesAndActions, init_V, init_policy

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
    return V
                
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

def main():
    #parameters
    discount = 0.95
    tol = 1e-3
    # Create environment
    env = gridworld.GridWorld(hard_version=False)
    V = value_iteration(env,discount,tol)
    policy = calcPolicy(env,V,discount)
    plot_gridworld(V,policy,[5,-1])
    plt.show()

    s = env.reset()
    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }
    # Simulate until episode is done
    done = False
    while not done:
        a = np.argmax(policy[s])
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)

    # Plot data and save to png file
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    plt.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
    plt.savefig('figures/gridworld/test_gridworld.png')


if __name__ == '__main__':
    main()
# %%
