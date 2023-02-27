# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
import copy
from gridfuncs import plot_gridworld, new_policy, init_V, init_policy


def Bellman(env,policy,V,s,discount):
    """
    Uses the Bellman update equation to calculate the current state's value estimate.
    Args:
        env (GridWorld object): The MDP environment object provided by gridworld.py
        policy(numpy.array): a numpy 2D array where the 1st dimension indicates the state index and the 2nd indicates the action taken. The array stores the probability of taking each action in the corresponding state.
        V (numpy array): The current estimate for state-values
        s (int): the current state to be evaluated
        discount (float): discount factor for the bellman equations from 0-1
    Returns:
        The state value estimate for the current state V[s]
    """
    #create state and action list
    states = np.arange(0,env.num_states,dtype=int)
    actions = np.arange(0,env.num_actions,dtype=int)
    V[s] = 0
    for a in actions:
        for s1 in  states:
            r_s_a = env.r(s,a)      #reward of action in state
            p_s1_s_a = env.p(s1,s,a)   #probability of moving into state s1 given current state s and action a
            pi_s_a = policy[s,a]    #probability of choosing the action a in state s
            V[s] += pi_s_a*p_s1_s_a*(r_s_a+discount*V[s1])
    return V[s]

def policy_evaluation(env, policy=None, discount=0.95, tol = 1e-3):
    """
    Uses policy evaluation to iteratively calculate the state-value estimates for the current environment
    Args:
        env (GridWorld object): The MDP environment object provided by gridworld.py
        policy (numpy.array): a numpy 2D array where the 1st dimension indicates the state index and the 2nd indicates the action taken. The array stores the probability of taking each action in the corresponding state.
        discount (float): discount factor for the bellman equations from 0-1
        tol (float): tolerance that maximum error must meet to halt evaluation. tol<<1
    Returns:
        The converged state estimate for the given policy
    """
    if policy is None:
        #generate uniform policy
        policy = np.ones([env.num_states,env.num_actions],dtype=float)*1/env.num_actions
    
    #create state and action list
    states = np.arange(0,env.num_states,dtype=int)
    #initialize state-value estimate
    V = init_V(env)

    #initialize delta to infinity
    delta = np.Inf
    #run policy evaluation until delta is smaller than tolerance
    while delta>tol:
        delta = 0               #maximum per state error
        for s in states:
            v = V[s]            #store previous estimate
            V[s] = Bellman(env,policy,V,s,discount) # calculate next estimate with Bellman eq
            delta = np.max([delta,np.abs(v-V[s])])  # store delta between old and new if largest delta so far
    return V
        

def policy_improvement(env,V,policy,discount):
    """
    Calculates the optimal deterministic policy for the current gridworld state-values. 
    Args:
        env (GridWorld object): The MDP environment object provided by gridworld.py
        V (numpy.array): A 1D numpy array containing a value for each state in the gridworld
        policy (numpy.array): a numpy 2D array where the 1st dimension indicates the state index and the 2nd indicates the action taken. The array stores the probability of taking each action in the corresponding state.
        discount (float): discount factor for the bellman equations from 0-1
    Returns:
        policy(numpy.array): A new policy based on the input value function.a numpy 2D array where the 1st dimension indicates the state index and the 2nd indicates the action taken. The array stores the probability of taking each action in the corresponding state.
        policy_is_stable(bool): A boolean flag that indicates whether the policy was changed (False) or remained the same after improvement (True)
    """
    
    #get actions and states list
    states = np.arange(0,env.num_states,dtype=int)
    policy_is_stable = True
    for s in states:
        old_policy = copy.deepcopy(policy[s])
        policy[s] = new_policy(env,V,s,discount)
        if not np.array_equal(old_policy,policy[s]):
            policy_is_stable = False
    return (policy, policy_is_stable)

def main():
    #parameters
    discount = 0.95
    tol = 1e-3
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    # Initialize simulation
    s = env.reset()
    count = 0
    policy_is_stable = False
    policy = init_policy(env)
    while not policy_is_stable:
        #loop through policy eval and policy improvement until policy stabilizes
        V = policy_evaluation(env, policy, discount, tol)
        # if count==0:
        #     plot_gridworld(V,policy,[5,-1])
        policy, policy_is_stable = policy_improvement(env,V, policy, discount)
        count+=1

    plot_gridworld(V,policy,[5,-1])
    plt.show()
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
