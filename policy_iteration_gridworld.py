# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
import copy


def plot_gridworld(V,policy,shape):
    """
    Args:
        V (numpy array): 1D numpy array showing the value function for each state s, V[s]
        policy (numpy array): 2D numpy array that gives the probability of choosing an action a at each state, policy[s,a]
        shape (tuple(int)): The shape of the gridworld
    Ouput:
        A plot showing the value function and policy decisions in a gridworld
        """
    #Reshape the V and policy into grids for easy plotting
    V_grid = V.reshape(shape)
    shape = V_grid.shape
    policy_grid = policy.reshape(shape+(-1,))
    #Plot value estimate as an image using plt.imshow()
    plt.figure()    
    plt.subplot(1,2,1)
    plt.imshow(V_grid,cmap='hot')
    ax = plt.gca()
    # Major ticks
    ax.set_xticks(np.arange(0, shape[0], 1))
    ax.set_yticks(np.arange(0, shape[1], 1))
    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, shape[0]+1, 1))
    ax.set_yticklabels(np.arange(1, shape[1]+1, 1))
    # Minor ticks
    ax.set_xticks(np.arange(-.5, shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-.5, shape[1], 1), minor=True)
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)
    #plot value text in grid
    Vrange = max(V)-min(V)
    minV = min(V)
    for (x,y), val in np.ndenumerate(V_grid):
        #change font color if background too dark
        c="black" if val-minV>Vrange/10 else "white"
        #plot value at grid location
        ax.text(y,x,"%.2f"%val,
                color=c,
                horizontalalignment="center",
                verticalalignment="center")
        
    #Plot policy directions
    plt.subplot(1,2,2)
    ax = plt.gca()
    # Major ticks
    ax.set_xticks(np.arange(0, shape[0], 1))
    ax.set_yticks(np.arange(0, shape[1], 1))
    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, shape[0]+1, 1))
    ax.set_yticklabels(np.arange(1, shape[1]+1, 1))
    # Minor ticks
    ax.set_xticks(np.arange(-.5, shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-.5, shape[1], 1), minor=True)
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='b', linestyle='-', linewidth=2)
    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)
    #plot arrows for each action with any probability
    for (x,y,a), prob in np.ndenumerate(policy_grid):
        if a==0:
            #point right
            dx=0.1
            dy=0
        elif a==1:
            #point up
            dx = 0
            dy = 0.1
        elif a==2:
            #point left
            dx = -0.1
            dy = 0
        else:
            #point down
            dx = 0
            dy = -0.1
        if prob>0:
            #if action has probability of ocurring under policy, plot its arrow
            plt.arrow(y,5-x,dx,dy,head_width = .1)
    plt.draw()


def Bellman(env,policy,V,s,discount):
    """
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
    Args:
        env (GridWorld object): The MDP environment object provided by gridworld.py
        policy(numpy.array): a numpy 2D array where the 1st dimension indicates the state index and the 2nd indicates the action taken. The array stores the probability of taking each action in the corresponding state.
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
    V = np.zeros(env.num_states,dtype=float)

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

def new_policy(env,V,s,discount):
    """
    Args:
    Returns:
    """
    states = np.arange(0,env.num_states)
    actions = np.arange(0,env.num_actions)
    best_actions = []
    old_val = None
    for a in actions:
        val = 0
        for s1 in states:
            val+=env.p(s1,s,a)*(env.r(s,a)+discount*V[s1])
        if old_val is None:
            best_actions = [a]
            old_val = val
        elif val>old_val:
            best_actions = [a]
            old_val=val
        elif val == old_val:
            best_actions.append(a)
    
    #generate new policy from the best action
    new_p = 1/len(best_actions)
    new_policy = [new_p if a in best_actions else 0 for a in actions]
    return new_policy
        

def policy_improvement(env,V,policy,discount):
    """
    Args:
        env
        V
        policy
        discount
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
    policy = np.ones([env.num_states,env.num_actions],dtype=float)*1/env.num_actions
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
