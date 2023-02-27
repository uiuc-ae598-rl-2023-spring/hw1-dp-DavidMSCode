import matplotlib.pyplot as plt
import numpy as np


def new_policy(env,V,s,discount):
    """
    Calculates what the most optimal determinstic policy should be for the current state given the gridworld's state-value's
    Args:
        env (GridWorld object): The MDP environment object provided by gridworld.py
        V (numpy.array): A 1D numpy array containing a value for each state in the gridworld
        s (int): The current state for policy to be computed
        discount (float): discount factor for the bellman equations from 0-1
    Returns:
        new_policy (numpy.array): 1D array with a length equal to number of possible actions in the current state with the probbaility of taking each action. sum(new_policy)=1
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

def plot_gridworld(V,policy,shape):
    """
    Args:
        V (numpy array): 1D numpy array showing the value function for each state s, V[s]
        policy (numpy array): 2D numpy array that gives the probability of choosing an action a at each state, policy[s,a]
        shape (tuple(int)): The shape of the gridworld
    Ouput:
        A plot showing the value function and policy decisions in a gridworld
        """
    if not V is None:
        #Reshape the V and policy into grids for easy plotting
        V_grid = V.reshape(shape)
        shape = V_grid.shape
        
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
    if not policy is None:
        policy_grid = policy.reshape(shape+(-1,))        
        #Plot policy directions
        plt.subplot(1,2,2)
        ax = plt.gca()
        # Major ticks
        ax.set_xticks(np.arange(0, shape[0], 1))
        ax.set_yticks(np.arange(-shape[1], 0, 1))
        # Labels for major ticks
        ax.set_xticklabels(np.arange(1, shape[0]+1, 1))
        ax.set_yticklabels(np.arange(shape[1], 0, -1))
        # Minor ticks
        ax.set_xticks(np.arange(-.5, shape[0], 1), minor=True)
        ax.set_yticks(np.arange(-shape[1]-.5, .5, 1), minor=True)
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
                plt.arrow(y,-x-1,dx,dy,head_width = .1)
    plt.draw()


def getStatesAndActions(env):
    """
    Returns two 1D arrays containing all states and all actions
    Args:
        env (GridWorld object): The MDP environment object provided by gridworld.py
    Returns:
        states (numpy.array): Int array of all states in the gridworld environment
        actions (numpy.array): Int array of all actions in the gridworld environment
    """
    states = np.arange(0,env.num_states,dtype=int)
    actions = np.arange(0,env.num_actions,dtype=int)
    return (states, actions)

def init_V(env):
    """
    Returns an initialized state-value 1D vector containing all zeros
    Args:
        env (GridWorld object): The MDP environment object provided by gridworld.py
    Returns:
        V (numpy.array): initial state value array
    """
    V = np.zeros(env.num_states,dtype=float)
    return V

def init_policy(env):
    """
    Returns an initialized policy 2D array containing equal probability for choosing each action for each state
    Args:
        env (GridWorld object): The MDP environment object provided by gridworld.py
    Returns:
        policy (numpy.array): Initial policy array
    """
    policy = np.ones([env.num_states,env.num_actions],dtype=float)*1/env.num_actions
    return policy

def init_Q(env):
    """
    Returns an initialized policy 2D array for Q learning and SARSA
    Args:
        env (GridWorld object): The MDP environment object provided by gridworld.py
    Returns:
        policy (numpy.array): Initial policy array
    """
    Q = np.zeros([env.num_states,env.num_actions],dtype=float)*1/env.num_actions
    return Q