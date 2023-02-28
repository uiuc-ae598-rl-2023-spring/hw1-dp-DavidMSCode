import matplotlib.pyplot as plt
import numpy as np
import random

def greedyAction(Q,s):
    """
    Chooses the most optimal action based on current state and the estimated q state-action value
    Args:
        Q (numpy.array): a 2D array of state action pair values
        s (int): the current state
    Returns:
        a (int): The action to be taken
    """
    a = np.argmax(Q[s])
    return a

def epsilonGreedy(env,Q,s,epsi):
    """
    Chooses an action optimally with a chance to explore instead
    Args:
        env (GridWorld): The MDP environment object provided by gridworld.py
        Q (numpy.array): a 2D array of state action pair values
        s (int): the current state
        epsi (float): The chance (0-1) of exploring rather than taking the greedy action
    Returns:
        a (int): The action to be taken
    """
    nA = env.num_actions    #number of possible actions
    p_greedy = 1-epsi       #chance of choosing greedy action. The epsilon/|A(s)| term is accounted for in the random selection.
    roll = random.random()  #generate a random float between 0 and 1
    if roll <= p_greedy:
        a = greedyAction(Q,s)   #if roll below random threshhold choose the greedy action
    else:
        a = int(np.floor((roll-p_greedy)/(epsi/nA)))     #Splits remaining chance between all actions including the greedy action
    return a


def QtoPolicy(env,Q):
    policy = init_policy(env)
    states,actions = getStatesAndActions(env)
    for s in states:
        old_qmax = None
        best_actions = []
        for a in actions:
            if old_qmax is None:
                best_actions = [a]
                old_qmax = Q[s,a]
            elif Q[s,a] > old_qmax:
                best_actions = [a]
                old_qmax = Q[s,a]
            elif Q[s,a] == old_qmax:
                best_actions.append(a)

        new_p = 1/len(best_actions)
        policy[s] = [new_p if a in best_actions else 0 for a in actions]
    
    return policy
        

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


def TD0ValueEstimate(env,policy,alf,discount,max_episodes):
    """Estimates the value function given a policy"""
    V = init_V(env)
    for i in np.arange(0,max_episodes):
        s = env.reset()
        done = False
        while not done:
            a = np.argmax(policy[s])
            (s1, r, done) = env.step(a)                     #transition to next state
            V[s] = V[s] + alf*(r + discount*V[s1]-V[s])  #update Q with SARSA
            s=s1
    return V

def runPendulumEpisode(env,policy):
    """
    Args:
        env (environment object): discrete pendulum MDP object
        policy (numpy array): 2D numpy array that gives the probability of choosing an action a at each state, policy[s,a]
    Returns:
        log (dict): Log of episode steps, states, actions, rewards and the internal pendulum angles and velocities
    """
    # Initialize simulation
    s = env.reset()
    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
        'theta': [env.x[0]],        # agent does not have access to this, but helpful for display
        'thetadot': [env.x[1]],     # agent does not have access to this, but helpful for display
    }
    # Simulate until episode is done
    done = False
    while not done:
        a = np.argmax(policy[s])
        (s, r, done) = env.step(a)
        #store simulation info
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
        log['theta'].append(env.x[0])
        log['thetadot'].append(env.x[1])
    return log

def runGridworldEpisode(env,policy):
    """
    Simulates and episode of the gridworld when given a policy
    Args:
        env (environment object): gridworld MDP object
        policy (numpy array): 2D numpy array that gives the probability of choosing an action a at each state, policy[s,a]
    Returns:
        log (dict): Log of episode steps, states, actions & rewards
    """
    # Initialize simulation
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
        #store simulation info
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
    return log

def plotLog(log,title="Title"):
    """
    Plots the states, actions and rewards for Qlearning and SARSA. Also the angle and velocity for the discretized pendulum
    """
    n_plots = 2 if 'theta' in log and 'thetadot' in log else 1
    fig = plt.figure(figsize=(10, 6*n_plots))
    ax = fig.add_subplot(n_plots,1,1)
    fig.suptitle(title)
    ax.plot(log['t'], log['s'])
    ax.plot(log['t'][:-1], log['a'])
    ax.plot(log['t'][:-1], log['r'])
    ax.legend(['s', 'a', 'r'])
    ax.grid()
    if 'theta' in log and 'thetadot' in log:
        ax = fig.add_subplot(n_plots,1,2)
        ax.plot(log['t'], log['theta'])
        ax.plot(log['t'], log['thetadot'])
        ax.legend(['theta', 'thetadot'])
        ax.grid()

def PlotLearningCurve(logs,title="Title"):
    """Plots learning curve for SARSA and Q Learning from logs"""
    fig = plt.figure()
    max_episodes = len(logs)
    Val = np.zeros(max_episodes)
    for i in range(0,max_episodes):
        Val[i] = sum(logs[i]["r"])
    plt.plot(range(0,max_episodes),Val)
    plt.xlabel("Episodes")
    plt.ylabel("Episode Return")
    plt.title(title)
    ax = plt.gca()
    if 'theta' in logs[0]:
        ax.set_ylim([-10, 100])
    plt.grid()
    plt.show()

def plotMeanV(meanV,title="Title"):
    """
    Plots learning curve for policy iteration and value iteration algorithms 
    """
    fig = plt.figure()
    plt.plot(np.arange(0,len(meanV)),meanV)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Value Function")
    plt.title(title)
    plt.grid()
    plt.show()

def plot_gridworld(V,policy,shape,title="Title",hide_values=False,plot_arrows=True):
    """
    Args:
        V (numpy array): 1D numpy array showing the value function for each state s, V[s]
        policy (numpy array): 2D numpy array that gives the probability of choosing an action a at each state, policy[s,a]
        shape (tuple(int)): The shape of the gridworld
    Ouput:
        A plot showing the value function and policy decisions in a gridworld
        """
    n_plots = 2 if plot_arrows else 1
    if not V is None:
        #Reshape the V and policy into grids for easy plotting
        V_grid = V.reshape(shape)
        shape = V_grid.shape
        
        #Plot value estimate as an image using plt.imshow()
        fig = plt.figure()
        ax = fig.add_subplot(1,n_plots,1)
        fig.suptitle(title)
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
        if not hide_values:
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
        else:
            plt.colorbar()
    if not policy is None and  plot_arrows:
        policy_grid = policy.reshape(shape+(-1,))        
        #Plot policy directions
        ax = fig.add_subplot(1,n_plots,2)
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