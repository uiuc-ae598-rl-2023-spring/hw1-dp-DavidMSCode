# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from RLSharedFunctions import init_Q, epsilonGreedy, init_V, QtoPolicy, plot_gridworld, plotLog, runGridworldEpisode, PlotLearningCurve, TD0ValueEstimate


def SARSA_episode(env,Q,alf,epsi,discount):
    """
    Simulates 1 episode of the gridworld while running SARSA
    Args:
        env (GridWorld): The MDP environment object provided by gridworld.py
        Q (numpy.array): a 2D array of state action pair values
        alf (float): Step size (0,1] for adjusting Q
        epsi (float): The chance (0-1) of exploring rather than taking the greedy action
        discount (float): The discount for future step values to ensure convergance
    Returns:
        Q (numpy.array): The modified state-action pair values at the end of the episode
        log(dict): a dict containing the episode states, actions and rewards.
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
    a = epsilonGreedy(env,Q,s,epsi)                     #choose the first action
    #run SARSA until episode ends
    while not done:
        (s1, r, done) = env.step(a)                     #transition to next state
        a1 = epsilonGreedy(env,Q,s1,epsi)               #choose next action
        Q[s,a] = Q[s,a]+alf*(r + discount*Q[s1,a1]-Q[s,a])  #update Q with SARSA
        s = s1                                          #make next state the current state
        a = a1                                          #make next action the current action
        # Store current step in log
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
    return (Q,log)

def SARSA(env, alf, epsi, discount,max_episodes):
    """
    Chooses an action optimally with a small chance of choosing a random action
    Args:
        env (GridWorld): The MDP environment object provided by gridworld.py
        alf (float): Step size (0,1] for adjusting Q
        epsi (float): The chance (0-1) of exploring rather than taking the greedy action
        discount (float): The discount for future step values to ensure convergance
        max_episodes (int): The number of episodes to be simulated for learning
    Returns:
        Q (numpy.array): The trained state-action pair values
        log(dict): a dict containing the episode states, actions and rewards.
    """
    Q = init_Q(env)                         #Initialize the state-action pair values 
    logs = []                               #init empty log list
    for  i in np.arange(0,max_episodes):
        #Simulate Gridworld over the max episodes while running SARSA
        Q, log = SARSA_episode(env,Q,alf,epsi,discount)
        #store logs in list
        logs.append(log)
    return (Q,logs)


def SARSAGridworld(alf=0.5,epsi=0.1,discount=0.95,max_episodes=5000):
    # Create environment
    env = gridworld.GridWorld(hard_version=False)


    Q, logs = SARSA(env,alf,epsi,discount,max_episodes)
    policy = QtoPolicy(env,Q)
    V = TD0ValueEstimate(env,policy,alf,discount,1000)
    return (env,Q,policy,V,logs)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    figsubdirs = "figures/gridworld/"
    alf = 0.5           #learning step size
    epsi = .1           #exploration chance
    discount = 0.95     #future state pair value disc"ount
    env,Q,policy,V,logs = SARSAGridworld(alf,epsi,discount,3000)
    plot_gridworld(V,policy,(5,-1),title="SARSA Gridworld")
    # plt.savefig(figsubdirs+"Test SARSA Gridworld Learned Value and Policy",bbox_inches='tight',dpi=200)
    log = runGridworldEpisode(env,policy)
    plotLog(log,title="SARSA Gridworld Trajectory")
    PlotLearningCurve(logs,title="SARSA Learning Curve")
    plt.show()

# %%
