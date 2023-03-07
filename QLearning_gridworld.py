# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from RLSharedFunctions import init_Q, epsilonGreedy, greedyAction, init_V, QtoPolicy, plot_gridworld, runGridworldEpisode, plotLog, PlotLearningCurve

def TD0ValueEstimate(env,policy,alf,discount,max_episodes):
    V = init_V(env)
    for i in np.arange(0,max_episodes):
        s = env.reset()
        done = False
        while not done:
            a = np.argmax(policy[s])
            (s1, r, done) = env.step(a)                     #transition to next state
            V[s] = V[s] + alf*(r + discount*V[s1]-V[s])  #update V with TD
            s=s1
    return V

def QLearning_episode(env,Q,alf,epsi,discount):
    """
    Simulates 1 episode of the gridworld while running Q Learning
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
    #run QL until episode ends
    while not done:
        a = epsilonGreedy(env,Q,s,epsi)                         #choose the next action
        (s1, r, done) = env.step(a)                             #transition to next state
        Q[s,a] = Q[s,a]+alf*(r + discount*max(Q[s1])-Q[s,a])    #update Q with Q Learning
        s = s1                                                  #make next state the current state
        # Store current step in log
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
    return (Q,log)

def QLearning(env, alf, epsi, discount,max_episodes=5000):
    """
    Trains Q for gridworld using QLearning
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
        #Simulate Gridworld over the max episodes while running Q Learning
        Q, log = QLearning_episode(env,Q,alf,epsi,discount)
        #store logs in list
        logs.append(log)
    return (Q,logs)


def QLGridworld(alf=0.5,epsi=0.1,discount=0.95,max_episodes = 3000):
    # Create environment
    env = gridworld.GridWorld(hard_version=False)
    Q, logs = QLearning(env,alf,epsi,discount, max_episodes)
    policy = QtoPolicy(env,Q)
    V = TD0ValueEstimate(env,policy,alf,discount,1000)
    return (env,Q,policy,V,logs)

if __name__ == '__main__':
    alf = 0.5           #learning step size
    epsi = .1           #exploration chance
    discount = 0.95     #future state pair value disc"ount
    env,Q,policy,V,logs = QLGridworld(alf,epsi,discount,3000)
    plot_gridworld(V,policy,(5,-1),title="Q Learning")
    log = runGridworldEpisode(env,policy)
    plotLog(log,title="Q Learning Gridworld Trajectory")
    PlotLearningCurve(logs,title="Q Learning Learning Curve")

# %%
