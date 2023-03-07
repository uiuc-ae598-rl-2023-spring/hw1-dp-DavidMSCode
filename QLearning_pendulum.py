# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import discrete_pendulum
from RLSharedFunctions import init_Q, epsilonGreedy, init_V, QtoPolicy, plot_gridworld, runPendulumEpisode, plotLog, PlotLearningCurve

def TD0ValueEstimate(env,policy,alf,discount,max_episodes):
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

def QLearning_episode(env,Q,alf,epsi,discount):
    """
    Simulates 1 episode of the discretized pendulum while running SARSA
    Args:
        env (environment): The MDP environment object provided by pendulum.py
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
        'theta': [env.x[0]],        # agent does not have access to this, but helpful for display
        'thetadot': [env.x[1]],     # agent does not have access to this, but helpful for display
    }
    # Simulate until episode is done
    done = False
    #run QL until episode ends
    while not done:
        a = epsilonGreedy(env,Q,s,epsi)                         #choose the next action
        (s1, r, done) = env.step(a)                             #transition to next state
        Q[s,a] = Q[s,a]+alf*(r + discount*max(Q[s1])-Q[s,a])    #update Q with Q Learning
        s = s1                                        #make next action the current action
        # Store current step in log
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
        log['theta'].append(env.x[0])
        log['thetadot'].append(env.x[1])
    return (Q,log)

def QLearning(env, alf, epsi0, discount,max_episodes=5000):
    """
    Args:
        env (env object): The MDP environment object provided by gridworld.py or discrete_pendulum.py
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
        # epsi  = (1-i/max_episodes)*epsi0  #time dependent exploration
        epsi = epsi0                        #constant exploration
        #Simulate Gridworld over the max episodes while running SARSA
        Q, log = QLearning_episode(env,Q,alf,epsi,discount)
        #store logs in list
        logs.append(log)
    return (Q,logs)


def QLearningPendulum(n_theta=31,n_theta_dot=31,alf=0.15,epsi=0.8,discount=0.95,max_episodes=2000):
    """
    Trains a policy for balancing a discretized pendulum using the SARSA algorithm 
    """
    env = discrete_pendulum.Pendulum(n_theta,n_theta_dot,31)
    #Train with SARSA
    Q, logs = QLearning(env,alf,epsi,discount,max_episodes)
    #Derive policy from trained Q
    policy = QtoPolicy(env,Q)
    #Use TD(0) to calculate V
    V = TD0ValueEstimate(env,policy,alf,discount,1000)
    
    return (env,Q,policy,V,logs)

if __name__ == '__main__':
    n_theta = 31
    n_theta_dot = 31
    (env,Q,policy,V,logs) = QLearningPendulum(n_theta,n_theta_dot,alf=0.15,epsi=0.8,discount=0.95,max_episodes=2000)
    plot_gridworld(V,policy,(n_theta,n_theta_dot),title="Q Learning Pendulum Value Function",hide_values=True,plot_arrows=False)
    log = runPendulumEpisode(env,policy)
    plotLog(log,title="Q Learning Trained Agent Trajectory")
    PlotLearningCurve(logs,title="Q Learning Learning Curve")
# %%
