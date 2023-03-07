
# %%
import numpy as np
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

def SARSA_episode(env,Q,alf,epsi,discount):
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
        log['theta'].append(env.x[0])
        log['thetadot'].append(env.x[1])
    return (Q,log)

def SARSA(env, alf, epsi0, discount,max_episodes=5000):
    """
    Chooses an action optimally with a small chance of choosing a random action
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
        Q, log = SARSA_episode(env,Q,alf,epsi,discount)
        #store logs in list
        logs.append(log)
    return (Q,logs)


def SARSAPendulum(n_theta=31,n_theta_dot=31,alf=0.15,epsi=0.8,discount=0.95,max_episodes=2000):
    """
    Trains a policy for balancing a discretized pendulum using the SARSA algorithm 
    """
    env = discrete_pendulum.Pendulum(n_theta,n_theta_dot,31)
    #Train with SARSA
    Q, logs = SARSA(env,alf,epsi,discount,max_episodes)
    #Derive policy from trained Q
    policy = QtoPolicy(env,Q)
    #Use TD(0) to calculate V
    V = TD0ValueEstimate(env,policy,alf,discount,1000)
    
    return (env,Q,policy,V,logs)

def attempt():
    log = runPendulumEpisode(env,policy)
    plotLog(log,title="SARSA Trained Agent Trajectory")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    figsubdirs = "figures/pendulum/"
    n_theta = 15
    n_theta_dot = 21
    alf = 0.2
    epsi = 0.8
    maxEps = 800
    print("Alpha = {:.2f}, Epsilon = {:.2f}".format(alf,epsi))
    (env,Q,policy,V,logs) = SARSAPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=maxEps)
    plot_gridworld(V,policy,(n_theta,n_theta_dot),title="SARSA Pendulum Value Function",hide_values=False,plot_arrows=False,valueFontSize=4)
    plt.savefig(figsubdirs+"Test SARSA Pendulum Learned Value and Policy",bbox='tight',dpi=200)
    PlotLearningCurve(logs,title="SARSA Learning Curve")
    attempt()
    plt.show()
# %%

