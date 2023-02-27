# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from gridfuncs import init_Q

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

def SARSA(env, alf, epsi, discount,max_episodes=8000):
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
    return logs


def main():
    alf = 0.5           #learning step size
    epsi = .1           #exploration chance
    discount = 0.95     #future state pair value discount
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    logs = SARSA(env,alf,epsi,discount)
    log = logs[-1]
    # Plot data and save to png file
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    plt.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
    plt.savefig('figures/gridworld/test_gridworld.png')


if __name__ == '__main__':
    main()

# %%
