# %%
"""This function takes all the model free algorithms and plotting functions and returns the results for the pendulum"""
import numpy as np
from SARSA_pendulum import SARSAPendulum
from QLearning_pendulum import QLearningPendulum
from RLSharedFunctions import plot_gridworld, runPendulumEpisode, plotLog, PlotLearningCurve
alphas = np.linspace(0.05,0.95,5)
epsilons = np.linspace(0.05,0.95,4)
"""SARSA PENDULUM"""
alf = 0.15
epsi = 0.8
n_theta = 31
n_theta_dot = 31
(env,Q,policy,V,logs) = SARSAPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=2000)
plot_gridworld(V,policy,(n_theta,n_theta_dot),title="SARSA Pendulum Value Function",hide_values=True,plot_arrows=False)
log = runPendulumEpisode(env,policy)
plotLog(log,title="SARSA Trained Agent Trajectory")
PlotLearningCurve(logs,title="SARSA Pendulum Learning Curve")
for alf in alphas:
    env,Q,policy,V,logs = SARSAPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=2000)
    PlotLearningCurve(logs,title="SARSA Pendulum Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
alf = 0.15
for epsi in epsilons:
    env,Q,policy,V,logs = SARSAPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=2000)
    PlotLearningCurve(logs,title="SARSA Pendulum Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))

"""Q LEARNING PENDULUM"""
alf = 0.15
epsi = 0.8
n_theta = 31
n_theta_dot = 31
(env,Q,policy,V,logs) = QLearningPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=2000)
plot_gridworld(V,policy,(n_theta,n_theta_dot),title="Q Learning Pendulum Value Function",hide_values=True,plot_arrows=False)
log = runPendulumEpisode(env,policy)
plotLog(log,title="Q Learning Trained Agent Trajectory")
PlotLearningCurve(logs,title="Q Learning Pendulum Learning Curve")

for alf in alphas:
    env,Q,policy,V,logs = QLearningPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=2000)
    PlotLearningCurve(logs,title="Q Learning Pendulum Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
alf = 0.15
for epsi in epsilons:
    env,Q,policy,V,logs = QLearningPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=2000)
    PlotLearningCurve(logs,title="Q Learning Pendulum Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
# %%
