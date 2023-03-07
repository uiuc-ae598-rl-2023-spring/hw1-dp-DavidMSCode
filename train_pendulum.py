# %%
"""This function takes all the model free algorithms and plotting functions and returns the results for the pendulum"""
import numpy as np
import matplotlib.pyplot as plt
from SARSA_pendulum import SARSAPendulum
from QLearning_pendulum import QLearningPendulum
from RLSharedFunctions import plot_gridworld, runPendulumEpisode, plotLog, PlotLearningCurve
"""IO stuff"""
figsubdirs = "figures/pendulum/"
do_SARSA = False
do_QL = True
alf0 = 0.25
epsi0 = 0.1
"""Plotting ranges"""
alphas = np.linspace(0.05,0.95,5)
epsilons = np.linspace(0.05,0.95,5)
max_eps = 5000
if do_SARSA:
    """SARSA PENDULUM"""
    alf = alf0
    epsi = epsi0
    n_theta = 15
    n_theta_dot = 21
    (env,Q,policy,V,logs) = SARSAPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=max_eps)
    plot_gridworld(V,policy,(n_theta,n_theta_dot),title="SARSA Pendulum",hide_values=False,plot_arrows=False,valueFontSize=4)
    plt.savefig(figsubdirs+"SARSA Pendulum Learned Value and Policy",bbox_inches='tight',dpi=200)
    log = runPendulumEpisode(env,policy)
    plotLog(log,title="SARSA Trained Agent Trajectory")
    plt.savefig(figsubdirs+"SARSA Pendulum Trained Agent Trajectory",bbox_inches='tight',dpi=200)
    PlotLearningCurve(logs,title="SARSA Pendulum Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
    plt.savefig(figsubdirs+"SARSA Pendulum Learning Curve",bbox_inches='tight',dpi=200)
    print("Iterating through SARSA Alpha and Epsilon")
    for alf in alphas:
        env,Q,policy,V,logs = SARSAPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=max_eps)
        PlotLearningCurve(logs,title="SARSA Pendulum Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
        plt.savefig(figsubdirs+"SARSA Pendulum Learning Curve alpha={:.2f} epsilon={:.2f}.png".format(alf,epsi),bbox_inches='tight',dpi=200)
    alf = alf0
    for epsi in epsilons:
        env,Q,policy,V,logs = SARSAPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=max_eps)
        PlotLearningCurve(logs,title="SARSA Pendulum Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
        plt.savefig(figsubdirs+"SARSA Pendulum Learning Curve alpha={:.2f} epsilon={:.2f}.png".format(alf,epsi),bbox_inches='tight',dpi=200)
    print("Finished SARSA")
if do_QL:
    """Q LEARNING PENDULUM"""
    alf = alf0
    epsi = epsi0
    n_theta = 15
    n_theta_dot = 21
    (env,Q,policy,V,logs) = QLearningPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=max_eps)
    plot_gridworld(V,policy,(n_theta,n_theta_dot),title="Q Learning Pendulum",hide_values=False,plot_arrows=False,valueFontSize=4)
    plt.savefig(figsubdirs+"Q Learning Pendulum Learned Value and Policy",bbox_inches='tight',dpi=200)
    log = runPendulumEpisode(env,policy)
    plotLog(log,title="Q Learning Trained Agent Trajectory")
    plt.savefig(figsubdirs+"Q Learning Pendulum Trained Agent Trajectory",bbox_inches='tight',dpi=200)
    PlotLearningCurve(logs,title="Q Learning Pendulum Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
    plt.savefig(figsubdirs+"Q Learning Pendulum Learning Curve",bbox_inches='tight',dpi=200)
    print("Iterating through Q Learning Alpha and Epsilon")
    for alf in alphas:
        env,Q,policy,V,logs = QLearningPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=max_eps)
        PlotLearningCurve(logs,title="Q Learning Pendulum Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
        plt.savefig(figsubdirs+"Q Learning Pendulum Learning Curve alpha={:.2f} epsilon={:.2f}.png".format(alf,epsi),bbox_inches='tight',dpi=200)
    alf = alf0
    for epsi in epsilons:
        env,Q,policy,V,logs = QLearningPendulum(n_theta,n_theta_dot,alf=alf,epsi=epsi,discount=0.95,max_episodes=max_eps)
        PlotLearningCurve(logs,title="Q Learning Pendulum Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
        plt.savefig(figsubdirs+"Q Learning Pendulum Learning Curve alpha={:.2f} epsilon={:.2f}.png".format(alf,epsi),bbox_inches='tight',dpi=200)
    print("Finished Q Learning")
