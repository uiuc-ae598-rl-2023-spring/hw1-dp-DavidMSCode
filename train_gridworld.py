# %%
"""This file takes all the algorithms and plotting functions and outputs the results for the GridWorld"""
import numpy as np
import matplotlib.pyplot as plt
from policy_iteration_gridworld import PIGridworld
from value_iteration_gridworld import VIGridworld
from SARSA_gridworld import SARSAGridworld
from QLearning_gridworld import QLGridworld
from RLSharedFunctions import plot_gridworld, plotMeanV, runGridworldEpisode, plotLog, PlotLearningCurve
"""IO Stuff"""
figsubdirs = "figures/gridworld/"
alf0 = 0.5           #learning step size
epsi0 = .1           #exploration chance
"""Plotting Ranges"""
alphas = np.linspace(0.05,0.95,5)
epsilons = np.linspace(0.05,0.95,5)
"""Policy Iteration"""
discount = 0.95
tol = 1e-3
env, policy, V, meanV = PIGridworld(discount = 0.95, tol = 1e-3)
plot_gridworld(V,policy,(5,-1),"Policy Iteration Gridworld Value Function and Policy")
plt.savefig(figsubdirs+"Policy Iteration Learned Value and Policy",bbox_inches='tight',dpi=200)
log = runGridworldEpisode(env,policy)
plotLog(log,"Policy Iteration Gridworld Trained Trajectory")
plt.savefig(figsubdirs+"Policy Iteration Gridworld Trained Agent Trajectory",bbox_inches='tight',dpi=200)
plotMeanV(meanV,"Policy Iteration Gridworld Learning Curve")
plt.savefig(figsubdirs+"Policy Iteration Gridworld Learning Curve",bbox_inches='tight',dpi=200)

"""Value Iteration"""
discount = 0.95
tol = 1e-3
env, policy, V, meanV = VIGridworld(discount)
plot_gridworld(V,policy,(5,-1),"Value Iteration Gridworld Value Function and Policy")
plt.savefig(figsubdirs+"Value Iteration Gridworld Learned Value and Policy",bbox_inches='tight',dpi=200)
log = runGridworldEpisode(env,policy)
plotLog(log,"Value Iteration Converged Trajectory")
plt.savefig(figsubdirs+"Value Iteration Gridworld Trained Agent Trajectory",bbox_inches='tight',dpi=200)
plotMeanV(meanV,"Value Iteration Learning Curve")
plt.savefig(figsubdirs+"Value Iteration Gridworld Learning Curve",bbox_inches='tight',dpi=200)

"""SARSA"""
alf = alf0           #learning step size
epsi = epsi0           #exploration chance
discount = 0.95     #future state pair value disc"ount
env,Q,policy,V,logs = SARSAGridworld(alf,epsi,discount,3000)
plot_gridworld(V,policy,(5,-1),title="SARSA Gridworld Value Function and Policy")
plt.savefig(figsubdirs+"SARSA Gridworld Learned Value and Policy",bbox_inches='tight',dpi=200)
log = runGridworldEpisode(env,policy)
plotLog(log,title="SARSA Gridworld Trained Trajectory")
plt.savefig(figsubdirs+"SARSA Gridworld Trained Agent Trajectory",bbox_inches='tight',dpi=200)
PlotLearningCurve(logs,title="SARSA Gridworld Learning Curve")
plt.savefig(figsubdirs+"SARSA Gridworld Learning Curve",bbox_inches='tight',dpi=200)
for alf in alphas:
    env,Q,policy,V,logs = SARSAGridworld(alf,epsi,discount,3000)
    PlotLearningCurve(logs,title="SARSA Gridworld Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
    plt.savefig(figsubdirs+"SARSA Gridworld Learning Curve alpha={:.2f} epsilon={:.2f}.png".format(alf,epsi),bbox_inches='tight',dpi=200)
alf = alf0
for epsi in epsilons:
    env,Q,policy,V,logs = SARSAGridworld(alf,epsi,discount,3000)
    PlotLearningCurve(logs,title="SARSA Gridworld Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
    plt.savefig(figsubdirs+"SARSA Gridworld Learning Curve alpha={:.2f} epsilon={:.2f}.png".format(alf,epsi),bbox_inches='tight',dpi=200)

"""QLearning"""
alf = alf0           #learning step size
epsi = epsi0           #exploration chance
discount = 0.95     #future state pair value disc"ount
env,Q,policy,V,logs = QLGridworld(alf,epsi,discount,3000)
plot_gridworld(V,policy,(5,-1),title="Q Learning Gridworld Value Function and Policy")
plt.savefig(figsubdirs+"Q Learning Gridworld Learned Value and Policy",bbox_inches='tight',dpi=200)
log = runGridworldEpisode(env,policy)
plotLog(log,title="Q Learning Gridworld Trained Trajectory")
plt.savefig(figsubdirs+"Q Learning Gridworld Trained Agent Trajectory",bbox_inches='tight',dpi=200)
PlotLearningCurve(logs,title="Q Learning Gridworld Learning Curve")
plt.savefig(figsubdirs+"Q Learning Gridworld Learning Curve",bbox_inches='tight',dpi=200)
for alf in alphas:
    env,Q,policy,V,logs = QLGridworld(alf,epsi,discount,3000)
    PlotLearningCurve(logs,title="Q Learning Gridworld Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
    plt.savefig(figsubdirs+"Q Learning Gridworld Learning Curve alpha={:.2f} epsilon={:.2f}.png".format(alf,epsi),bbox_inches='tight',dpi=200)
alf = alf0
for epsi in epsilons:
    env,Q,policy,V,logs = QLGridworld(alf,epsi,discount,3000)
    PlotLearningCurve(logs,title="Q Learning Gridworld Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
    plt.savefig(figsubdirs+"Q Learning Gridworld Learning Curve alpha={:.2f} epsilon={:.2f}.png".format(alf,epsi),bbox_inches='tight',dpi=200)

# %%
