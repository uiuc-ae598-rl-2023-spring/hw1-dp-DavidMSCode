# %%
"""This file takes all the algorithms and plotting functions and outputs the results for the GridWorld"""
import numpy as np
from policy_iteration_gridworld import PIGridworld
from value_iteration_gridworld import VIGridworld
from SARSA_gridworld import SARSAGridworld
from QLearning_gridworld import QLGridworld
from RLSharedFunctions import plot_gridworld, plotMeanV, runGridworldEpisode, plotLog, PlotLearningCurve
alphas = np.linspace(0.05,0.95,5)
epsilons = np.linspace(0.05,0.95,4)
"""Policy Iteration"""
discount = 0.95
tol = 1e-3
env, policy, V, meanV = PIGridworld(discount = 0.95, tol = 1e-3)
plot_gridworld(V,policy,(5,-1),"Policy Iteration Gridworld Value Function and Policy")
log = runGridworldEpisode(env,policy)
plotLog(log,"Policy Iteration Trained Trajectory")
plotMeanV(meanV,"Policy Iteration Learning Curve")

"""Value Iteration"""
discount = 0.95
tol = 1e-3
env, policy, V, meanV = VIGridworld(discount)
plot_gridworld(V,policy,(5,-1),"Value Iteration Gridworld Value Function and Policy")
log = runGridworldEpisode(env,policy)
plotLog(log,"Value Iteration Converged Trajectory")
plotMeanV(meanV,"Value Iteration Learning Curve")

"""SARSA"""
alf = 0.5           #learning step size
epsi = .1           #exploration chance
discount = 0.95     #future state pair value disc"ount
env,Q,policy,V,logs = SARSAGridworld(alf,epsi,discount,3000)
plot_gridworld(V,policy,(5,-1),title="SARSA Gridworld Value Function and Policy")
log = runGridworldEpisode(env,policy)
plotLog(log,title="SARSA Gridworld Trained Trajectory")
PlotLearningCurve(logs,title="SARSA Gridworld Learning Curve")
for alf in alphas:
    env,Q,policy,V,logs = SARSAGridworld(alf,epsi,discount,3000)
    PlotLearningCurve(logs,title="SARSA Gridworld Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
alf = 0.5
for epsi in epsilons:
    env,Q,policy,V,logs = SARSAGridworld(alf,epsi,discount,3000)
    PlotLearningCurve(logs,title="SARSA Gridworld Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))

"""QLearning"""
alf = 0.5           #learning step size
epsi = .1           #exploration chance
discount = 0.95     #future state pair value disc"ount
env,Q,policy,V,logs = QLGridworld(alf,epsi,discount,5000)
plot_gridworld(V,policy,(5,-1),title="Q Learning Gridworld Value Function and Policy")
log = runGridworldEpisode(env,policy)
plotLog(log,title="Q Learning Gridworld Trained Trajectory")
PlotLearningCurve(logs,title="Q Learning Gridworld Learning Curve")

for alf in alphas:
    env,Q,policy,V,logs = QLGridworld(alf,epsi,discount,5000)
    PlotLearningCurve(logs,title="Q Learning Gridworld Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
alf = 0.5
for epsi in epsilons:
    env,Q,policy,V,logs = QLGridworld(alf,epsi,discount,5000)
    PlotLearningCurve(logs,title="Q Learning Gridworld Learning Curve a ={:.2f}, e ={:.2f}".format(alf,epsi))
# %%
