# Jangwon Park, 2022
# MIE1613: Stochastic Simulation
# This file contains functions related to simulation output processing.

#%% Import modules

# standard modules
import numpy as np
import matplotlib.pyplot as plt

#%% functions

def draw_sample_path(x,qs,T,N,h,m,K,lambdas,mus,tau,P,trajectories):
    """
    Draw sample path from simulation and overlay \
        with optimal fluid trajectories.
    Arguments:
        x: arrival times (vector)
        qs: queue length process (matrix)
    """
    cut = int(T*(1+P))
    trajectories = trajectories[:cut,:].copy()
    
    plt.close('all')
    plt.figure()
    colors = ['b','r']
    labels = []
    try:
        cut2 = np.where(x==T*tau)[0][0] + 1
    except:
        cut2 = len(x)
    for i in range(N):
        plt.plot(x[:cut2],qs[i,:cut2],drawstyle='steps-post',color=colors[i],linewidth=0.75)
        labels.append("Queue " + str(i+1))
    
    # Plot vertical lines at beginning of each day
    for d in range(T+1):
        plt.axvline(x=d*tau,color="k",alpha=0.25,linewidth="0.5")
    
    # Plot fluid trajectories
    ticks = np.array(range(trajectories.shape[0]))/P*tau
    for i in range(N):
        plt.plot(ticks,trajectories[:,i],color=colors[i],linestyle="--",linewidth=0.75)
    
    fontsize = 14
    plt.xlabel("$t$", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel(r"$\bar{X}^\eta(t)$, $x(t)$", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.legend(labels, fontsize=fontsize)
    plt.show()