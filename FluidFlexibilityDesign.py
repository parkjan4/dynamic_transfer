# Jangwon Park, 2022

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import time
import random
from gurobipy import *


#%% functions
def extract_output(T,N,x0,P,I,y,z):
    """
    Extract output from optimized Gurobi decision variables.
    Arguments:
        T: number of periods
        N: number of queues
        x0: initial state in fluid control problem
        P: number of fixed-width intervals per period
        I: Optimal transfer decisions from Gurobi model
        y: Decision variable from optimized Gurobi model 
        z: Decision variable from optimized Gurobi model 
    Output:
        Transfers: optimal transfer decisions (TxNxN matrix)
    """
    
    Transfers = np.zeros((T,N,N))
    for k in range(T):
        for i in range(N):
            for j in range(N):
                Transfers[k,i,j] = I[k,i,j].x
    
    pre_states = np.zeros((T,N))
    pre_states[0,:] = x0 # first period
    for k in range(1,T):
        for i in range(N):
            pre_states[k,i] = z[k-1,i,P].x
    
    post_states = np.zeros((T,N))
    for k in range(T):
        for i in range(N):
            post_states[k,i] = y[k,i,0].x
            
    return Transfers, pre_states, post_states


def extract_trajectory(z,T,N,P):
    trajectories = np.zeros(((T)*(P+1),N))
    for q in range(N):
        idx = 0
        for k in range(T):
            for m in range(P+1):
                trajectories[idx,q] = z[k,q,m].x # variable from gurobi
                idx += 1
                
    return trajectories


def draw_trajectory(x,T,N,trajectories,ax):
    """
    Draw fluid trajectories (discontinuous).
    """
    # Append initial state
    trajectories = np.array(np.concatenate([np.matrix(x),trajectories],axis=0))
    
    # discontinuous trajectories for plotting
    xaxis = [0]
    discont_trajectories = [trajectories[0,:]]
    idx = 0
    for t in range(T):
        for m in range(P+1):
            idx += 1
            if idx==t*(P+1)+1:
                xaxis.append(xaxis[-1]) # for jump discontinuity, set same time index
            else:
                xaxis.append(xaxis[-1] + 1/P)
            
            discont_trajectories.append(trajectories[idx,:])
    discont_trajectories = np.array(discont_trajectories)
    
    # color = next(ax._get_lines.prop_cycler)['color']
    
    # Plot trajectories
    for q in range(N):
        plt.plot(xaxis,discont_trajectories[:,q],linewidth=2,label="Queue" + str(q+1))
    plt.legend(fontsize=18)
    plt.xticks(np.arange(0,T+1,1),fontsize=22)
    plt.xlabel("$k$",fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylabel("$x^k$",fontsize=22)
    # plt.title("Fluid trajectory",fontsize=24)
    plt.grid(alpha=0.1)
    return

        
def solve_fluid_control(T,N,h,m,K,lambdas,mus,tau,P,x0,\
                        freq=1,E=None,UB=GRB.INFINITY):
    """
    Solve multi-period fluid control problem.
    Arguments:
        T: number of periods (days)
        N: number of queues
        h: holding cost (vector)
        m: variable transfer cost (NxN matrix)
        K: fixed transfer cost (NxN matrix)
        lambdas: arrival rates (TxN matrix)
        mus: service rates (TxN matrix)
        tau: length of one period
        P: number of fixed-width intervals per period
        x0: initial state 
        freq: frequency of patient transfers (days)
        G: network structure (edges not used for transfer; set)
        UB: max. daily number of transfers
    Outputs:
        Transfers: optimal transfer decisions (TxNxN matrix)
        pre_states: queue lengths prior to transfer (TxNxN matrix)
        post_states: queue lengths after transfer (TxNxN matrix)
        model.objVal: optimal objective function value
    """
    
    # model declaration
    model = Model("multi_fluid_control")
    
    # decision variables
    I = model.addVars(T,N,N,lb=0,ub=GRB.INFINITY,name="transfer")
    # no. customers in each "bin" or interval in a period
    y = model.addVars(T+1,N,P+1,lb=-GRB.INFINITY) 
    z = model.addVars(T+1,N,P+1,lb=0) # max{y_i^t,0}
    # Ind = model.addVars(T,N,N,vtype=GRB.BINARY,name="indicator")
    
    # objective function
    model.setObjective(quicksum(h[i] * tau/P * \
                    quicksum(0.5*(z[k,i,t] + z[k,i,t-1]) \
                             for t in range(1,P+1))
                    for i in range(N) for k in range(T)) \
                    + quicksum(quicksum(m[i,j]*I[k,i,j] for j in range(N))
                    # + quicksum(K[i,j]*Ind[k,i,j] for j in range(N))
                    for i in range(N) for k in range(T)), GRB.MINIMIZE)
    
    # constraints on I (transfer)
    for i in range(N):
        # first period
        model.addConstr(quicksum(I[0,i,j] for j in range(N)) <= z[0,i,0]) 
        # model.addConstr(quicksum(I[0,i,j] for j in range(N)) <= x0[i]) 
        for k in range(1,T):
            model.addConstr(quicksum(I[k,i,j] for j in range(N)) \
                        <= z[k-1,i,P]) # cap on transfer out
                    
    # upper bound on daily transfers
    model.addConstrs(quicksum(I[k,i,j] for j in range(N)) \
                 <= UB for i in range(N) for k in range(T))
    
    # network configuration constraints
    model.addConstrs(I[k,i,i]==0 for k in range(T) for i in range(N)) # no self-edges
    if E!=None: # If None, full flexibility
        all_edges = itertools.combinations(range(N),2)
        for edge in all_edges:
            if edge in E:
                continue # Allow transfers
            else:
                q1 = edge[0]
                q2 = edge[1]
                
                # Can't transfer q1<->q2
                model.addConstrs(I[k,q1,q2] == 0 for k in range(T)) 
                model.addConstrs(I[k,q2,q1] == 0 for k in range(T)) 
    
    # constraints on y and z
    for i in range(N):  
        # initial condition
        model.addConstr(y[0,i,0] >= z[0,i,0]
                    + quicksum(I[0,j,i] for j in range(N)) 
                    - quicksum(I[0,i,j] for j in range(N))) 
        # model.addConstr(y[0,i,0] >= x0[i]
        #             + quicksum(I[0,j,i] for j in range(N)) 
        #             - quicksum(I[0,i,j] for j in range(N))) 
        for k in range(T):
            # constraints for discretized intervals within a period
            for t in range(1,P+1):
                model.addConstr(y[k,i,t] >= \
                            y[k,i,t-1] + (lambdas[k,i]-mus[k,i])*tau/P)
            for t in range(P+1):
                if (t==0) and (k==0):
                    model.addConstr(z[k,i,t] >= x0[i])
                else:
                    model.addConstr(z[k,i,t] >= y[k,i,t])
                # model.addConstr(z[k,i,t] >= y[k,i,t])
            
        # border between two adjacent periods
        for k in range(1,T):
            model.addConstr(y[k,i,0] >= z[k-1,i,P]
                        + quicksum(I[k,j,i] for j in range(N)) 
                        - quicksum(I[k,i,j] for j in range(N)))
    
    # # constraints on indicator
    # ## just for 0th period
    # for k in range(T):
    #     for i in range(N):
    #         model.addConstr(Ind[k,i,i]==0) # since transfer to self not coded
    #         for j in range(N):
    #             model.addGenConstrIndicator(Ind[k,i,j],0,I[k,i,j]==0)
    #             # Note: sum(x) is like a big-M constant.
    #             # Note: If mu < lambda, MUST change sum(x) to another big constant.
    #             # model.addConstr(I[k,i,j] <= sum(x)*Ind[k,i,j]) 
                
                
    # # temporary: to be deleted
    # model.addConstr(I[0,0,1]==0)
        
    model.params.logtoconsole = 0
    model.optimize()
    
    # Extract output
    Transfers, pre_states, post_states = extract_output(T,N,x0,P,I,y,z)
    
    # Extract trajectories
    trajectories = extract_trajectory(z, T, N, P)
    return Transfers, pre_states, post_states, model.objVal, trajectories, model


def solve_robust_fluid_control(T,N,h,m,K,lambdas,mus,tau,P,x,\
                               freq=1,E=None,UB=GRB.INFINITY):
    """
    Solve worst-case multi-period fluid control problem w.r.t. initial conditions.
    Same arguments as solve_fluid_control.
    """
    
    model = Model("robust_multi_fluid_control")
    
    # decision variables
    x0 = model.addVars(N,lb=0,name="initial_conditions")
    p = model.addVars(T,N,P,lb=0,name="intraperiod_dynamics")
    q = model.addVars(T,N,lb=0,name="transfer_states")
    w = model.addVars(T,N,P+1,lb=-GRB.INFINITY,name="nonnegative_fluid_level")
    r = model.addVars(T,N,lb=0,name="transfer_bounds")
    s = model.addVars(T,N,N,lb=-GRB.INFINITY,name="flexibility_design")
    
    # objective function
    model.setObjective(quicksum((lambdas[k,i] - mus[k,i])*tau/P*p[k,i,t] \
                       for t in range(P) for k in range(T) for i in range(N))\
                       + quicksum(x0[i]*w[0,i,0] for i in range(N)),\
                       GRB.MAXIMIZE)
        
        
    # constraints (in order of appearance on my Ipad)
    model.addConstrs(-p[0,i,0] + q[0,i] == 0 for i in range(N))
    for i in range(N):
        model.addConstrs(p[k,i,t] - p[k,i,t+1] - w[k,i,t+1] == 0 \
                         for t in range(P-1) for k in range(T))
    
        model.addConstrs(p[k,i,P-1] - w[k,i,P] == 0 for k in range(T))
     
        model.addConstrs(-p[k,i,0] + q[k,i] - w[k,i,0] == 0 for k in range(1,T))
        
        
    model.addConstrs(-q[0,i] + w[0,i,0] + r[0,i] <= h[i]*tau/2/P for i in range(N))
    for i in range(N):
        model.addConstrs(w[k,i,t] <= h[i]*tau/P \
                         for t in range(1,P) for k in range(T))
        
        model.addConstrs(-q[k,i] + w[k-1,i,P] + r[k,i] <= h[i]*tau/2/P for k in range(1,T))
        
        model.addConstrs(w[k,i,0] <= h[i]*tau/2/P for k in range(1,T))
        
        model.addConstr(w[T-1,i,P] <= h[i]*tau/2/P)
    
    
    # network configuration constraints
    model.addConstrs(-r[k,i] + s[k,i,i] <= 0 \
                     for k in range(T) for i in range(N)) # b/c no self-edges
    all_edges = itertools.combinations(range(N),2)
    for edge in all_edges:
        q1 = edge[0]
        q2 = edge[1]
        if edge in E:
            model.addConstrs(q[k,q1] - q[k,q2] - r[k,q1] <= m[q1,q2] for k in range(T))
            model.addConstrs(q[k,q2] - q[k,q1] - r[k,q2] <= m[q2,q1] for k in range(T))
        else:
            model.addConstrs(q[k,q1] - q[k,q2] - r[k,q1] + s[k,q1,q2] <= 0 \
                             for k in range(T))
            model.addConstrs(q[k,q2] - q[k,q1] - r[k,q2] + s[k,q2,q1] <= 0 \
                             for k in range(T))

    
    # constraints on initial conditions (ambiguity set)
    model.addConstr(quicksum(x0[i] for i in range(N)) == x) # Option 1: includes extreme cases
    
    model.params.logtoconsole = 0
    model.params.NonConvex = 2
    model.optimize()
    
    # for t in range(P):
    #     for k in range(T):
    #         for i in range(N):
    #             print("p[{},{},{}]: {}".format(k,i,t,p[k,i,t].x))
    
    # print(w[0,0,0].x, w[0,1,0].x)
    
    return model.objVal
    

#%% Study value of flexibility (one design)
# N = 2 # number of queues

# # cost parameters
# lambdas = [0.9] * N
# mus = [1] * N
# h = [2] * N
# m = np.ones((N,N))*5
# K = np.ones((N,N))*15

# # other parameters
# tau = 10
# P = 100 # level of discretization in fluid control
# T = 10 # number of periods in fluid control
# x = [0] * (N-1) + [10] # initial conditions

# # Define flexibility design to evaluate 
# # E = [(i,i+1) for i in range(0,N,2)]
# E = None

# # Solve fluid control problem
# sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
#                                     np.array([lambdas]*T),\
#                                     np.array([mus]*T),tau,P,x,E=E)
# # print(sol[0])
# print("Optimal cost: {:.2f}, E: {}".format(obj, E))

# # Plot optimal trajectories
# plt.close('all')
# plt.figure(figsize=(15,7))
# ax = plt.gca()
# draw_trajectory(x, T, N, trajectories, ax)


#%% Study value of flexibility (two designs only!)

# Ns = range(2,32,2) # number of queues
# rel = [] # relative performances 

# for N in Ns:
#     # cost parameters
#     lambdas = [0.9] * N
#     mus = [1] * N
#     h = [2] * N
#     m = np.ones((N,N))*5
#     K = np.ones((N,N))*15
    
#     # other parameters
#     tau = 10
#     P = 100 # level of discretization in fluid control
#     T = N*2 # number of periods in fluid control
#     # x = [0] * (N-1) + [N*2] # initial conditions
#     # x = [1]*N # initial conditions
#     x = [1]*int(N/2) + [3]*int(N/2) # initial conditions
    
#     # Define flexibility designs to evaluate 
#     Es = ([(i,i+1) for i in range(0,N,2)], None)
#     objs = []
#     for E in Es:
#         # Solve fluid control problem
#         sol, pre, post, obj, trajectories, _ = solve_fluid_control(T,N,h,m,K,\
#                                             np.array([lambdas]*T),\
#                                             np.array([mus]*T),tau,P,x,E=E)
#         # Save optimal cost
#         objs.append(obj)        
    
#     # Compute relative performance
#     rel.append(objs[-1]/objs[0]) # -1 is by default fully flexible design
#     print("N: {}, r(N): {:.4f}".format(N, rel[-1]))


#%% For above cell: Plot relative performances
# densities = [(N/2)/(N*(N-1)/2) for N in Ns]

# plt.close("all")
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx() # secondary y-axis
# plot1 = ax1.plot(Ns,rel,linewidth=2,label="$r(N)$: all customers in one queue")
# plot2 = ax1.plot(Ns,[1]*len(Ns),label="$r(N)$: uniformly distributed")
# plot3 = ax2.plot(Ns,densities,linewidth=2,linestyle='--',label=r"$\rho(N)$")

# # axes decoration
# axs = [ax1, ax2]
# for ax in axs:
#     ax.set_xlabel("$N$",fontsize=24)
#     ax.set_xticks(Ns,fontsize=18)
#     ax.grid(alpha=0.33)
# plots = plot1 + plot2 + plot3
# labs = [p.get_label() for p in plots]
# ax.legend(plots, labs, loc="best", fontsize=18)
# ax1.set_ylabel("$r(N)$",fontsize=24)
# ax2.set_ylabel(r"$\rho(N)$",fontsize=24)
# plt.title("Relative performance of a paired network",fontsize=24)
# plt.show()


#%% Robust fluid control problem
# N = 6 # number of queues

# # cost parameters
# lambdas = [0.9] * N
# mus = [1] * N
# h = [2] * N
# m = np.ones((N,N))*5
# K = np.ones((N,N))*0

# # other parameters
# tau = 10
# P = 30 # level of discretization in fluid control
# T = 10 # number of periods in fluid control
# x = [0] * (N-1) + [10] # initial conditions

# # Define flexibility design to evaluate 
# # E = [(i,i+1) for i in range(0,N,2)]
# E = itertools.combinations(range(N),2) # full flexibility

# # Solve fluid control problem
# tic = time.time()
# obj = solve_robust_fluid_control(T,N,h,m,K,\
#                                   np.array([lambdas]*T),\
#                                   np.array([mus]*T),tau,P,sum(x),E=E)
    
# print("Worst cost: {:.5f}, time elapsed (min): {:.2f}".format(obj,(time.time()-tic)/60))


#%% Sampling x0


#%% Samples of x0
Ns = range(2,22,2)

# non-cost parameters
tau = 10
P = 100 # level of discretization in fluid control

num_samples = 100
eps = float("inf")
for N in Ns:
    # cost parameters
    lambdas = [0.9] * N
    mus = [1] * N
    h = [2] * N
    m = np.ones((N,N))*5
    K = np.ones((N,N))*15
    
    x = 10*N # number of customers
    T = int(x/2) # number of periods in fluid control
    xu = [x/N] * N # uniform distribution
    
    rels = []
    variances = [] # variance of initial conditions
    alphas = [1] * N # parameters for Dirichlet distribution
    for j in range(num_samples):
        x0 = x * np.random.dirichlet(alphas) 
        while np.std(x0 - xu) > eps:
            x0 = x * np.random.dirichlet(alphas)
        
        # Define flexibility designs to evaluate 
        x0 = sorted(x0) # increasing order
        Es = ([(i,i+1) for i in range(0,N,2)], None)
        objs = []
        
        for E in Es:
            # Solve fluid control problem
            sol, pre, post, obj, trajectories, _ = solve_fluid_control(T,N,h,m,K,\
                                                np.array([lambdas]*T),\
                                                np.array([mus]*T),tau,P,x0,E=E)
            # Save optimal cost
            objs.append(obj)        
        
        # Compute relative performance
        rels.append(objs[-1]/objs[0]) # objs[-1] by default is fully flexible design
        variances.append(np.var(x0))
        print("N: {}, r(N): {:.4f}, variance: {:.4f}".format(N, objs[-1]/objs[0], np.var(x0)))
    
    print("N: {}, Expected: {:.4f}, Worst: {:.4f}, SD: {:.4f}"\
          .format(N, np.mean(rels), min(rels), np.std(rels)))
    print("------------------------------------------------------------------")

# # N = 10 # number of queues
# # eps = 1 # average deviation
# # x = 100 # initial number of customers
# # xu = [x/N] * N # reference distribution

# # sampler = Model("Sampler")

# # x_init = sampler.addVars(N,lb=0,name="initial_conditions")
# # z = sampler.addVars(N,lb=-GRB.INFINITY)

# # sampler.setObjective(0,GRB.MINIMIZE)

# # sampler.addConstr(quicksum(x_init[i] for i in range(N)) <= x)
# # sampler.addConstr(quicksum(z[i] for i in range(N)) <= N*eps)
# # sampler.addConstrs(z[i] >= x_init[i] - xu[i] for i in range(N))
# # sampler.addConstrs(z[i] >= -x_init[i] + xu[i] for i in range(N))

# # sampler.params.logtoconsole = 0
# # sampler.optimize()


# # x0 = []
# # for i in range(N):
# #     x0.append(x_init[i].x)
    
# # print(x0)


# # x0 = []
# # for i in range(N):
# #     if i==N-1: # last queue
# #         x0.append(x-sum(x0))
# #     else:
# #         ci = max(0,N*eps - sum(x0))
# #         sample = random.uniform(max(0,xu[i]-ci),xu[i]+ci)
# #         x0.append(sample) # Randomly sample and save

# # # Check if level of variation is not exceeded
# # avg_dev = np.abs(np.array(x0)-np.array(xu)).sum() / N
# # print(sum(x0), x0)
# # print("Total deviation:",avg_dev)