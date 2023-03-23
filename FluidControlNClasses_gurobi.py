# Jangwon Park, 2022

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import time
from gurobipy import *


#%% functions

def extract_output(T,N,x0,M,I,y,z):
    """
    Extract output from optimized Gurobi decision variables.
    Arguments:
        T: number of stages
        N: number of queues
        x0: initial state in fluid control problem
        M: number of fixed-width intervals per stage
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
    pre_states[0,:] = x0 # first stage
    for k in range(1,T):
        for i in range(N):
            pre_states[k,i] = z[k-1,i,M].x
    
    post_states = np.zeros((T,N))
    for k in range(T):
        for i in range(N):
            # post_states[k,i] = y[k,i,0].x
            post_states[k,i] = pre_states[k,i] - sum([I[k,i,j].x for j in range(N)]) \
                                               + sum([I[k,j,i].x for j in range(N)])
            
    return Transfers, pre_states, post_states


def extract_trajectory(z,T,N,M):
    trajectories = np.zeros(((T)*(M+1),N))
    for q in range(N):
        idx = 0
        for k in range(T):
            for m in range(M+1):
                trajectories[idx,q] = z[k,q,m].x # variable from gurobi
                idx += 1
                
    return trajectories


def draw_trajectory(x,T,trajectories,ax,title,labels):
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
    
    # Plot trajectories
    for q in range(N):
        color = next(ax._get_lines.prop_cycler)['color']
        if q==0:
            plt.plot(xaxis,discont_trajectories[:,q],linewidth=2,color=color,label=labels[q])
        else:
            plt.plot(xaxis,discont_trajectories[:,q],linewidth=1,color=color,label=labels[q])
    plt.legend(fontsize=18)
    plt.xticks(np.arange(0,T+1,1),fontsize=22)
    plt.xlabel("$k$",fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylabel("$x_{ik}$",fontsize=22)
    plt.ylim([0,8.5])
    plt.title(title,fontsize=24)
    plt.grid(alpha=0.33)
    return
        
        
def solve_fluid_control(T,N,h,m,K,lambdas,mus,tau,P,x0,\
                        freq=1,E=None,UB=GRB.INFINITY,force_zero=False,\
                        force_pos=False,force_bound=False,force_interior=False,senders=[]):
    """
    Solve multi-stage fluid control problem.
    Arguments:
        T: number of periods (days)
        N: number of queues
        h: holding cost (vector)
        m: variable transfer cost (NxN matrix)
        K: fixed transfer cost (NxN matrix)
        lambdas: arrival rates (TxN matrix)
        mus: service rates (TxN matrix)
        tau: length of one stage
        P: number of fixed-width intervals per stage
        x0: initial state 
        freq: frequency of patient transfers (days)
        G: network structure (edges not used for transfer; set)
        UB: max. daily number of transfers
        force_zero: Force zero transfers in period 1.
        force_pos: Force positive transfers in period 1.
        force_bound: Force a boundary solution in period 1.
        force_interior: Force an interior solution in period 1.
        senders: Set of sender queues to use in constraints
    Outputs:
        Transfers: optimal transfer decisions (TxNxN matrix)
        pre_states: queue lengths prior to transfer (TxNxN matrix)
        post_states: queue lengths after transfer (TxNxN matrix)
        model.objVal: optimal objective function value
    """
    
    # model declaration
    model = Model("multistage_fluid_control")
    
    # decision variables
    tau = tau * freq # controlling time scale of fluid control problem
    I = model.addVars(T+1,N,N,lb=0,ub=GRB.INFINITY,name="transfer")
    # no. customers in each "bin" or interval in a stage
    y = model.addVars(T+1,N,P+1,lb=-GRB.INFINITY) 
    z = model.addVars(T+1,N,P+1,lb=0) # max{y_i^t,0}
    # Ind = model.addVars(T,N,N,vtype=GRB.BINARY,name="indicator") # queue pair-specific 
    # Ind = model.addVars(T,N,vtype=GRB.BINARY,name="indicator") # queue-specific
    Ind = model.addVars(T,vtype=GRB.BINARY,name="indicator")  # system-wide joint setup
    
    # objective function
    model.setObjective(quicksum(h[i] * tau/P * \
                    quicksum(0.5*(z[k,i,t] + z[k,i,t-1]) \
                             for t in range(1,P+1))
                    for i in range(N) for k in range(T)) \
                    + quicksum(K*Ind[k] for k in range(T)) # uncomment for system-wide joint setup cost
                    + quicksum(quicksum(m[i,j]*I[k,i,j] for j in range(N)) \
                    # + quicksum(K[i,j]*Ind[k,i,j] for j in range(N)) # uncomment for queue pair-specific
                    # + K[i]*Ind[k,i] # uncomment for queue-specific setup costs
                    for i in range(N) for k in range(T)), GRB.MINIMIZE)
    
    # constraints on I (transfer)
    for i in range(N):
        # first stage
        model.addConstr(quicksum(I[0,i,j] for j in range(N)) <= z[0,i,0]) 
        # model.addConstr(quicksum(I[0,i,j] for j in range(N)) <= x0[i]) 
        for k in range(1,T):
            model.addConstr(I[k,i,i] == 0) # no transfers to itself
            model.addConstr(quicksum(I[k,i,j] for j in range(N)) \
                        <= z[k-1,i,P]) # cap on transfer out
            
        # no decision at the end of final stage
        model.addConstrs(I[T,i,j]==0 for j in range(N)) 
        
    # upper bound on daily transfers
    model.addConstrs(quicksum(I[k,i,j] for j in range(N)) \
                 <= UB for i in range(N) for k in range(T))
    
    # network configuration constraints
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
            # constraints for discretized intervals within a stage
            for t in range(1,P+1):
                # if t>=P/3 and t<2*P/3:
                #     raise_factor = 1.2
                #     model.addConstr(y[k,i,t] >= \
                #                 y[k,i,t-1] + (raise_factor*lambdas[k,i]-mus[k,i])*tau/P) # time-varying arrivals
                # else:
                model.addConstr(y[k,i,t] >= \
                            y[k,i,t-1] + (lambdas[k,i]-mus[k,i])*tau/P)
            for t in range(P+1):
                if (t==0) and (k==0):
                    model.addConstr(z[k,i,t]>=x0[i])
                else:
                    model.addConstr(z[k,i,t] >= y[k,i,t])
                # model.addConstr(z[k,i,t] >= y[k,i,t])
            
            # border between two adjacent stages
            model.addConstr(y[k+1,i,0] >= z[k,i,P]
                        + quicksum(I[k+1,j,i] for j in range(N)) 
                        - quicksum(I[k+1,i,j] for j in range(N)))
    
    # constraints on indicator
    ## just for 0th period
    for k in range(T):
        for i in range(N): 
            # model.addGenConstrIndicator(Ind[k,i],0,quicksum(I[k,j,i] for j in range(N))==0) # uncomment to implement queue-specific setup costs
            # model.addConstr(Ind[k,i,i]==0) # since transfer to self not coded (uncomment to implement queue pair-specific setup costs)
            for j in range(N): # uncomment to implement queue pair-specific setup costs or system-wide joint setup costs
                # model.addGenConstrIndicator(Ind[k,i,j],0,I[k,i,j]==0) # uncomment to implement queue pair-specific setup costs
                # Note: sum(x) is like a big-M constant.
                # Note: If mu < lambda, MUST change sum(x) to another big constant.
                # model.addConstr(I[k,i,j] <= sum(x)*Ind[k,i,j]) 
                model.addGenConstrIndicator(Ind[k],0,I[k,i,j]==0) # uncomment to implement system-wide joint setup costs
                

    # Force no transfers in period 1
    if force_zero:
        model.addConstrs(I[0,i,j]==0 for i in range(N) for j in range(N))
    elif force_pos:
        # model.addConstr(quicksum(I[0,i,j] for i in range(N) for j in range(N)) >= 0.01)
        model.addConstr(Ind[0]==1) # only use for system-wide joint setup cost
    elif force_bound:
        # boundaries = np.round((np.array(mus[0,:]) - np.array(lambdas[0,:]))*tau,2) # must be known a priori
        boundaries = np.array([1,1,1])
        Ind2 = model.addVars(N,vtype=GRB.BINARY)  # interior indicator
        # Recall: y[0,i,0] is the post-transfer state at queue i at the start of period 0
        for i in range(N):
            model.addGenConstrIndicator(Ind2[i],0,y[0,i,0]<=boundaries[i])
        model.addConstr(quicksum(Ind2[i] for i in range(N)) <= N-1) # If ==N, it means interior
    elif force_interior:
        # Below is a heuristic for pairwise constant setup costs
        boundaries = np.round((np.array(mus[0,:]) - np.array(lambdas[0,:]))*tau,2) + 0.1
        Ind2 = model.addVars(N,vtype=GRB.BINARY)  # interior indicator
        # Recall: y[0,i,0] is the post-transfer state at queue i at the start of period 0
        for i in range(N):
            model.addGenConstrIndicator(Ind2[i],0,y[0,i,0]<=boundaries[i])
        model.addConstr(quicksum(Ind2[i] for i in range(N)) == N) # If ==N, it means interior
    if senders: # Find optimal solution within a particular "direction" of transfer (just in period 0)
        model.addConstrs(I[0,j,s]==0 for j in range(N) for s in senders) # can't receive any
        receivers = list(set(range(3)) - set(senders))
        model.addConstrs(I[0,r,j]==0 for j in range(N) for r in receivers) # can't send any
        
    # # temporary: to be deleted
    # model.addConstr(I[0,2,0] == 0)
    # model.addConstr(I[0,1,0] == 0)
    # model.addConstr(y[0,1,0] == 11) # post-transfer state of queue 0 at the very start of period 0
    # model.addConstr(y[0,2,0] == 1) # post-transfer state of queue 0 at the very start of period 0
        
    model.params.logtoconsole = 0
    model.optimize()
    
    # Extract output
    Transfers, pre_states, post_states = extract_output(T,N,x0,P,I,y,z)
    
    # Extract trajectories
    trajectories = extract_trajectory(z, T, N, P)
    
    return Transfers, pre_states, post_states, model.objVal, trajectories, model


def solve_instance_discrete(N,h,m,K,lambdas,mus,tau,T,x):
    # model
    P = Model("fluid_control_discrete")
    
    # decision variables
    I = P.addVars(N,N,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="transfer")
    y = P.addVars(N,T+1,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS) # no. customers in each "bin" or interval
    z = P.addVars(N,T+1,lb=0) # max{y_i^t,0}
    Ind = P.addVars(N,N,vtype=GRB.BINARY,name="indicator")
    
    # objective function
    P.setObjective(quicksum(h[i]*tau/T*quicksum(0.5*(z[i,t] + z[i,t-1]) for t in range(1,T+1)) 
                            + quicksum(m[i,j]*I[i,j] for j in range(N))
                            + quicksum(K[i,j]*Ind[i,j] for j in range(N))
                            for i in range(N)), GRB.MINIMIZE)
    # P.setObjective(quicksum(0.5*h[i]/(mus[i]-lambdas[i])*(x[i] + quicksum(I[j,i] for j in range(N)) - quicksum(I[i,j] for j in range(N)))**2 
                            # + quicksum(m[i,j]*I[i,j] for j in range(N)) for i in range(N)), GRB.MINIMIZE)
    
    # constraints on I
    for i in range(N):
        P.addConstr(I[i,i] == 0) # no transfers to itself
        P.addConstr(quicksum(I[i,j] for j in range(N)) <= x[i]) # cap on transfer out
        
    # constraints on y and z
    for i in range(N):
        P.addConstr(y[i,0] == x[i] + quicksum(I[j,i] for j in range(N)) 
                  - quicksum(I[i,j] for j in range(N))) # initial condition y_i^0
        for t in range(1,T+1):
            P.addConstr(y[i,t] == y[i,t-1] + (lambdas[i]-mus[i])*tau/T)
        for t in range(T+1):
            P.addConstr(z[i,t] >= y[i,t])
    
    # constraints on indicator
    for i in range(N):
        P.addConstr(Ind[i,i]==0) # since transfer to self not coded
        for j in range(N):
            P.addGenConstrIndicator(Ind[i,j],0,I[i,j]==0)
        
    P.params.logtoconsole = 0
    P.optimize()
    
    Transfers = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            Transfers[i,j] = I[i,j].x
        
    # Compute final state
    new_x = []
    for i in range(N):
        transfer_out = sum(Transfers[i,:])
        transfer_in = sum(Transfers[:,i])
        new_x.append(x[i] + transfer_in - transfer_out)
        
    return Transfers,new_x,P.objVal
    

def solve_instance_ncvx(N,h,m,K,lambdas,mus,tau,x):
    # model
    P = Model("fluid_control_ncvx")
    
    # decision variables
    I = P.addVars(N,N,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="transfer")
    z = P.addVars(N,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)
    sigma = P.addVars(N,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)
    Ind = P.addVars(N,N,vtype=GRB.BINARY,name="indicator")
    
    # objective function
    P.setObjective(quicksum(h[i]*((x[i] + quicksum(I[j,i] for j in range(N)) 
                                  - quicksum(I[i,j] for j in range(N)))*z[i] 
                                  + 0.5*(lambdas[i]-mus[i])*z[i]**2
                                  )
                            + quicksum(m[i,j]*I[i,j] for j in range(N))
                            + quicksum(K[i,j]*Ind[i,j] for j in range(N))
                            for i in range(N)), GRB.MINIMIZE)
            
    # constraints on I
    for i in range(N):
        P.addConstr(I[i,i] == 0) # no transfers to itself
        P.addConstr(quicksum(I[i,j] for j in range(N)) <= x[i]) # cap on transfers
    
    # constraints on aux. variables
    for i in range(N):
        P.addConstr(sigma[i] == (x[i] + quicksum(I[j,i] for j in range(N)) 
                  - quicksum(I[i,j] for j in range(N))) / (mus[i]-lambdas[i]))
        P.addGenConstrMin(z[i], [sigma[i],tau]) # z_i = min{sigma_i, tau}
    
    # constraints on indicator
    for i in range(N):
        P.addConstr(Ind[i,i]==0) # since transfer to self not coded
        for j in range(N):
            P.addGenConstrIndicator(Ind[i,j],0,I[i,j]==0)
    
    P.params.NonConvex = 2 # Solve non-convex problem
    P.params.logtoconsole = 0
    P.optimize()


    # model output
    Transfers = np.zeros((N,N))
    Indicators = np.zeros((N,N)) # mostly for sanity check
    Sigma = np.zeros((N,N))
    Z = np.zeros(N) # min{sigma_i, tau}, mostly for sanity check
    for i in range(N):
        for j in range(N):
            Transfers[i,j] = I[i,j].x
            Indicators[i,j] = Ind[i,j].x
        Sigma[i] = sigma[i].x
        Z[i] = z[i].x
        
    # Compute final state
    new_x = []
    for i in range(N):
        transfer_out = sum(Transfers[i,:])
        transfer_in = sum(Transfers[:,i])
        new_x.append(x[i] + transfer_in - transfer_out)
        
    return Transfers,Indicators,Sigma,Z,new_x,P.objVal


#%% Solve multi-period, two-queue fluid problem and visualize trajectories.
# plt.close('all')
# # plt.figure(figsize=(15,7))
# # ax = plt.gca()

# # parameters
# N = 2 # number of queues

# ## cost parameters
# lambdas = [0.8, 0.9]
# mus = [1, 1]
# h = [1, 1]
# m = np.ones((N,N))*5
# K = np.ones((N,N))*15

# ## other parameters
# tau = 10
# Ps = [100]
# for P in Ps:
#     # P = 100 # level of discretization in fluid control
#     T = 10 # number of stages in fluid control
#     x = [7,1]
    
#     # # sufficienet conditions for transferring i -> j (just sanity check)
#     # i, j = 0, 1
#     # r = (h[i]-h[j])*T*tau - m[i,j]
#     # cond1 = r > 0
#     # x_thresh = (mus[i]-lambdas[i])*(T*tau-r/2/h[i]) + K[i,j]/r
#     # cond2 = x[i] >= x_thresh
#     # K_ub = (mus[i]-lambdas[i])*r**2 / 2 / h[i]
#     # Istar = x[i] - (mus[i]-lambdas[i])*(m[i,j]+h[j]*tau)/h[i]
    
#     # print("r:",r)
#     # print(cond1)
#     # print("LB:",x_thresh)
#     # print(cond2)
    
#     # Solve fluid control problem
#     sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
#                                         np.array([lambdas]*T),\
#                                         np.array([mus]*T),tau,P,x)
#     print(sol[0], obj)
#     # print("P: {}, Opt. cost: {:.10f}".format(P,obj))
#     # # Plot optimal trajectories
#     # draw_trajectory(x, T, trajectories, ax, title='', labels=["Q1 ($h_1$=" + 
#     #                 str(h[0]) +")","Q2 ($h_2$=" + str(h[1]) + ")"])


#%% Solve multi-period, two-queue fluid problem and generate heatmap of V and opt. action
# Takes a few seconds to run.

plt.close('all')
scale = 10

# parameters
N = 2 # number of queues

## cost parameters
lambdas = [0.55, 0.65]
mus = [1, 1]
h = [1, 1]
m = np.ones((N,N))*2
# K = np.ones((N,N))*1
K = 5

## other parameters
tau = 1 / (scale * (sum(lambdas)+sum(mus)))
# tau = 1 / (sum(lambdas)+sum(mus))
P = 50 # level of discretization in fluid control
# T = 10 # number of stages in fluid control

M = 10 # controls "size" of heatmap
Obj = np.zeros((M+1,M+1))
Pol = np.zeros((M+1,M+1))

n_iter = 0
# for i,x1 in enumerate(range(M)):
for i,I in enumerate(range(M+1)):
    for j,x2 in enumerate(range(M+1)):
        n_iter += 1
            
        x1 = int(I - x2)
        if x1 < 0: continue
        
        # if j < i: continue # Show only the upper/lower triangle
        x = [x1,x2]
        
        # Solve fluid control problem
        emptying_times = []
        for q in range(N): 
            emptying_times.append(x[q] / (mus[q]-lambdas[q]))
        T = max(1,int(np.ceil( max(emptying_times) / tau ))) # prevent 0-period problems
        sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
                                            np.array([lambdas]*T),\
                                            np.array([mus]*T),tau,P,x)
        
        # Check if not transferring is an equally optimal solution in period 0.
        if np.round(sol[0],2).sum() > 0:
            sol2, _, _, obj2, _, _ = solve_fluid_control(T,N,h,m,K,\
                                                        np.array([lambdas]*T),\
                                                        np.array([mus]*T),tau,P,x,
                                                        force_zero=True)
            if np.abs(obj2-obj)/obj * 100 <= 0.001:
                sol = sol2
        
        # # Compute just the holding cost (based on post-transfer states)
        # holding_cost = 0
        # for n in range(N):
        #     for k in range(T):
        #         if post[k][n] >= (mus[n]-lambdas[n])*tau:
        #             holding_cost += h[n]*(post[k][n]*tau + 0.5*(lambdas[n]-mus[n])*tau**2)
        #         else:
        #             holding_cost += h[n]/(2*(mus[n]-lambdas[n]))*post[k][n]**2
                    
        Obj[i,j] = obj
        # Obj[i,j] = holding_cost
        Pol[i,j] = int(sol[0][0,1] - sol[0][1,0]) # only one of the two will be positive when m>0 or K>0.
        # print(x, Pol[i,j])
        print("Progress: {:.1f}%".format(100*n_iter/(M+1)**2))
        
# # Draw heatmap of V
# plt.figure()
# plt.imshow(np.round(Obj),alpha=0.8)
# plt.colorbar()
# plt.xlabel("$x_2$")
# plt.ylabel("$x_1 + x_2$")
# plt.subplots_adjust(left=0.3)
# plt.title("$V_0$")
# plt.tight_layout()
# # plt.savefig("heatmap_fluid_three_queues_12",bbox_inches='tight')

# ## Overlay with values
# for (j,i), label in np.ndenumerate(Obj):
#     plt.text(i,j,round(label),ha='center',va='center',fontsize=14)
# plt.show()


# # Draw heatmap of opt. policy
# fsize = 20
# fig = plt.figure()
# A = np.ma.masked_where(Pol == 0, Pol)
# cmap = mpl.cm.get_cmap("GnBu_r").copy()
# cmap.set_bad(color='white')
# plt.imshow(np.round(A),alpha=0.8,cmap=cmap)
# # plt.colorbar()
# plt.xlabel("$x_2$",fontsize=fsize)
# plt.ylabel("$x_1 + x_2$",fontsize=fsize)
# # plt.title("Fluid policy (continuous-time)",fontsize=fsize)
# plt.xticks(np.arange(0,M+1,1),fontsize=fsize)
# plt.yticks(np.arange(0,M+1,1),fontsize=fsize)
# plt.subplots_adjust(left=0.3)
# plt.grid(alpha=0.1)
# plt.tight_layout()

# ## Overlay with values
# for (j,i), label in np.ndenumerate(A):
#     if round(label)==0:
#         continue # don't plot 0's
#     plt.text(i,j,round(label),ha='center',va='center',fontsize=int(0.9*fsize))
# # plt.savefig("opt_policy_two_queues_symmetric",bbox_inches='tight')
# fig.set_size_inches(10, 10)
# plt.tight_layout()
# plt.show()

np.save("saved_policies/Fluid_policy_identical_light_close_large_scale"+str(scale)+".npy",Pol)

# # # Check if V is non-decreasing
# # ## V is non-decreasing if V(y1,y2) >= V(x1,x2) for any state s.t. y1+y2 > x1+x2
# # fail = 0
# # for x1 in range(M):
# #     for x2 in range(M):
# #         # Iterate over all "cells" for which the total #customers is smaller
# #         # Requires nested loops
# #         for w in range(x1,M):
# #             for v in range(x2,M):
# #                 if Obj[w,v] < Obj[x1,x2]: # violates non-decreasing property
# #                     fail = 1
# #                     break
# #             if fail:
# #                 break
# #         if fail:
# #             break
# #     if fail:
# #         break
    
# # if fail:
# #     print("V is NOT non-decreasing.")
# # else:
# #     print("V IS non-decreasing.")


#%% Generate heatmap (N=2) of optimal actions, for sanity check

# # problem parameters
# N = 2 # number of queues
# h = [1]*N # holding costs at each queue
# m = np.ones((N,N))*5 # variable transfer cost
# K = np.ones((N,N))*10 # setup cost in transfer
# for i in range(N):
#     m[i,i] = 0
#     K[i,i] = 0
# lambdas = [0.9]*N # arrival rate to each queue
# mus = [1]*N # service rate at each queue
# tau = 10 # length of time period

# M = 10 # max. no. customers in system
# T = 100 # Only for solve_instance_discrete: number of intervals per stage
# opt = np.zeros((M+1,M+1)) # optimal decisions
# for i in range(M+1):
#     for j in range(M+1):
#         x = [i,j]
        
#         ### Sanity check using solve_instance_discrete
#         Transfers, new_x, obj = solve_instance_discrete(N,h,m,K,lambdas,mus,tau,T,x)
                    
#         # ### Sanity check using solve_instance_ncvx
#         # Transfers, _, _, _, new_x, _= solve_instance_ncvx(N,h,m,K,lambdas,mus,tau,x)
       
#         a1 = Transfers[0,1]
#         a2 = Transfers[1,0]
#         opt[i,j] = a1 - a2 # Save optimal action 

# print(opt)

        

#%% Solve three-queues (non-convex quadratic program)

# # problem parameters
# N = 3 # number of queues
# # h = [1]*N # holding costs at each queue
# h = [2,2,1]
# m = np.ones((N,N))*1 # variable transfer cost
# K = np.ones((N,N))*3 # setup cost in transfer
# for i in range(N):
#     m[i,i] = 0
#     K[i,i] = 0
# lambdas = [0.1]*N # arrival rate to each queue
# mus = [1]*N # service rate at each queue
# tau = 10 # length of time period
# x = [6,3,6]

# Transfers, _, _, _, new_x,obj = solve_instance_ncvx(N,h,m,K,lambdas,mus,tau,x)
# print("\nSolving non-convex program:")
# print(obj)
# print("Initial state:",x)
# print("Final state:",new_x)
# # print("Final state:",np.round(new_x))

# # M = 10 # max. no. customers in system
# # opt = np.zeros((M+1,M+1)) # optimal decisions
# # for i in range(M+1):
# #     for j in range(M+1):
# #         x = [i,j]
# #         Transfers, _, _, _, new_x = solve_instance_ncvx(N,h,m,K,lambdas,mus,tau,x)
# #         a1 = round(Transfers[0,1])
# #         a2 = round(Transfers[1,0])
# #         opt[i,j] = a1 - a2 # Save optimal action 



#%% Solve transfer optimization/optimal transport problem (N queues)

# m = np.array([[0,  1,  1.1,5],
#               [1,  0,  1.2,4],
#               [1.1,1.2,0,  4.1],
#               [5,  4,  4.1,0]]) # must satisfy triangular inequality
# N = m.shape[0]
# Qs = np.arange(0,N,1)

# ### pairwise uniform setup costs
# K = 3
    
# x = np.array([1,1,10,5]) # pre-transfer state
# y = np.array([5,5,4,3]) # post-transfer state

# ### Transfer optimization problem as an optimal transport problem
# setup = 0 # if 1, include setup cost in obj
# opt_transport = 1 # if 1, solve as an optimal transport problem

# model = Model("transfer_opt")

# I = model.addVars(N,N,lb=0,ub=GRB.INFINITY,name="transfer")
# Ind = model.addVars(N,N,vtype=GRB.BINARY,name="indicator") # pairwise setup cost

# if setup:
#     model.setObjective(quicksum(m[i,j]*I[i,j] + K*Ind[i,j] for i in range(N) for j in range(N)))
# else:
#     model.setObjective(quicksum(m[i,j]*I[i,j] for i in range(N) for j in range(N)))


# if opt_transport:
#     ### optimal transport constraints
#     D = Qs[(y-x) >= 0]
#     S = Qs[(y-x) < 0]
#     model.addConstrs(quicksum(I[i,j] for i in S) == y[j]-x[j] for j in D)
#     model.addConstrs(quicksum(I[i,j] for j in D) == x[i]-y[i] for i in S)
# else:
#     ### transfer opt constraints
#     model.addConstrs(quicksum(I[i,j] for i in range(N)) - quicksum(I[j,l] for l in range(N)) == y[j]-x[j] for j in range(N))


# # setup cost constraints
# for i in range(N):
#     for j in range(N):
#         model.addGenConstrIndicator(Ind[i,j],0,I[i,j]==0)
        

# ### force decisions
# # dec = np.array( [[0, 0, 0, 0],
# #                   [0, 0, 0, 0],
# #                   [2, 4, 0, 0],
# #                   [2, 0, 0, 0]])
# # for i in range(N):
# #     for j in range(N):
# #         model.addConstr(I[i,j]==dec[i,j])

        
# model.params.logtoconsole = 0
# model.optimize()

# sol = np.zeros((N,N))
# for i in range(N):
#     for j in range(N):
#         sol[i,j] = I[i,j].x
# print("Opt. obj:",model.objVal)
# print("Setup:", setup, "Opt. transport:", opt_transport, "\n", sol)


#%% Solve three-queues

# # problem parameters
# N = 3 # number of queues
# h = [1,1,1] # holding costs at each queue
# # m = np.array([[0,3,4],
# #               [3,0,2],
# #               [4,2,0]]) # must satisfy triangular inequality
# m = np.array([[0,1,1],
#               [1,0,1],
#               [1,1,0]]) # must satisfy triangular inequality

# # ### queue pair-specific setup costs
# K = np.ones((N,N))*3 # setup cost in transfer
# for i in range(N):
#     K[i,i] = 0

# ### queue-specific setup costs
# # K = [5, 5, 5]
    
# lambdas = [0.9, 0.9, 0.9] # arrival rate to each queue
# mus = [1]*N # service rate at each queue
# tau = 10 # length of time period

# M = 30
# Pol = -1*np.ones((M+1,M+1)) # x-axis is x1, y-axis is x2. Total conserved at M customers.
#                                   # Save total #customers transferred to queue 1.
# for x1 in range(M+1):
#     for x3 in range((M+1)-x1):
#         x2 = (M-x1)-x3
#         x = [x1,x2,x3]
    
#         T = 20 # number of periods in fluid control
#         P = 100 # number of intervals per period (discretization)
        
#         # Get cost of optimal action under K=0
#         sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
#                                                     np.array([lambdas]*T),\
#                                                     np.array([mus]*T),tau,P,x)
        
#         print("x2: {}, x3: {}, net inflow: {:.2f}, obj = {:.5f}".format(x[1],x[2],sol[0][1,0]+sol[0][2,0],obj))
#         # print("\nSolving discretized fluid control:")
#         # print(obj)
#         # print("Initial state:\n",np.round(pre,2))
#         # print("Final state:\n",np.round(post,2))

#         Pol[x2,x1] = sol[0][1,0] + sol[0][2,0]

# # # Draw heatmap of opt. policy
# # plt.close('all')
# # plt.figure()
# # plt.imshow(Pol,alpha=0.8)
# # plt.colorbar()
# # plt.ylabel("$x_2$",fontsize=14)
# # plt.xlabel("$x_1$",fontsize=14)
# # plt.title("$h=[1,1,1]$ ($n$ constant at 10)",fontsize=14)
# # plt.xticks(fontsize=14)
# # plt.yticks(fontsize=14)

# # ## Overlay with values
# # for (j,i), label in np.ndenumerate(Pol):
# #     plt.text(i,j,np.round(label,1),ha='center',va='center')
# # plt.savefig("opt_policy_3_queues_nonidentical_h_symmetric_m_heavytraffic",bbox_inches='tight')
# # plt.show()


# # # Test only a single initial condition, instead of the nested for-loops
# # x = [0,0,10]
# # T = 10 # number of stages in fluid control
# # P = 100 # number of intervals per period (discretization)

# # # Get cost of optimal action under K=0
# # sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
# #                                             np.array([lambdas]*T),\
# #                                             np.array([mus]*T),tau,P,x)
# # print("x2: {}, x3: {}, net inflow: {:.2f}, obj = {:.5f}".format(x[1],x[2],sol[0][1,0]+sol[0][2,0],obj))


#%% Generate matrix for heatmap for three-queues (for above)
# start = time.time()

# # problem parameters
# N = 3 # number of queues
# # h = [1]*N # holding costs at each queue
# h = [2,1.5,1]
# m = np.ones((N,N))*1 # variable transfer cost
# K = np.ones((N,N))*3 # setup cost in transfer
# for i in range(N):
#     m[i,i] = 0
#     K[i,i] = 0
# lambdas = [0.5]*N # arrival rate to each queue
# mus = [1]*N # service rate at each queue
# tau = 10 # length of time period
# T = 100

# M = 12 # fixed no. customers in system
# opt_dec = np.empty((M+1,M+1,M+1,N))
# state_space = itertools.product(*[range(M+1)]*(N-1)) # All initial states
# end_states = [] # stores the state of the cheapest queue after transfer
# for x in state_space:
#     if sum(x) > M: 
#         # Infeasible state
#         opt_dec[x][:] = np.nan
#         continue 
    
#     x = x + (M - sum(x),) # Augment state to include queue 3
    
#     # Get cost of optimal action under K=0
#     Transfers, new_x, obj = solve_instance_discrete(N,h,m,K,lambdas,mus,tau,T,x)

#     # Get optimal actions as a 3d vector (hardcoded)
#     a = []
#     a.append(Transfers[0,1] - Transfers[1,0]) # 1->2
#     a.append(Transfers[0,2] - Transfers[2,0] )# 1->3
#     a.append(Transfers[1,2] - Transfers[2,1] )# 2->3
#     opt_dec[x] = a # store optimal actions
    
#     final_x = tuple(np.round(new_x,5))
#     end_states.append(final_x[-1])
    
#     # Compute total number of customers moved
#     tot = round(sum(np.abs(a)))
    
#     # Check if each queue is on the receiving/sending end
#     receiver = {}
#     sender = {}
#     for i in range(N):
#         if sum(np.round(Transfers[:,i])) > 0:
#             receiver[i] = 1
#         else:
#             receiver[i] = 0
#         if sum(np.round(Transfers[i,:])) > 0:
#             sender[i] = 1
#         else:
#             sender[i] = 0
#     num_receivers = sum(receiver.values())
#     num_senders = sum(sender.values())
    
#     # print(x,'-->',final_x,'Total num:',tot)
#     # if tot==0:
#     #     print(final_x)
#     if sender[0] and (sender[1]==0) and (receiver[1]==0) and receiver[2]:
#         print(x,'-->',final_x,'Total num:',tot)
        
#     # # Check if a queue is both sending and receiving
#     # if (sender[0] and receiver[0]) or (sender[1] and receiver[1]) or (sender[2] and receiver[2]):
#     #     print("Got it!")
    
# end = time.time()
# print(end-start)



#%% Plot heatmap for three-queues

# # Do: x1 (y-axis) vs. x3 (x-axis). Overlay with numbers.
# plt.close('all')

# matrix12 = np.zeros((M+1,M+1))
# matrix13 = np.zeros((M+1,M+1))
# matrix23 = np.zeros((M+1,M+1))
# for x1 in range(M+1):
#     for x2 in range(0,M-x1+1,1):
#         x3 = M - x1 - x2
#         matrix12[x1,x2] = opt_dec[x1,x2,x3,0] # Get action 1 -> 2
#         matrix13[x1,x3] = opt_dec[x1,x2,x3,1] # Get action 1 -> 3
#         matrix23[x2,x3] = opt_dec[x1,x2,x3,2] # Get action 2 -> 3

# ### Matrix 1->2
# plt.figure()
# plt.imshow(np.round(matrix12),alpha=0.8)
# plt.colorbar()
# plt.ylabel("x1")
# plt.xlabel("x2")
# plt.title("Asymmetric problem (1->2)")
# plt.subplots_adjust(left=0.3)
# plt.savefig("heatmap_fluid_three_queues_12")

# # Overlay with values
# for (j,i), label in np.ndenumerate(matrix12):
#     plt.text(i,j,round(label),ha='center',va='center')
# plt.show()

# ### Matrix 1->3
# plt.figure()
# plt.imshow(np.round(matrix13),alpha=0.8)
# plt.colorbar()
# plt.ylabel("x1")
# plt.xlabel("x3")
# plt.title("Asymmetric problem (1->3)")
# plt.subplots_adjust(left=0.3)
# plt.savefig("heatmap_fluid_three_queues_13")

# # Overlay with values
# for (j,i), label in np.ndenumerate(matrix13):
#     plt.text(i,j,round(label),ha='center',va='center')
# plt.show()

# ### Matrix 2->3
# plt.figure()
# plt.imshow(np.round(matrix23),alpha=0.8)
# plt.colorbar()
# plt.ylabel("x2")
# plt.xlabel("x3")
# plt.title("Asymmetric problem (2->3)")
# plt.subplots_adjust(left=0.3)
# plt.savefig("heatmap_fluid_three_queues_23")

# # Overlay with values
# for (j,i), label in np.ndenumerate(matrix23):
#     plt.text(i,j,round(label),ha='center',va='center')
# plt.show()



#%% Investigate general structure in the 3-queue problem. Visualize trajectories.

# # parameters
# N = 3 # number of queues

# ## cost parameters
# lambdas = [0.9, 0.9, 0.9]
# mus = [1, 1, 1]
# h = [3, 3, 3]
# m = np.ones((N,N))*5 
# K = np.ones((N,N))*0

# ## other parameters
# tau = 10
# P = 100 # level of discretization per period
# T = 2 # number of periods
# x = [1,6.5,7.5] # initial state

# # xs = [[2,2,11], \
# #       [2,3,10], \
# #       [2,4,9], \
# #       [2,5,8], \
# #       [2,6,7], \
# #       [2,7,6], \
# #       [2,8,5], \
# #       [2,9,4],\
# #       [2,10,3],\
# #       [2,11,2]]

# # xs = [[0,2,13], \
# #       [0,3,12], \
# #       [0,4,11], \
# #       [0,5,10], \
# #       [0,6,9], \
# #       [0,7,8], \
# #       [0,8,7], \
# #       [0,9,6], \
# #       [0,10,5],\
# #       [0,11,4],\
# #       [0,12,3],\
# #       [0,13,2]]
    
# plt.close('all')
# # Solve fluid control problem
# sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
#                                     np.array([lambdas]*T),\
#                                     np.array([mus]*T),tau,P,x)
# print("Pre-transfer: {:.1f}, Post-transfer: {:.2f}, Opt. cost: {:.1f}".format(x[0], post[0][0], obj))
# # print(sol[0])

# # Plot optimal trajectories

# plt.figure(figsize=(15,7))
# ax = plt.gca()
# draw_trajectory(x, T, trajectories, ax, "$x^0$="+str(x), ["$x_1$", "$x_2$", "$x_3$"])



#%% Solve three-queues for different total numbers (M)

# # problem parameters
# N = 3 # number of queues
# h = [1,3,1] # holding costs at each queue
# # m = np.array([[0,3,4],
# #               [3,0,2],
# #               [4,2,0]]) # must satisfy triangular inequality
# m = np.array([[0,1,1],
#               [1,0,1],
#               [1,1,0]]) # must satisfy triangular inequality

# # # ### queue pair-specific setup costs
# # K = np.ones((N,N))*3 # setup cost in transfer
# # for i in range(N):
# #     K[i,i] = 0

# ### queue-specific setup costs
# K = [5, 5, 5]
    
# lambdas = [0.9, 0.9, 0.9] # arrival rate to each queue
# mus = [1]*N # service rate at each queue
# tau = 10 # length of time period

# Ms = np.arange(0,51,1)
# # Pol_even = np.zeros((len(Ms),len(Ms))) # Stores combined number transferred to queue 1
# Pol_concen = np.zeros((len(Ms),len(Ms)))
# for M in Ms:
#     T = int(max(10,M)) # should be long enough to have stationary fluid policies
#     P = 100 # number of intervals per period (discretization)
#     for x1 in range(M+1):
#         # # Once x1 is decided, evenly distribute remaining customers
#         # x2 = int((M-x1)/2) 
#         # x3 = int(M - x1 - x2)
#         # x = [x1,x2,x3]
#         # sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
#         #                                             np.array([lambdas]*T),\
#         #                                             np.array([mus]*T),tau,P,x)
#         # Pol_even[M,x1] = sol[0][1,0] + sol[0][2,0] # combined number transferred to queue 1 
        
#         # Once x1 is decided, put all remaining customers in one queue
#         x2 = M-x1
#         x3 = 0
#         x = [x1,x2,x3]
#         sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
#                                                     np.array([lambdas]*T),\
#                                                     np.array([mus]*T),tau,P,x)
#         Pol_concen[M,x1] = sol[0][1,0] + sol[0][2,0] # combined number transferred to queue 1 


#%% Draw heatmap for above
# from helper_functions import save_heatmap_2D
# plt.close("all")
# save_heatmap_2D([], Pol_even[:51,:51], [])
# save_heatmap_2D([], Pol_concen, [])

#%% Is kappa(x2-x1) >= kappa(S(x2)-S(x1)) ?

# for trial in range(100):
#     # problem parameters
#     N = 3 # number of queues
#     h = [1,3,2] # holding costs at each queue
#     m = np.array([[0,3,4],
#                   [3,0,2],
#                   [4,2,0]]) # must satisfy triangular inequality
#     # m = np.array([[0,1,1],
#     #               [1,0,1],
#     #               [1,1,0]]) # must satisfy triangular inequality
    
#     # ### queue pair-specific setup costs
#     # K = np.ones((N,N))*3 # setup cost in transfer
#     # for i in range(N):
#     #     K[i,i] = 0
#     # K = np.array([[0,3,4],
#     #               [3,0,2],
#     #               [4,2,0]])
    
#     ### queue-specific setup costs
#     K = [7, 3, 5]
        
#     lambdas = [0.5, 0.9, 0.95] # arrival rate to each queue
#     mus = [1]*N # service rate at each queue
#     tau = 10 # length of time period
    
#     n = 10 # total number of customers
#     T = 10 # number of periods
#     P = 100 # level of discretization in fluid control problem
    
#     # Two vectors from simplex(n)
#     x1 = np.random.rand(N)
#     x1 = x1 / sum(x1) * n # components will sum to n
#     x2 = np.random.rand(N)
#     x2 = x2 / sum(x2) * n # components will sum to n
    
#     # Get opt post-transfer states from x1 and x2
#     sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
#                                                 np.array([lambdas]*T),\
#                                                 np.array([mus]*T),tau,P,x1)
#     S_x1 = post[0]
    
#     sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
#                                                 np.array([lambdas]*T),\
#                                                 np.array([mus]*T),tau,P,x2)
#     S_x2 = post[0]
    
#     # Compare setup costs (queue-specific setup costs)
#     setup_cost = lambda x: 1*(np.round(x,5)>0) @ K
    
#     print("kappa(x2-x1):",setup_cost(x1-x2))
#     print("kappa(S_x2-S_x1):",setup_cost(S_x1-S_x2))
#     condition_met = setup_cost(x1-x2) >= setup_cost(S_x1-S_x2)
#     print("Condition true?",condition_met)
#     if condition_met==False:
#         pass

#%% Is \Sigma(n) convex?

n = 5 # tot. #customers. Assume fixed.

# problem parameters
h = [1,1,1] # holding costs at each queue
N = len(h) # number of queues
m = np.array([[0,1,1],
              [1,0,1],
              [1,1,0]])*2 # must satisfy triangular inequality

# queue pair-specific setup costs
# K = np.array([[0,1,1],
#               [1,0,1],
#               [1,1,0]])*5 # must satisfy triangular inequality

# queue-specific (individual) setup costs
# K = np.array([5, 5, 5])

### System-wide joint setup cost
K = 5

lambdas = [0.75, 0.75, 0.75] # arrival rate to each queue
mus = [1,1,1] # service rate at each queue
tau = 5 # length of time period
P = 1000 # level of discretization
T = n # proportionally long horizon (num periods; in paper, this is M)

np.random.seed(33)
n_trials = 10000

X = np.zeros((n_trials,N)) # Collect all randomly sampled initial states
Y = np.zeros((n_trials,N)) # Collect optimal post-transfer states, if transferring is optimal
tags = np.zeros(n_trials) # 1 if not transferring is optimal; 0 if transferring optimal; -1 if indifferent; -2 if transferring and not transferring are only slightly different
i = -1

minmin = float('inf')
orderupto = []

def temporary1(minmin, orderupto):
    """Find order up to point(s) and minimum threshold for sending at queue 1"""
    """Originally written for identical queues"""
    if len(senders) == 2:
        if (1 in senders) and (2 in senders):
        # print(np.round(x,2))
        # print(np.round(sol[0],2))
        # print(np.round(post[0],2))
            if np.argmin(x[senders]) == 1:
                if x[1] < minmin:
                    minmin = x[1]
                print("Minmin:", minmin)
    if len(receivers) == 2:
        if (1 in receivers) and (0 in receivers):
            if np.argmax(x[receivers]) == 1:
                orderupto.append(np.round(post[0][1],2))
                print("Order up tos:", set(orderupto))  
    print('====================')
    return minmin, orderupto


def temporary2(x, post, obj, lambdas, mus, tau):
    """Check if the queue with the largest "gap" is never a receiver"""
    receivers = np.where(np.round(post[0]-x,2)>0)[0]
    max_idx = np.argmax(x + (np.array(lambdas) - np.array(mus))*tau)
    condition = max_idx in receivers # must be false
    if condition == True:
        raise Exception("Condition not true")
        

def temporary3(x, lambdas, mus, tau):
    """Check if queues with less than (lam-mu)tau customers are never senders"""
    senders = np.where(np.round(post[0]-pre[0],2)<0)[0]
    candidates = set(np.where(x < (np.array(mus)-np.array(lambdas))*tau)[0])
    for c in candidates:
        if c in senders:
            raise Exception("Condition not true")


def temporary4(x, lambdas, mus, tau):
    """Check if queues with more than (lam-mu)tau customers are never receivers"""
    receivers = np.where(np.round(post[0]-x,2)>=0)[0] # includes non-participants
    candidates = set(np.where(x >= (np.array(mus)-np.array(lambdas))*tau)[0])
    for c in candidates:
        if c in receivers:
            raise Exception("Condition not true")


def temporary5(x, h, lambdas, mus, tau):
    """Check if queue with smallest hi never sends customers except to prevent idleness at other queues"""
    senders = np.where(np.round(post[0]-pre[0],2)<0)[0]
    q = np.argmin(np.array(h))
    if q in senders:
        condition = False
        other_qs = set(range(len(h))) - set([q])
        for queue in other_qs:
            condition = condition or (x[queue] < (mus[queue]-lambdas[queue])*tau)
        if condition == False:
            raise Exception("Condition not true")


def temporary6(x, lambdas, mus, tau, no_transfer):
    """Check if it is optimal to transfer whenever there is at least
    one queue with less than (lam-mu)tau"""
    tmp = np.where(x < (np.array(mus) - np.array(lambdas))*tau)[0]
    if len(tmp) > 0:
        if no_transfer == 0: # i.e., transferring is optimal
            pass
        else:
            raise Exception("Condition not true")


def temporary7(x, post, lambdas, mus, tau, h):
    """Queue i with i=argmax h_i*mu_i is never a receiver when x_i >= idle point"""
    i = np.argmax(np.array(h)*np.array(mus)/np.array(lambdas))
    if x[i] >= np.round((mus[i]-lambdas[i])*tau,4):
        receivers = np.where(np.round(post[0]-x,2)>0)[0]
        if i in receivers:
            raise Exception("Condition not true")


def sufficient_con_check(K2, sol, x, K, lambdas, mus, tau):
    """
    Sufficient condition for monotonicity of the no-transfer region.
    V_k^K2(f(x)) - V_k^K1(f(x)) <= K2 - K1 + V_k^K2(f(y)) - V_k^K2(f(y)) for all y
    """
    func = lambda x: x + (np.array(lambdas)-np.array(mus))*tau # transition function
    
    if np.round(sol,2).sum() > 0:
        pass
    else:
        # Solve problem with one less period using f(x) as initial state
        _, _, _, obj_K1, _, _ = solve_fluid_control(T-1,N,h,m,K,\
                                                    np.array([lambdas]*(T-1)),\
                                                    np.array([mus]*(T-1)),\
                                                    tau,P,func(x))
        _, _, _, obj_K2, _, _ = solve_fluid_control(T-1,N,h,m,K2,\
                                                    np.array([lambdas]*(T-1)),\
                                                    np.array([mus]*(T-1)),\
                                                    tau,P,func(x))
        LHS = np.round(obj_K2 - obj_K1,4)
        
        # Solve problem with one less period using random y and f(y) as the initial state
        y = np.random.uniform(size=len(x)) 
        y = y / sum(y) * sum(x)
        _, _, _, obj_K1_y, _, _ = solve_fluid_control(T-1,N,h,m,K,\
                                                    np.array([lambdas]*(T-1)),\
                                                    np.array([mus]*(T-1)),\
                                                    tau,P,func(y))
        _, _, _, obj_K2_y, _, _ = solve_fluid_control(T-1,N,h,m,K2,\
                                                    np.array([lambdas]*(T-1)),\
                                                    np.array([mus]*(T-1)),\
                                                    tau,P,func(y))
        RHS = np.round(K2 - K + obj_K2_y - obj_K1_y,4)
        
        if LHS <= RHS:
            pass
        else:
            print(np.round(x,2))
            print(np.round(y,2))
            raise Exception("Condition not true")
        

while i < n_trials - 1:
    x = np.random.uniform(size=N) # Randomly sample initial state
    x = x / sum(x) * n # normalize and ensure sum=n
    # if i>=n_trials-1000:
    # if x[0] >= 0.5:
    #     continue # sample only within certain region 
            
    i += 1
    
    sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
                                                        np.array([lambdas]*T),\
                                                        np.array([mus]*T),tau,P,x)
    
    
    # See if not transferring is an equally optimal solution in period 0.
    obj2 = 0
    if np.round(sol[0],2).sum() > 0:
        sol2, _, _, obj2, _, _ = solve_fluid_control(T,N,h,m,K,\
                                                        np.array([lambdas]*T),\
                                                        np.array([mus]*T),tau,P,x,
                                                        force_zero=True)
            
    if np.abs(obj2-obj)/obj * 100 <= 0.001:
        sol = sol2
    
    # sufficient_con_check(K+5, sol[0], x, K, lambdas, mus, tau)
                
    # See if there is an equally optimal boundary point
    objb = 0
    # # boundaries = np.round((np.array(mus) - np.array(lambdas))*tau,2) # known a priori
    # boundaries = np.array([1,1,1])
    # if (np.round(post[0],2) > boundaries).sum() == N: # criterion for an interior point
    #     solb, _, postb, objb, _, _ = solve_fluid_control(T,N,h,m,K,\
    #                                                     np.array([lambdas]*T),\
    #                                                     np.array([mus]*T),tau,P,x,\
    #                                                     force_bound=True)
    
    # # See if there is an equally optimal interior point (heuristic only for pairwise constant setup costs)
    obji = 0
    # boundaries = np.round((np.array(mus) - np.array(lambdas))*tau,2) + 0.1
    # if (np.round(post[0],2) > boundaries).sum() == N: # criterion for an interior point
    #     soli, _, posti, obji, _, _ = solve_fluid_control(T,N,h,m,K,\
    #                                                     np.array([lambdas]*T),\
    #                                                     np.array([mus]*T),tau,P,x,\
    #                                                     force_interior=True)

    # # # See if transferring is an equally optimal solution in period 0.
    # # obj3 = 0
    # # if np.round(sol[0],2).sum() == 0:
    # #     sol3, _, _, obj3, _, _ = solve_fluid_control(T,N,h,m,K,\
    # #                                                     np.array([lambdas]*T),\
    # #                                                     np.array([mus]*T),tau,P,x,
    # #                                                     force_pos=True)
    
    # We use 0.001%
    if (100*np.abs(obj-obj2)/obj<=0.001):# or (100*np.abs(obj-obj3)/obj<=0.01):
        no_transfer = -1 # indifferent   
        post[0] = x
    elif (100*np.abs(obj-objb)/obj<=0.01):
        no_transfer = 0 # transferring is optimal
        post = postb # found an equally optimal boundary point
        # print(np.round(x,2))
        # print(np.round(post[0],2))
        # print("=============")
    elif (100*np.abs(obj-obji)/obj<=0.001):
        no_transfer = 0 # transferring is optimal
        post = posti # found an equally optimal boundary point
    else:
        # Find out if transferring was optimal in period 0
        no_transfer = 0
        if np.round(sol[0],2).sum() == 0:   
            no_transfer = 1
        else:
            # Check if there is an equally optimal solution
            # senders = np.where(np.round(post[0]-pre[0],2)<0)[0]
            # receivers = list(set(range(N)) - set(senders)) # includes non-participants
            # temporary7(x, post, lambdas, mus, tau, h)
            
            # print(np.round(x,2))
            # print(np.round(post[0],2))
            # print("=============")
            # raise Exception("Condition not true")
            pass
    
    # # temporary6(x, lambdas, mus, tau, no_transfer)
            
    # Collect
    X[i,:] = x
    Y[i,:] = post[0]
    tags[i] = no_transfer
    
    if (i+1)%50==0:
        print("Progress:", np.round((i+1)/n_trials*100,1), "%")
        

#%% Visualize no-transfer region, three queues (for above)
draw = True
if draw:
    size = 10
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    # First plot tag 1 (not transferring is optimal)
    ax.scatter(X[tags==1,0],X[tags==1,1],X[tags==1,2],c="blue",label=r"$x \in \Sigma(\cdot)$",s=size,alpha=0.03)
    
    # Then plot tag 0 (transferring is optimal)
    ax.scatter(X[tags==0,0],X[tags==0,1],X[tags==0,2],c="red",label=r"$x \notin \Sigma(\cdot)$",s=size)
    
    # Then plot tag -1 (indifferent); color the same as "no transfer"
    ax.scatter(X[tags==-1,0],X[tags==-1,1],X[tags==-1,2],c="blue",s=size,alpha=0.03)#,label="Indifferent",s=size)
    
    # # Then plot tag -2 (transferring vs. not transferring are only slightly different)
    # ax.scatter(X[tags==-2,0],X[tags==-2,1],X[tags==-2,2],c="orange",label="Nearly indifferent",s=size)
    
    # # No-transfer region is at least as large as this
    # con = np.array([True] * X.shape[0])
    # for i in range(N):
    #     con = con * (X[:,i] > 2*(mus[i]-lambdas[i])*tau)
    # ax.scatter(X[con,0],X[con,1],X[con,2],c="orange",s=size)
    
    # Then plot optimal post-transfer states (if transferring was optimal)
    ax.scatter(Y[tags==0,0],Y[tags==0,1],Y[tags==0,2],c="lime",label=r"$\pi^*(x)$ (target state)",s=size)
    
    fsize = 20
    ax.set_xlim3d(0,n)
    ax.set_ylim3d(0,n)
    ax.set_zlim3d(0,n)
    ax.set_xlabel("$x_1$",fontsize=fsize)
    ax.set_ylabel("$x_2$",fontsize=fsize)
    ax.set_zlabel("$x_3$",fontsize=fsize)
    ax.tick_params(axis='both', labelsize=fsize)
    ax.view_init(azim=45, elev=30)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.legend(fontsize=fsize)
    
    # # Shows directions of optimal transfer (optional)
    # quiveropts = dict(color='black', alpha=0.5, linewidths=(1.5,), \
    #                   length=0.6, arrow_length_ratio=0.33, normalize=True)
    # ax.quiver(X[:,0],X[:,1],X[:,2],Y[:,0]-X[:,0],Y[:,1]-X[:,1],Y[:,2]-X[:,2],\
    #           **quiveropts)
    fig.set_size_inches(10, 10)
    plt.tight_layout()
    
    
#%% "Vector field" visualization (only works for m_ij>0, K=0 case)
# n = 10 # tot. #customers. Assume fixed.

# # problem parameters
# h = [3,3,3] # holding costs at each queue
# N = len(h) # number of queues
# m = np.array([[0,2,2],
#               [2,0,2],
#               [2,2,0]]) # must satisfy triangular inequality
# K = 0

# lambdas = [0.9, 0.9, 0.9] # arrival rate to each queue
# mus = [1,1,1] # service rate at each queue
# tau = 10 # length of time period
# P = 100 # level of discretization
# T = 10 # proportionally long horizon

# num = 21
# x1 = np.linspace(0,n,num)
# x2 = np.linspace(0,n,num)

# X = np.zeros((num**2,N)) # Collect all randomly sampled initial states
# Y = np.zeros((num**2,N)) # Collect optimal post-transfer states, if transferring is optimal
# tags = np.zeros(num**2) # 1 if not transferring is optimal; 0 if transferring optimal; -1 if indifferent; -2 if transferring and not transferring are only slightly different
# w = -1
# for i in range(num):
#     for j in range(num):
#         x = np.array([x1[i], x2[j], n - x1[i] - x2[j]])
#         if sum(x[:-1]) > n:
#             continue
#         w += 1
        
#         sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
#                                                             np.array([lambdas]*T),\
#                                                             np.array([mus]*T),tau,P,x)
        
        
#         # See if not transferring is an equally optimal solution in period 0.
#         obj2 = 0
#         if np.round(sol[0],2).sum() > 0:
#             sol2, _, _, obj2, _, _ = solve_fluid_control(T,N,h,m,K,\
#                                                             np.array([lambdas]*T),\
#                                                             np.array([mus]*T),tau,P,x,
#                                                             force_zero=True)

#         # See if there is an equally optimal boundary point
#         objb = 0
#         boundaries = np.round((np.array(mus) - np.array(lambdas))*tau,2) # known a priori
#         if (np.round(post[0],2) > boundaries).sum() == N: # criterion for an interior point
#             solb, _, postb, objb, _, _ = solve_fluid_control(T,N,h,m,K,\
#                                                             np.array([lambdas]*T),\
#                                                             np.array([mus]*T),tau,P,x,\
#                                                             force_bound=True)

#         # We use 0.01% b/c that's the default opt. gap in Gurobi.
#         if (100*np.abs(obj-obj2)/obj<=0.001):
#             no_transfer = -1 # indifferent   
#             post[0] = x
#         elif (100*np.abs(obj-objb)/obj<=0.001):
#             no_transfer = 0 # transferring is optimal
#             post = postb # found an equally optimal boundary point
#         else:
#             # Find out if transferring was optimal in period 0
#             no_transfer = 0
#             if np.round(sol[0],2).sum() == 0:   
#                 no_transfer = 1
        
#         # Collect
#         X[w,:] = x
#         Y[w,:] = post[0]
#         tags[w] = no_transfer
        
# condition = ~np.all(X==0, axis=1)
# X = X[condition]
# Y = Y[condition]
# tags = tags[condition]

        
#%% Prove/disprove the "sufficient condition" to extend same structure to other models of setup cost


# n = 10 # tot. #customers. Assume fixed.
# N = 3 # number of queues

# # problem parameters
# h = [1,1,1] # holding costs at each queue
# m = np.array([[0,2,2],
#               [2,0,2],
#               [2,2,0]]) # must satisfy triangular inequality

# ## queue pair-specific setup costs
# K = np.array([[0,5,5],
#               [5,0,5],
#               [5,5,0]]) # must satisfy triangular inequality

# lambdas = [0.9, 0.7, 0.5] # arrival rate to each queue
# mus = [1]*N # service rate at each queue
# tau = 10 # length of time period
# P = 100 # level of discretization
# T = n # proportionally long horizon


# def check_directions(diff_y1, diff_y2, silent=0):
#     """
#     Check if optimal directions of transfers from y1 and y2 match.
#     """
#     if silent:
#         pass
#     else:
#         print("From y1:", diff_y1)
#         print("From y2:", diff_y2)
        
#     sign_y1 = np.where(diff_y1<0)[0]
#     sign_y2 = np.where(diff_y2<0)[0]
#     if len(sign_y1) != len(sign_y2): # number of senders not matching
#         if (len(sign_y1)==0) or (len(sign_y2)==0):
#             pass
#         else:
#             return 0
#     else:
#         if set(sign_y1)==set(sign_y2):
#             pass
#         else:
#             return 0
#     return 1
        

# n_trials = 10000
# for t in range(n_trials):
#     # Sample an arbitrary state
#     x = np.random.uniform(size=N) # Randomly sample initial state
#     x = x / sum(x) * n # normalize and ensure sum=n
    
#     # Fix a "direction"
#     num_senders = np.random.choice([1,2]) # w/ equal prob., choose number of senders
#     senders = list(np.random.choice(list(range(N)), size=num_senders, replace=False)) # randomly select senders
#     receivers = list(set(range(3)) - set(senders))
    
#     # Find minimum point along this direction from x
#     sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
#                                                         np.array([lambdas]*T),\
#                                                         np.array([mus]*T),tau,P,x,\
#                                                         senders=senders)
    
#     # Sample two states y1 and y2 in this direction
#     x = np.copy(pre[0])
#     Sp = np.copy(post[0])
    
#     diff = Sp - x
#     y1 = np.copy(x)
#     y2 = np.copy(x)
#     for r in receivers:
#         y1[r] = x[r] + np.random.sample()*diff[r] # add a random amount s.t. y1r <= Spr
#         y2[r] = x[r] + np.random.sample()*diff[r]
#     budget1 = sum([y1[r]-x[r] for r in receivers])
#     budget2 = sum([y2[r]-x[r] for r in receivers])
#     actual_senders = list(np.where(diff<0)[0]) # could differ from the original set "senders"
#     for s in actual_senders: 
#         y1[s] = max(0,x[s] - budget1)
#         budget1 -= x[s]
#         if budget1<=1e-5:
#             break
#     for s in actual_senders: 
#         y2[s] = max(0,x[s] - budget2)
#         budget2 -= x[s]
#         if budget2<=1e-5:
#             break
    
#     if (np.round(sum(x),2)!=n) or (np.round(sum(Sp),2)!=n) or (np.round(sum(y1),2)!=n) or (np.round(sum(y2),2)!=n):
#         raise Exception("Total number of customers not matching!")
    
#     # Compute globally optimal solutions from y1 and y2 (without directional restriction)
#     _, pre_y1, post_y1, obj_y1, _, _ = solve_fluid_control(T,N,h,m,K,\
#                                                         np.array([lambdas]*T),\
#                                                         np.array([mus]*T),tau,P,y1)
#     _, pre_y2, post_y2, obj_y2, _, _ = solve_fluid_control(T,N,h,m,K,\
#                                                         np.array([lambdas]*T),\
#                                                         np.array([mus]*T),tau,P,y2)
    
#     # Check if "optimal directions" are identical.
#     # i.e., each index should be matching in terms of sign.
#     diff_y1 = np.round(post_y1[0] - y1,4)
#     diff_y2 = np.round(post_y2[0] - y2,4)
#     same = check_directions(diff_y1, diff_y2)
#     if same==1:
#         pass
#     else:
#         # Try imposing the "other" direction to test equally optimal sols
#         _, pre_y1_2, post_y1_2, obj_y1_2, _, _ = solve_fluid_control(T,N,h,m,K,\
#                                                             np.array([lambdas]*T),\
#                                                             np.array([mus]*T),tau,P,y1,\
#                                                             senders=list(np.where(diff_y2<0)[0]))
#         if np.round(obj_y1,4)==np.round(obj_y1_2,4):
#             # equally optimal solutions
#             diff_y1_2 = np.round(post_y1_2[0] - y1,4)
#             same1 = check_directions(diff_y1_2, diff_y2, silent=1)
#         else:
#             same1 = 0
        
#         _, pre_y2_2, post_y2_2, obj_y2_2, _, _ = solve_fluid_control(T,N,h,m,K,\
#                                                             np.array([lambdas]*T),\
#                                                             np.array([mus]*T),tau,P,y2,\
#                                                             senders=list(np.where(diff_y1<0)[0]))
#         if np.round(obj_y2,4)==np.round(obj_y2_2,4):
#             # equally optimal solutions
#             diff_y2_2 = np.round(post_y2_2[0] - y2,4)
#             same2 = check_directions(diff_y1, diff_y2_2, silent=1)
#         else:
#             same2 = 0
        
#         if same1 or same2:
#             print("From y1 (modified):", diff_y1_2)
#             print("From y2 (modified):", diff_y2_2)
#             pass
#         else:
#             raise Exception("Optimal directions not matching!")
    
#     if (t+1)%10==0:
#         print("Progress:", np.round((t+1)/n_trials*100,1), "%")
#     print('-----------')
    
    
    
    
    
    
    
    
    
    
#%% For delta>0 s.t. |x1-x2|<delta, does there exist eps>0 s.t. |S1-S2|<eps<=delta?

# n = 5 # tot. #customers. Assume fixed.
# N = 3 # number of queues

# # problem parameters
# h = [1,1,1] # holding costs at each queue
# m = np.array([[0,2,2],
#               [2,0,2],
#               [2,2,0]]) # must satisfy triangular inequality

# ### System-wide joint setup cost
# K = 0 # We're interested in the minimum points not accounting for K

# lambdas = [0.9, 0.9, 0.9] # arrival rate to each queue
# mus = [1]*N # service rate at each queue
# tau = 5 # length of time period
# P = 100 # level of discretization
# T = n # proportionally long horizon
# np.random.seed(3)

# n_trials = 10
# i = -1
# while i < n_trials - 1:
#     x = np.random.uniform(size=N) # Randomly sample initial state
#     x1 = x / sum(x) * n # normalize and ensure sum=n
    
#     delta = 1e-2
#     x2 = np.copy(x1)
#     for j in range(N):
#         # random perturbation
#         if np.random.rand() < 0.5:
#             x2[j] = x1[j] + np.random.rand()*(delta/N)
#         else:
#             x2[j] = x1[j] - np.random.rand()*(delta/N)
    
#     deviation = np.round(np.linalg.norm(np.round(x1,4)-np.round(x2,4)),5)
#     if deviation > delta:
#         raise Exception("x1 and x2 deviate more than delta!")
    
#     # Get value functions
#     _, _, post1, obj1, _, _ = solve_fluid_control(T,N,h,m,K,\
#                                                 np.array([lambdas]*T),\
#                                                 np.array([mus]*T),tau,P,x1)
#     _, _, post2, obj2, _, _ = solve_fluid_control(T,N,h,m,K,\
#                                                 np.array([lambdas]*T),\
#                                                 np.array([mus]*T),tau,P,x2)
    
    