# Jangwon Park, 2022

import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from gurobipy import *

import SimFunctions
import SimClasses
import SimRNG
from output_processing import draw_sample_path


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
            post_states[k,i] = y[k,i,0].x
            
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
    plt.ylabel("$x^k$",fontsize=22)
    plt.title(title,fontsize=24)
    plt.grid(alpha=0.33)
    return
        
        
def solve_fluid_control(T,N,h,m,K,lambdas,mus,tau,P,x0,\
                        freq=1,E=None,UB=GRB.INFINITY,force_zero=False,\
                        force_pos=False,force_bound=False,senders=[]):
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
        boundaries = np.round((np.array(mus[0,:]) - np.array(lambdas[0,:]))*tau,2) # must be known a priori
        Ind2 = model.addVars(N,vtype=GRB.BINARY)  # interior indicator
        # Recall: y[0,i,0] is the post-transfer state at queue i at the start of period 0
        for i in range(N):
            model.addGenConstrIndicator(Ind2[i],0,y[0,i,0]<=boundaries[i])
        model.addConstr(quicksum(Ind2[i] for i in range(N)) <= N-1) # If ==N, it means interior
    if senders: # Find optimal solution within a particular "direction" of transfer (just in period 0)
        model.addConstrs(I[0,j,s]==0 for j in range(N) for s in senders) # can't receive any
        receivers = list(set(range(3)) - set(senders))
        model.addConstrs(I[0,r,j]==0 for j in range(N) for r in receivers) # can't send any
        
    # # temporary: to be deleted
    # model.addConstr(I[0,2,0] == 0)
    # model.addConstr(I[0,1,0] == 0)
    # model.addConstr(y[0,0,0] == 7) # post-transfer state of queue 0 at the very start of period 0
    # model.addConstr(y[0,2,0] == 1) # post-transfer state of queue 0 at the very start of period 0
        
    model.params.logtoconsole = 0
    model.optimize()
    
    # Extract output
    Transfers, pre_states, post_states = extract_output(T,N,x0,P,I,y,z)
    
    # Extract trajectories
    trajectories = extract_trajectory(z, T, N, P)
    
    return Transfers, pre_states, post_states, model.objVal, trajectories, model


#%% convergence experiments (N=2)

# N = 2 # number of queues

# # problem parameters
# h = [1,1.5] # holding costs at each queue
# m = np.array([[0,1],
#               [1,0]])

# # queue pair-specific setup costs
# K = np.array([[0,4],
#               [4,0]])

# lambdas = [0.9,0.9] # arrival rate to each queue
# mus = [1,1] # service rate at each queue
# tau = 10 # length of time period
# P = 100 # level of discretization


# scales = [1,5,10,20,30]
# transfers = []
# for scale in scales:
#     x0 = scale * np.array([5,0]) # initial condition
#     T = int(sum(x0)) # proportionally long horizon
    
#     # Solve fluid control problem
#     sol, pre, post, obj, trajectories, model = solve_fluid_control(T,N,h,m,K,\
#                                                         np.array([lambdas]*T),\
#                                                         np.array([mus]*T),tau,P,x0)
    
#     print('Scale: {}, Transferred: {}'.format(scale, np.round(sol[0].sum(),0)))
#     transfers.append(sol[0].sum())
    
# # Plot transfers vs. scale
# plt.close("all")
# plt.figure()
# plt.plot(scales,transfers)
# plt.grid(alpha=0.33)
# plt.xlabel("Scale")
# plt.ylabel("Number transferred")
# plt.show()

#%% Convergence of (scaled) sample path under optimal fluid policy

n = 10 # scaling factor
mus = np.array([1, 1])*n # service rates (scaled)
lambdas = np.array([0.55, 0.95])*n # arrival rates (scaled)
h = [1,1] # holding costs at each queue
m = np.array([[0,1],
              [1,0]])*1
K = 1 # System-wide joint setup cost
init_x = np.array([2, 5]) * n # initial state (scaled)
# init_x = np.array([2, 10]) # initial state (non-scaled)
tau = 10 # length of single period
P = 100 # level of discretization (higher is better to have dotted lines line up properly)
N = len(h) # number of queues
T = int(np.ceil(max(init_x / (mus-lambdas)) / tau)) # number of periods (epochs)
# T = int(np.ceil(max(init_x / (mus/n-lambdas/n)) / tau)) # number of periods (epochs)
lambdas = np.tile(lambdas,T).reshape(T,N)
mus = np.tile(mus,T).reshape(T,N)

ZRNG = SimRNG.InitializeRNSeed()  # seed for simulation
ns = 3                            # seed for numpy
np.random.seed(ns)

# Streams: Each queue has 2 unique streams (one for arrival process and one for service process)
reps = 0
stream = 1
# while reps < 20:
ZRNG.append(int(np.mean(np.random.choice(ZRNG, size=int(np.sqrt(len(ZRNG))), replace=False))))
ZRNG.append(int(np.mean(np.random.choice(ZRNG, size=int(np.sqrt(len(ZRNG))), replace=False))))
Streams = []
for i in range(N):
    Streams.append(stream)
    stream += 1

# Create queue and server (resource) objects
Queues = []
Resources = []
for i in range(N):
    # Create and store FIFO queue object
    Queues.append(SimClasses.FIFOQueue())
    
    # Create and store resource object
    Hospital = SimClasses.Resource()
    Hospital.SetUnits(1) # single server
    Resources.append(Hospital)

# Define performance measures
DTStats_w = [] # for wait time statistics
DTStats_t = [] # for transfer cost statistics
CTStats = []
for i in range(N):
    wait_time = SimClasses.DTStat() # expected wait time / patient
    transfer_cost = SimClasses.DTStat() # expected transfer cost
    DTStats_w.append(wait_time)
    DTStats_t.append(transfer_cost)
    CTStats.append(Queues[i].WIP)

Calendar = SimClasses.EventCalendar() # event calendar (sorts order of events)
SD = SimRNG.Expon # exponentially dist'd service times
IAD = SimRNG.Expon # exponentially dist'd inter-arrival times

def Arrival(k):
    """
    Arrival logic.
    k: current stage in simulation
    """
    
    new_patient = SimClasses.Entity()
    
    # Probabilistically assign patient to a queue
    p = lambdas[k,:] / sum(lambdas[k,:]) 
    queues = list(range(N))
    queue = np.random.choice(queues, size=1, p=p)[0] # Select queue
    new_patient.Classtype = queue # new association
    
    Queue = Queues[queue]
    Hospital = Resources[queue]
    if Hospital.Busy==0:
        Hospital.Seize(1) # Serve new customer immediately
        
        # Record patient's wait time (which is zero)
        wait_time = DTStats_w[queue] # Get the right DTStat
        wait_time.Record(SimClasses.Clock - new_patient.CreateTime)
        
        # Schedule departure
        meanST = 1/mus[k,queue] # mean service time
        stream = Streams[queue]
        SimFunctions.SchedulePlus(Calendar,"EndOfService",
                                  SD(ZRNG,meanST,stream),new_patient)
    else:
        Queue.Add(new_patient) # holding cost recorded automatically
    
    # Schedule next arrival 
    meanTBA = 1/sum(lambdas[k,:]) # mean time between arrivals
    SimFunctions.Schedule(Calendar,"Arrival",IAD(ZRNG,meanTBA,1)) 
    
def EndOfService(k,departing_patient):
    """
    Service logic.
    k: current stage in simulation
    departing_patient: entity object of patient whose service just finished
    """
    # Get the associated queue/hospital index
    queue = departing_patient.Classtype 
    Queue = Queues[queue]
    Hospital = Resources[queue]
    Hospital.Free(1) 
    
    if Queue.NumQueue() > 0:
        # expected holding cost updated automatically
        new_patient = Queue.Remove() 
        Hospital.Seize(1) # Begin serving immediately
        
        # Record patient's wait time in advance
        wait_time = DTStats_w[queue] # Get the right DTStat
        wait_time.Record(SimClasses.Clock - new_patient.CreateTime)
        
        # Schedule departure
        meanST = 1/mus[k,queue] # mean service time
        stream = Streams[queue]
        SimFunctions.SchedulePlus(Calendar,"EndOfService",
                                  SD(ZRNG,meanST,stream),new_patient)

def ImplementPolicy(pol):
    """
    Implement optimal fluid policy.
    Arguments:
        pol: optimal fluid policy (NxN matrix)
    """
    for i in range(N):
        Queue = Queues[i]
        for j in range(i+1,N):
            Queue2 = Queues[j]
            # number of transfers TO Queue2
            to_queue2 = int(pol[i,j]) # round down to ensure feasibility
            # number of transfers FROM Queue2
            from_queue2 = int(pol[j,i]) # round down to ensure feasibility
            
            # Tiny numerical errors can lead to negative numbers
            if to_queue2 < 0:
                to_queue2 = 0
            if from_queue2 < 0:
                from_queue2 = 0
            
            if (to_queue2==0) and (from_queue2==0):
                continue # Skip if no transfers take place at all
        
            # patients in queue 1
            patients = list(range(Queue.NumQueue())) 
            # patients in queue 2
            patients2 = list(range(Queue2.NumQueue())) 
            if (len(patients)==0) and (len(patients2)==0):
                continue
            
            if to_queue2 > 0:
                outgoing = patients[-to_queue2:] # Select most recent patients
            else:
                outgoing = []
            if from_queue2 > 0:
                incoming = patients2[-from_queue2:] # Select most recent patients
            else:
                incoming = []
            # outgoing = np.random.choice(patients,size=to_queue2,replace=False) # random selection
            # incoming = np.random.choice(patients2,size=from_queue2,replace=False) # random selection
            
            # Remove patients
            outgoing_patients = []
            # Sort in decreasing order of patient index;
            # crucial for avoiding "index out of range" error      
            outgoing = sorted(outgoing,reverse=True) 
            for out in outgoing:
                patient = Queue.Remove(index=out)
                patient.Classtype = j # Change the associated queue
                outgoing_patients.append(patient)
                
            incoming_patients = []
            incoming = sorted(incoming,reverse=True) 
            for inc in incoming:
                patient = Queue2.Remove(index=inc)
                patient.Classtype = i # Change associated queue
                incoming_patients.append(patient)
            
            # Record transfer cost at each queue
            queue1_transfer_cost = DTStats_t[i]
            queue2_transfer_cost = DTStats_t[j]
            # Incur cost by num. of patients moved
            queue1_transfer_cost.Record(m[i,j] * to_queue2 + K*(to_queue2>0)) 
            # Incur cost by num. of patients moved
            queue2_transfer_cost.Record(m[j,i] * from_queue2 + K*(from_queue2>0)) 
            
            # Add patients
            for new_patient in incoming_patients:
                Queue.Add(new_patient)
                
            for new_patient in outgoing_patients:
                Queue2.Add(new_patient)


# SIMULATE ONE REPLICATION (sample path)
warmup = 0 
SimFunctions.SimFunctionsInit(Calendar, Queues, CTStats, DTStats_w + DTStats_t, Resources)
SimFunctions.Schedule(Calendar,"EndSimulation",T*tau)

times = [0] # Store sample event times
# Store sample queue lengths
all_queue_lengths = np.zeros((N,1)) 
for i in range(N):
    all_queue_lengths[i,0] = init_x[i]

# Start system at a given initial state (if non-zero)
if sum(init_x) > 0:
    for i in range(N):
        Queue = Queues[i]
        for j in range(int(init_x[i])):
            new_patient = SimClasses.Entity()
            new_patient.Classtype = i
            Queue.Add(new_patient)

# current stage (updated +=1 after solving fluid control)
k = 0

# Schedule all fluid control now (beginning of each day)
for t in range(0,T):
    SimFunctions.Schedule(Calendar,"FluidControl",t*tau)

# Schedule the first arrival
# mean time between arrivals in first stage
meanTBA = 1/sum(lambdas[k,:]) 
SimFunctions.Schedule(Calendar,"Arrival",IAD(ZRNG,meanTBA,1))

NextEvent = Calendar.Remove()
SimClasses.Clock = NextEvent.EventTime

if NextEvent.EventType == "Arrival":
    Arrival(k)
elif NextEvent.EventType == "FluidControl":
    x0 = np.array([Queues[i].NumQueue() for i in range(N)]) # "initial" state in fluid control
    pol, _, _, obj, trajectories, _ = solve_fluid_control(T,N,h,m,K,lambdas/n,mus/n,tau,P,x0/n)
    # pol, _, _, obj, trajectories, _ = solve_fluid_control(T,N,h,m,K,lambdas/n,mus/n,tau,P,x0)
    ImplementPolicy(pol[0]*n) # Implement the optimal policy    
    # ImplementPolicy(pol[0]) # Implement the optimal policy    
    # k += 1 # Advance stage

while NextEvent.EventType != "EndSimulation":
# while True:
    NextEvent = Calendar.Remove()
    SimClasses.Clock = NextEvent.EventTime
    
    if NextEvent.EventType == "Arrival":
        Arrival(k)
    elif NextEvent.EventType == "EndOfService":
        EndOfService(k,NextEvent.WhichObject)
    elif NextEvent.EventType == "FluidControl":
        x0 = np.array([Queues[i].NumQueue() for i in range(N)]) # "initial" state in fluid control
        
        pol, _, _, _, _, _ = solve_fluid_control(T,N,h,m,K,lambdas/n,mus/n,tau,P,x0/n)
        # pol, _, _, _, _, _ = solve_fluid_control(T,N,h,m,K,lambdas/n,mus/n,tau,P,x0)
        ImplementPolicy(pol[0]*n) # Implement the optimal policy   
                                  # Multiply by n because sample paths will be
                                  # scaled down by n eventually.
        # ImplementPolicy(pol[0])
        # k += 1 # Advance stage
    
    # Gather queue lengths and event times
    col_to_append = np.zeros((N,1))
    for y in range(N):
        col_to_append[y] = Queues[y].NumQueue()
    
    # horizontal concatenation
    all_queue_lengths = np.append(all_queue_lengths, col_to_append, 1) 
    times.append(SimClasses.Clock)
        
    # if sum(col_to_append)==0:
    #     break
    
times = np.array(times)

# Compute total holding cost, and transfer cost
tot_holding_cost = 0
tot_transf_cost = 0
for i in range(N):
    tot_area_under_curve = CTStats[i].Area + CTStats[i].Xlast * (SimClasses.Clock - CTStats[i].Tlast)
    tot_holding_cost += tot_area_under_curve * h[i]
    tot_transf_cost += DTStats_t[i].Sum
tot_cost = (tot_holding_cost + tot_transf_cost) / n # need to scale down by n
# tot_cost = (tot_holding_cost + tot_transf_cost)
print("Sample path cost:", tot_cost)
print("Fluid cost:", obj)

reps += 1

scaled_paths = all_queue_lengths / n
# scaled_paths = all_queue_lengths
periods_to_plot = 4
draw_sample_path(times,scaled_paths,periods_to_plot,N,h,m,K,lambdas,mus,tau,P,trajectories)



