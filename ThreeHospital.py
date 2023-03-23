# Jangwon Park, 2022
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
plt.close('all')


#%% Parameters
eps = 1e-4 # Used to judge convergence of value functions

parameters = {
    'beta': 0.05, # Discount rate
    'lambda1': 0.1, # Arrival rate to hospital 1
    'lambda2': 0.1, # Arrival rate to hospital 2
    'lambda3': 0.1, # Arrival rate to hospital 3
    'mu1': 1, # Service rate at hospital 1
    'mu2': 1, # Service rate at hospital 2
    'mu3': 1, # Service rate at hospital 3
    'K1': 0.3, # Fixed cost of transfer (1<->2)
    'K2': 0.3, # Fixed cost of transfer (2<->3)
    'K3': 0.3, # Fixed cost of transfer (3<->1)
    'm': 0.1, # Unit transfer cost per patient
    'h1': 1, # Unit holding cost per patient at hospital 1
    'h2': 1, # Unit holding cost per patient at hospital 2
    'h3': 1, # Unit holding cost per patient at hospital 3
    'M': 10 # Max. cap on system size
    }


#%% Value iteration 

def solve_instance(params,N=3):
    beta = params['beta']
    lambdas = []
    mus = []
    Ks = []
    hs = []
    for j in range(N): # Iterating over the 3 hospitals
        index = str(j+1)
        lambdas.append(params['lambda'+index])
        mus.append(params['mu'+index])
        Ks.append(params['K'+index])
        hs.append(params['h'+index])
    m = params['m']
    M = params['M']
    uniform_rate = sum(lambdas) + sum(mus) + beta # For uniformization
    
    
    def compute_obj(V,I,i,a):
        # Evaluates the objective given a state and action.
        # Arguments:
            # V: Value functions (array)
            # I: Total number of patients in the system (scalar)
            # i: Number of patients in each hospital (vector)
            # a: Number of patients to transfer (vector)
            
        obj = 0
        
        q_lengths = []
        
        # Special case: first hospital
        obj += Ks[0]*(a[0]!=0) # Add fixed cost of transfer
        obj += m*np.abs(a[0]) # Add variable cost of transfer
        q_lengths.append(i[0] + a[-1] - a[0]) 
        obj += 1/uniform_rate * q_lengths[0] * hs[0] # Add congestion cost
        
        # General case: second to last hospitals
        for j in range(1,N):
            obj += Ks[j]*(a[j]!=0) # Add fixed cost of transfer
            obj += m*np.abs(a[j]) # Add variable cost of transfer
            q_lengths.append(i[j] + a[j-1] - a[j])
            obj += 1/uniform_rate * q_lengths[j] * hs[j] # Add congestion cost
                
        # # Used for debugging
        # if sum(np.array(q_lengths)<0) > 0:
        #     print("ERROR!")
        # if sum(q_lengths)!=I:
        #     print("ERROR!")
            
        # Define W(I,i+a)
        arrival_expr = 0
        departure_expr = 0
        for j in range(N-1):
            # Last hospital, by default, is not part of state (hence q_lengths[:-1])
            arrival_indices = [min(I+1,M)] + q_lengths[:-1] # List concatenation
            arrival_indices[j+1] = min(q_lengths[j]+1,M)
            arrival_expr += lambdas[j] * V[tuple(arrival_indices)]
            
            departure_indices = [max(I-1,0)] + q_lengths[:-1]
            if q_lengths[j]==0: # Hospital j is empty
                # Must not allow a departure from an empty queue.
                departure_indices[0] = I
                # departure_indices[j+1] = q_lengths[j]
            else:
                # departure_indices[0] = max(I-1,0)
                departure_indices[j+1] = max(q_lengths[j]-1,0)
            departure_expr += mus[j] * V[tuple(departure_indices)]
            
        # Special case: Last hospital
        arrival_expr += lambdas[-1] * V[tuple([min(I+1,M)] + q_lengths[:-1])]
        if q_lengths[-1]==0:
            departure_expr += mus[-1] * V[tuple([I] + q_lengths[:-1])]
        else:
            departure_expr += mus[-1] * V[tuple([max(I-1,0)] + q_lengths[:-1])]
        
        W = 1/uniform_rate * (arrival_expr + departure_expr)
        obj += W
        
        return obj
    
    
    def check_infeasible_action(a,i):
        # Check whether given actions "a" are infeasible.
        # Construct "action matrix" A (hard-coded for N=3)
        A = np.zeros((N,N))
        A[0,1] = a[0]
        A[1,0] = -a[0]
        A[1,2] = a[1]
        A[2,1] = -a[1]
        A[2,0] = a[2]
        A[0,2] = -a[2]
        A[0,0] = i[0] - A[0,A[0,:]>0].sum()
        A[1,1] = i[1] - A[1,A[1,:]>0].sum()
        A[2,2] = i[2] - A[2,A[2,:]>0].sum()
        
        if (A[0,0]<0) or (A[1,1]<0) or (A[2,2]<0):
            # Diagonal elements cannot be negative by definition.
            return True

        return False
        
    
    def iterate_general(V):
        # Enumerates all states and updates value functions.
        num_pairs = int(N*(N-1)/2) # N choose 2
        
        new_V = np.zeros([M+1]*N) # (M+1)^N dimensional array
        pol = np.zeros([M+1]*N + [num_pairs]) # (M+1)^N x num_pairs dimensional array
        for I in range(M+1): # Total number of patients in the system
            state_space = itertools.product(*[range(I+1)]*(N-1))
            for i in state_space:
                if sum(i) > I: continue # Infeasible state
                
                # tuple concatenations
                state = (I,) + i
                i = i + (I - sum(i),) # Include no. of patients in last hospital
                
                # Generate actions (hard-coded for N=3)
                actions = []          
                for j in range(num_pairs-1):
                    # Actions: No. of patients to transfer from hospital j to j+1
                    actions.append(range(-i[j+1],i[j]+1,1))
                
                # Special case: Last hospital
                actions.append(range(-i[0],i[-1]+1,1))
                
                # Find optimal action
                action_space = itertools.product(*actions)
                best_obj = float('inf')
                best_action = 0
                for a in action_space:        
                    if check_infeasible_action(a,i): continue # infeasible action
                    
                    obj = compute_obj(V,I,i,a)
                    
                    if obj <= best_obj: # Find minimum objective
                        best_obj = obj
                        best_action = a
                # print(I,i,best_action) # For debugging
                new_V[state] = best_obj
                pol[state][:] = best_action
                
        return new_V, pol
    
    
    def iterate_special(V):
        # Requires EQUAL holding costs to make sense!
        
        # SPECIAL STRUCTURE OF OPTIMAL POLICY:
        # If all hospitals are non-idling, action a=0 is always optimal.
        # Otherwise, we only need to optimize decisions of non-idle hospitals.
        num_pairs = int(N*(N-1)/2) # N choose 2
        
        new_V = np.zeros([M+1]*N) # (M+1)^N dimensional array
        pol = np.zeros([M+1]*N + [num_pairs]) # (M+1)^N x num_pairs dimensional array
        for I in range(M+1):
            state_space = itertools.product(range(I+1),repeat=N-1)
            for i in state_space:
                if sum(i) > I: continue # Infeasible state
                
                # tuple concatenations
                state = (I,) + i
                i = list(i) + [I - sum(i)] # Include no. of patients in last hospital
                
                ### ============== Exploit special structure ============== ###
                if sum(np.array(i)>0) == N: # No idle hospitals
                    best_action = tuple([0]*N) # "Doing nothing" is optimal
                    best_obj = 0
                    for j in range(N):
                        best_obj += 1/uniform_rate * i[j] * hs[j]
                        
                    arrival_expr = 0
                    departure_expr = 0
                    for j in range(N-1):
                        # Last hospital, by default, is not part of state (hence q_lengths[:-1])
                        arrival_indices = [min(I+1,M)] + i[:-1] # List concatenation
                        arrival_indices[j+1] = min(i[j]+1,M)
                        arrival_expr += lambdas[j] * V[tuple(arrival_indices)]
                        
                        departure_indices = [max(I-1,0)] + i[:-1]
                        departure_indices[j+1] = max(i[j]-1,0)
                        departure_expr += mus[j] * V[tuple(departure_indices)]
                        
                    # Special case: Last hospital
                    arrival_expr += lambdas[-1] * V[tuple([min(I+1,M)] + i[:-1])]
                    departure_expr += mus[-1] * V[tuple([max(I-1,0)] + i[:-1])]
                    
                    W = 1/uniform_rate * (arrival_expr + departure_expr)
                    best_obj += W
                    
                    new_V[state] = best_obj
                    pol[state][:] = best_action
                    continue
                ### ======================================================= ###
                
                # Generate actions (hard-coded for N=3)
                actions = []          
                for j in range(num_pairs-1):
                    # Actions: No. of patients to transfer from hospital j to j+1
                    actions.append(range(-i[j+1],i[j]+1,1))
                
                # Special case: Last hospital
                actions.append(range(-i[0],i[-1]+1,1))
                
                # Find optimal action
                action_space = itertools.product(*actions)
                best_obj = float('inf')
                best_action = 0
                for a in action_space:        
                    if check_infeasible_action(a,i): continue # infeasible action
                    
                    obj = compute_obj(V,I,i,a)
                    
                    if obj <= best_obj: # Find minimum objective
                        best_obj = obj
                        best_action = a
                # print(I,i,best_action) # For debugging
                new_V[state] = best_obj
                pol[state][:] = best_action
                
        return new_V, pol
       
    
    def converge_value(eps=1e-4):
        # Value iteration algorithm
        
        V = np.zeros([M+1]*N) # Initialize value functions at zeros
        num_iter = 1
    
        new_V, pol = iterate_general(V)
        # new_V, pol = iterate_special(V)
        
        # while (Mn-mn) > eps*mn: # I don't use this criterion because mn (defined as (new_V - V).min()) is always 0.
        while np.abs(V - new_V).max() > eps: # Must be <= for convergence
            V = new_V
            new_V, pol = iterate_general(V)
            # new_V, pol = iterate_special(V)
                
            num_iter += 1
            if num_iter%100==0:
                print("Iteration no.:",num_iter)
        
        return new_V, pol, num_iter
    
    V, pol, num_iter = converge_value(eps)
    
    return V, pol, num_iter


#%% Run algorithm
 
start = time.time()
V, pol, num_iter = solve_instance(parameters)
end = time.time()

print(V)
print(pol)
print("Solution time (s): {}, No. iterations: {}".format(end-start,num_iter))

save_heatmap_2D(V[:,:,5],pol[:,:,5,0],parameters)
save_heatmap_2D(V[:,:,5],pol[:,:,5,1],parameters)
save_heatmap_2D(V[:,:,5],pol[:,:,5,2],parameters)