# Jangwon Park, 2022
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
import time
# plt.close('all')


#%% Parameters
eps = 1e-3 # Used to judge convergence of value functions

parameters = {
    'beta': 0.0, # Discount rate
    'lambda1': 0.7, # Arrival rate to hospital 1
    'lambda2': 0.9, # Arrival rate to hospital 2
    'mu1': 1, # Service rate at hospital 1
    'mu2': 1, # Service rate at hospital 2
    'K': 5, # Fixed cost of transfer
    'm': 2, # Unit transfer cost per patient; this is "r" in the paper
    'h1': 1, # Unit holding cost per patient at hospital 1
    'h2': 1, # Unit holding cost per patient at hospital 2
    'M': 150 # Max. cap on system size
    }


#%% Value iteration 

def solve_instance(params,eta=1):
    beta = params['beta']
    lambda1 = params['lambda1'] * eta 
    lambda2 = params['lambda2'] * eta
    mu1 = params['mu1'] * eta
    mu2 = params['mu2'] * eta
    K = params['K']
    m = params['m']
    h1 = params['h1']
    h2 = params['h2']
    M = params['M']
    uniform_rate = lambda1 + lambda2 + mu1 + mu2 + beta # For uniformization
    
    
    def compute_obj(V,I,i,a):
        # This function evaluates the objective given a state and action.
        obj = 0
        obj += K*(a!=0) # Add fixed cost of transfer
        obj += m*np.abs(a) # Add variable cost of transfer
        
        q_length1 = I-i-a # Will never be < 0 or > M
        q_length2 = i+a # Will never be < 0 or > M
        obj += 1/uniform_rate * (
            q_length1*h1 + q_length2*h2 
            ) # Add congestion cost
        
        # empty system: Make (0,0) an absorbing state
        if I==0: 
            W = 0 # 0 b/c the transition probabilities out of (0,0) are 0.
            obj += W # should be 0
            return obj
        
        # Define W(I,i+a): Must not allow a departure from an empty queue.
        if I-i-a==0: # Hospital 1 is empty after taking action
            expr1 = V[I,i+a]
            expr2 = V[max(I-1,0),max(i+a-1,0)]
        elif i+a==0: # Hospital 2 is empty after taking action
            expr1 = V[max(I-1,0),i+a]
            expr2 = V[I,i+a]
        else: # General case
            expr1 = V[max(I-1,0),i+a]
            expr2 = V[max(I-1,0),max(i+a-1,0)]
        
        W = 1/uniform_rate * (
            lambda1 * V[min(I+1,M),i+a] +
            lambda2 * V[min(I+1,M),min(i+a+1,M)] +
            mu1 * expr1 + 
            mu2 * expr2
            )       

        obj += W
        return obj
    
    
    def iterate_general(V):
        # This function enumerates all states and updates value functions
        # This function uses the general action space
        new_V = np.zeros((M+1,M+1))
        pol = np.zeros((M+1,M+1))
        for I in range(M+1): # Total number of patients in the system
        # for I in np.linspace(0,M,21):
            for i in range(I+1): # Number of patients in hospital 2, i<=I
                A = range(-i,I-i+1,1) # Feasible actions
                best_obj = float('inf')
                best_action = 0
                for a in A:                    
                    obj = compute_obj(V,I,i,a)
                    
                    if obj <= best_obj: # Find minimum objective
                        best_obj = obj
                        best_action = a
                        
                new_V[I,i] = best_obj
                pol[I,i] = best_action
                
        return new_V, pol
    
    
    def iterate_special(V):
        # Must have h1 >= h2 to use this!
        
        # SPECIAL STRUCTURE OF OPTIMAL POLICY:
        # For each I, there is a level S(I) such that either a=0 OR a=S(I)-i 
        # is optimal, if i < S(I). If i >= S(I), a=0 is optimal.
        
        new_V = np.zeros((M+1,M+1)) # Save updated value functions
        pol = np.zeros((M+1,M+1)) # Save best action at each state
        order_levels = [0]*(M+1) # Save optimal order-up-to levels S(I)
        
        # Special case: I==0. Only i=0, a=0, S(I)=0 are possible.
        new_V[0,0] = compute_obj(V, 0, 0, 0)
        
        for I in range(1,M+1):
            # Find optimal S(I) at i==0
            value_a_0 = compute_obj(V, I, 0, 0) # Value at (I,0) under a=0
            best_obj = value_a_0
            best_action = 0
            for S in range(order_levels[I-1],I+1): # Monotonicity: S(I-1) <= S(I) <= I
                # Either a=0 or a=S is optimal
                objs = [value_a_0, compute_obj(V, I, 0, S)]
                min_idx = np.argmin(objs)
                obj = objs[min_idx]
                action = [0, S][min_idx]
                if obj <= (best_obj + eps): # Find minimizing S(I)
                    best_obj = obj
                    best_action = action
                    best_S = S
            new_V[I,0] = best_obj
            pol[I,0] = best_action
            
            # Apply the same S(I) for 1 <= i <= I-1
            for i in range(1,I):
                if i >= best_S:
                    # a=0 is optimal
                    best_obj = compute_obj(V, I, i, 0)
                    best_action = 0
                else: # i < best_S
                    if pol[I,i-1]==0:
                        # a=0 must be optimal (not proven in Down and Lewis, just conjectured)
                        best_obj = compute_obj(V, I, i, 0)
                        best_action = 0
                    else:
                        # Either a=0 or a=S-i is optimal
                        objs = [compute_obj(V,I,i,0), compute_obj(V,I,i,best_S-i)]
                        min_idx = np.argmin(objs)
                        best_obj = objs[min_idx]
                        best_action = [0, best_S-i][min_idx]
                new_V[I,i] = best_obj
                pol[I,i] = best_action
            
            # Special case: I=i (Hospital 1 is "idle")
            best_obj = float('inf')
            best_action = 0
            for a in range(-I,1): # Only non-positive actions are feasible
                obj = compute_obj(V,I,I,a)
                if obj <= best_obj: # Find minimum objective
                    best_obj = obj
                    best_action = a
            new_V[I,I] = best_obj
            pol[I,I] = best_action
                    
            order_levels[I] = best_S
            
        return new_V, pol, order_levels
    
    
    def evaluate_policy(eps,pol):
        # This function is an iterative policy evaluation algorithm
        V = np.zeros((M+1,M+1))
        num_iter = 0
        while True:
            num_iter += 1
            new_V = np.zeros((M+1,M+1))
            for I in range(M+1):
                for i in range(I+1):
                    new_V[I,i] = compute_obj(V,I,i,int(pol[I,i]))
            
            if np.abs(V - new_V).max() <= eps:
                break
            else:
                V = new_V
                continue
        return new_V, num_iter
    
    
    def converge_value(eps=1e-4,evaluate=False,pol=False,order_levels=False):
        # This function is essentially the value iteration algorithm
        
        if evaluate: # Policy evaluation
            new_V, num_iter = evaluate_policy(eps,pol)
        else: # Value iteration
            V = np.zeros((M+1,M+1)) # Initialize value functions at zeros
            num_iter = 1
        
            new_V, pol = iterate_general(V)
            # new_V, pol, order_levels = iterate_special(V)
            
            # while (Mn-mn) > eps*mn: # I don't use this criterion because mn (defined as (new_V - V).min()) is always 0.
            while np.abs(V - new_V).max() > eps: # Must be <= for convergence
                V = new_V
                new_V, pol = iterate_general(V)
                # new_V, pol, order_levels = iterate_special(V)
                    
                num_iter += 1
                if num_iter%100==0:
                    print("Iteration no.:",num_iter)
        
        return new_V, pol, num_iter, order_levels
    
    # pol = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #  [ 1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #  [ 2.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #  [ 3.,  2.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
    #  [ 4.,  3.,  2.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
    #  [ 5.,  4.,  3.,  2.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
    #  [ 6.,  5.,  4.,  3.,  2.,  0.,  0., -1.,  0.,  0.,  0.],
    #  [ 7.,  6.,  5.,  4.,  3.,  2.,  0.,  0., -1.,  0.,  0.],
    #  [ 8.,  7.,  6.,  5.,  4.,  3.,  2.,  0.,  0., -1.,  0.],
    #  [ 8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.,  0.,  0., -1.]])
    
    V, pol, num_iter, order_levels = converge_value(eps)
    
    return V, pol, num_iter, order_levels

#%% Run algorithm

def main():   
    # opt_action = []
    # lambda2 = np.linspace(0.1,2,20)
    # for lam in lambda2:
    #     parameters.update({'lambda2': lam})
    start = time.time()
    eta = 5 # scaling the system 
    V, pol, num_iter, order_levels = solve_instance(parameters, eta)
    end = time.time()
    #     opt_action.append(pol[-1,0]) # (I=M,i=0)
    # generate_step(parameters,lambda2,opt_action,"lambda_2","Optimal action","Optimal action at (I=M,i=0)")

    print("Solution time (s): {}, No. iterations: {}".format(end-start,num_iter))
    # print(V)
    # print(pol)
    # print(order_levels)
    # save_heatmap_2D({}, pol, parameters)
    return V, pol, order_levels, eta 

if __name__ == "__main__":
    V, pol, order_levels, eta = main()
    np.save("saved_policies/MDP_policy_identical_heavy_far_large_scale" + str(eta) + ".npy",pol)
    
#%% Observe optimal policy under "fluid scaling"

# M = parameters["M"]
# vals = [pol[i*5,i*0] for i in range(11)]

# plt.close("all")
# plt.figure()
# plt.plot(np.abs(vals))
# plt.grid(alpha=0.33)
# plt.title("Total number transferred at $n[5,0]$")
# plt.xlabel("$n$")
# plt.ylabel("Number transferred")
# plt.tight_layout()
# plt.show()