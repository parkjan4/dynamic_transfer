# Jangwon Park, 2022

import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

#%%
parameters = {
    'lambda1': 0.5, # Arrival rate to queue 1
    'lambda2': 0.5, # Arrival rate to queue 2
    'lambda3': 0.5, # Arrival rate to queue 3
    'mu1': 1, # Service rate at queue 1
    'mu2': 1, # Service rate at queue 2
    'mu3': 1, # Service rate at queue 3
    'm1': 1, # Unit transfer cost per customer from queue 1
    'm2': 1, # Unit transfer cost per customer from queue 2
    'm3': 1, # Unit transfer cost per customer from queue 3
    'h1': 1, # Unit holding cost per customer at queue 1
    'h2': 1, # Unit holding cost per customer at queue 2
    'h3': 1, # Unit holding cost per customer at queue 3
    'tau': 10 # Length of time period
    }

#%%

def compute_cost(N,x,I,params):
    # Compute one period cost.
    # Inputs:
        # x = vector of three queue lengths
        # I = action matrix (NxN)
    
    tau = params['tau']
    K = params['K']
    g = 0
    for i in range(N):
        # Transfers in and out are based on adding up positive numbers only.
        transfer_in = I[I[:,i]>0,i].sum() 
        transfer_out = I[i,I[i,:]>0].sum()
        
        index = str(int(i+1))
        lam = params['lambda'+index]
        mu = params['mu'+index]
        h = params['h'+index]
        m = params['m'+index]
        
        # Check condition on emptying time
        tmp = x[i] + transfer_in - transfer_out
        if tmp >= ((mu - lam)*tau):
            g += h * (tmp*tau + 0.5*(lam-mu)*tau**2)
        else:
            g += h/(2*(mu-lam)) * (tmp**2)
        
        # Compute fixed transfer cost
        fixed_cost = 0
        for j in range(N):
            fixed_cost += K[i,j] * (I[i,j] > 0)
            
        g += m*transfer_out + fixed_cost # Add all transfer costs
    
    return g

def check_infeasible_action(N,a,x):
    # Check whether given actions in "a" are infeasible.
    # Construct "action matrix" A (hard-coded for N=3)
    eps = 1e-6 # numerical error tolerance
    
    A = np.zeros((N,N))
    A[0,1] = a[0]
    A[1,0] = -a[0]
    A[0,2] = a[1]
    A[2,0] = -a[1]
    A[1,2] = a[2]
    A[2,1] = -a[2]
    A[0,0] = x[0] - A[0,A[0,:]>0].sum()
    A[1,1] = x[1] - A[1,A[1,:]>0].sum()
    A[2,2] = x[2] - A[2,A[2,:]>0].sum()
    
    if (A[0,0]<-eps) or (A[1,1]<-eps) or (A[2,2]<-eps):
        # Diagonal elements cannot be negative by definition.
        return True, np.nan

    # By design, diagonal elements of A are zero.
    A[0,0] = 0
    A[1,1] = 0
    A[2,2] = 0
    return False, A


# Find optimal action via enumeration
def solve_instance(N,M):
    # Find optimal actions under all possible initial states given N and M.
    # N: Number of queues
    # M: Total customers in the system (assumed fixed)
    
    K = np.ones((N,N)) * 3 
    for i in range(N):
        K[i,i] = 0
    parameters['K'] = K # Fixed cost of transfer
    
    opt_dec = np.empty((M+1,M+1,M+1,N)) # used to store optimal actions
    opt_dec[:] = np.nan # Initialize
        
    state_space = itertools.product(*[range(M+1)]*(N-1)) # All initial states
    for x in state_space:
        if sum(x) > M: 
            # Infeasible state
            opt_dec[x][:] = np.nan
            continue 
        
        x = x + (M - sum(x),) # Augment state to include queue 3
        
        # Generate actions
        actions = []
        actions.append(range(-x[1],x[0]+1,1)) # transfers from queue 1 to 2
        actions.append(range(-x[2],x[0]+1,1)) # transfers from queue 1 to 3
        actions.append(range(-x[2],x[1]+1,1)) # transfers from queue 2 to 3
        # actions.append(np.linspace(-x[1],x[0],50)) # transfers from queue 1 to 2
        # actions.append(np.linspace(-x[2],x[0],50)) # transfers from queue 1 to 3
        # actions.append(np.linspace(-x[2],x[1],50)) # transfers from queue 2 to 3
        
        # Find optimal action
        action_space = itertools.product(*actions)
        best_obj = float('inf')
        best_action = np.array([0]*N)
        for a in action_space: 
            is_infeasible, I = check_infeasible_action(N,a,x) # Check feasibility
            if is_infeasible: continue
            
            obj = compute_cost(N,x,I,parameters) 
            if obj <= best_obj:
                best_obj = obj
                best_action = a # 3-d vector
        
        opt_dec[x] = best_action
        
    return opt_dec


def get_costs(x,n,N=3):
    # Obtains a matrix of costs g
    # Arguments:
        # x: initial state as a 3-D tuple
        # n: number of actions to consider between two queues
    
    # K = np.ones((N,N)) * 3 
    # for i in range(N):
    #     K[i,i] = 0
    K = np.zeros((N,N))
    parameters['K'] = K # Fixed cost of transfer
    
    # Find optimal action
    first_actions = np.linspace(-x[1],x[0],n) # 1->2
    # first_actions = np.linspace(-x[2],x[1],n) # 2->3
    # X = np.zeros((n,n))
    # Y = np.zeros((n,n))
    # Z = np.zeros((n,n))
    X = {}
    Y = {}
    Z = {}
    for i1,a1 in enumerate(first_actions): 
        second_actions = np.linspace(-x[2],x[0]-a1*(a1>0),n) # 1->3
        # second_actions = np.linspace(-x[2],x[1]+a1*(a1<0),n) # 2->3
        # second_actions = np.linspace(-(x[2]+a1*(a1<0)),x[0],n) # 1->3
        x_ = np.zeros((n,n))
        y_ = np.zeros((n,n))
        z_ = np.zeros((n,n))
        for i2,a2 in enumerate(second_actions):
            best_obj = float('inf')
            third_actions = np.linspace(-(x[2]+a2*(a2<0)),x[1]+a1*(a1<0),n) # 2->3
            # third_actions = np.linspace(-(x[2]+a2*(a2<0)),x[0]-a1*(a1>0),n) # 1->3
            # third_actions = np.linspace(-(x[1]-a1*(a1>0)),x[0]-a2*(a2>0),n) # 1->2
            for i3,a3 in enumerate(third_actions):
                a = (a1,a2,a3)
                is_infeasible, I = check_infeasible_action(N,a,x) # Check feasibility
                if is_infeasible: continue
            
                obj = compute_cost(N,x,I,parameters) 
                if obj < best_obj:
                    best_obj = obj
                x_[i2,i3] = a2
                y_[i2,i3] = a3
                z_[i2,i3] = obj
            X[a1] = x_
            Y[a1] = y_
            Z[a1] = z_
            # X[i1,i2] = a1
            # Y[i1,i2] = a2
            # Z[i1,i2] = best_obj
    return X,Y,Z
    

#%% Solve an instance

# N = 3
# M = 12
# start = time.time()
# opt_dec_sym = solve_instance(N,M)
# end = time.time()
# print("Time elapsed (s):",end-start)
        
#%% Visualize optimal actions in heatmap 
# # Do: x1 (y-axis) vs. x3 (x-axis). Overlay with numbers.
# plt.close('all')

# matrix12 = np.zeros((M+1,M+1))
# matrix13 = np.zeros((M+1,M+1))
# matrix23 = np.zeros((M+1,M+1))
# for x1 in range(M+1):
#     for x2 in range(0,M-x1+1,1):
#         x3 = M - x1 - x2
#         matrix12[x1,x2] = opt_dec_sym[x1,x2,x3,0] # Get action 1 -> 2
#         matrix13[x1,x3] = opt_dec_sym[x1,x2,x3,1] # Get action 1 -> 3
#         matrix23[x2,x3] = opt_dec_sym[x1,x2,x3,2] # Get action 2 -> 3

# ### Matrix 1->2
# text = str(parameters).replace(',', '\n')
# plt.figure()
# plt.imshow(matrix12,alpha=0.8)
# plt.colorbar()
# plt.text(0, 0.1, text, transform=plt.gcf().transFigure)
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
# plt.imshow(matrix13,alpha=0.8)
# plt.colorbar()
# plt.text(0, 0.1, text, transform=plt.gcf().transFigure)
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
# plt.imshow(matrix23,alpha=0.8)
# plt.colorbar()
# plt.text(0, 0.1, text, transform=plt.gcf().transFigure)
# plt.ylabel("x2")
# plt.xlabel("x3")
# plt.title("Asymmetric problem (2->3)")
# plt.subplots_adjust(left=0.3)
# plt.savefig("heatmap_fluid_three_queues_23")

# # Overlay with values
# for (j,i), label in np.ndenumerate(matrix23):
#     plt.text(i,j,round(label),ha='center',va='center')
# plt.show()
        

#%% Visualize g (is it non-convex even without K?)

x = (4,4,4)
num_vals = 30
start = time.time()
X, Y, Z = get_costs(x,num_vals)
end = time.time()
print(end-start)

#%% 3D plotting
a1_vals = np.linspace(-4,4,num_vals)
i = 15 # Each i corresponds to some fixed value of the first action
idx = a1_vals[i]
plt.close('all')
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X[idx], Y[idx], Z[idx], cmap="coolwarm")
# ax.plot_surface(X, Y, Z, cmap="coolwarm")
ax.set_xlabel('1 -> 3')
ax.set_ylabel('2 -> 3')
ax.set_zlabel('g')
ax.set_title(str(x) + ", 1->2: " + str(a1_vals[i]))