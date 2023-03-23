# Jangwon Park, 2022

import numpy as np
import matplotlib.pyplot as plt

#%% 
def compute_cost(x1,x2,I,params):
    # Compute one-period cost 
    
    lambda1 = params['lambda1']
    lambda2 = params['lambda2']
    mu1 = params['mu1']
    mu2 = params['mu2']
    K = params['K']
    m = params['m']
    h1 = params['h1']
    h2 = params['h2']
    tau = params['tau']
    
    sigma1 = (x1 - I) / (mu1 - lambda1)
    sigma2 = (x2 + I) / (mu2 - lambda2)
    
    if (sigma1 < tau) and (sigma2 < tau):
        cost = h1*((x1 - I)*sigma1 + 1/2*(lambda1 - mu1)*sigma1**2) \
                 + h2*((x2 + I)*sigma2 + 1/2*(lambda2 - mu2)*sigma2**2) \
                 + m*np.abs(I) + K*(I!=0) 
    elif (sigma1 >= tau) and (sigma2 < tau):
        cost = h1*((x1 - I)*tau + 1/2*(lambda1 - mu1)*tau**2) \
                 + h2*((x2 + I)*sigma2 + 1/2*(lambda2 - mu2)*sigma2**2) \
                 + m*np.abs(I) + K*(I!=0)
    elif (sigma1 < tau) and (sigma2 >= tau):
        cost = h1*((x1 - I)*sigma1 + 1/2*(lambda1 - mu1)*sigma1**2) \
                 + h2*((x2 + I)*tau + 1/2*(lambda2 - mu2)*tau**2) \
                 + m*np.abs(I) + K*(I!=0)
    elif (sigma1 >= tau) and (sigma2 >= tau):
        cost = h1*((x1 - I)*tau + 1/2*(lambda1 - mu1)*tau**2) \
                 + h2*((x2 + I)*tau + 1/2*(lambda2 - mu2)*tau**2) \
                 + m*np.abs(I) + K*(I!=0)
    return cost
    
#%% Visualize structure of one-period cost g()

plt.close('all')

parameters = {
    'lambda1': 0.1, # Arrival rate to hospital 1
    'lambda2': 0.1, # Arrival rate to hospital 2
    'mu1': 1, # Service rate at hospital 1
    'mu2': 1, # Service rate at hospital 2
    'K': 5, # Fixed cost of transfer
    'm': 2, # Unit transfer cost per patient
    'h1': 1, # Unit holding cost per patient at hospital 1
    'h2': 1, # Unit holding cost per patient at hospital 2
    'tau': 10, # Fixed length between two decision epochs
    }

# #%% Structure of cost function g
# plt.close('all')

# plt.figure()
# configs = [(9,1),(8,2),(7,3),(6,4)]
# # var_costs = range(0,5,1)
# # fixed_costs = [2,4,6]
# # taus = [5,10,20]
# holding_costs = [1,1.5,2,2.5]
# for x1_0, x2_0 in configs:
#     # transfers = range(-x2_0,x1_0+1,1)
#     transfers = np.sort(np.append(np.linspace(-x2_0,x1_0,num=10000),0))
#     best_obj = float("inf")
#     g = []
#     for I in transfers:
#         cost = compute_cost(x1_0,x2_0,I,parameters)
#         g.append(cost)
#         if cost <= best_obj:
#             best_obj = cost
#             minimizer = I
    
#     print("Optimal action: ", minimizer)
#     plt.plot(transfers, g)
    
# plt.xlabel("I")
# plt.ylabel("Cost (g)")
# legend = []
# for config in configs:
#     legend.append(str(config))
# plt.legend(legend)
# # plt.legend(["m=0","m=1","m=2","m=3","m=4"])
# # plt.legend(["K=2","K=4","K=6"])
# # plt.legend(["tau=5","tau=10","tau=20"])
# # plt.legend(["h1=1.0","h1=1.5","h1=2.0","h1=2.5"])
# plt.grid(alpha=0.3)
# plt.title("g(x,I)")


#%% Compute optimal actions, symmetric problem

M = 10 # number of patients in system
opt_dec_sym = np.zeros((M+1,M+1)) # Save optimal decisions

# symmetric problem
for x1 in range(M+1):
    for x2 in range(M+1):
        transfers = range(-x2,x1+1,1)
        # transfers = np.sort(np.append(np.linspace(-x2,x1,num=1000),0))
        best_obj = float("inf")
        for I in transfers:
            cost = compute_cost(x1,x2,I,parameters)
            if cost <= best_obj:
                best_obj = cost
                best_action = I
        opt_dec_sym[x1,x2] = best_action

#%% Show optimal actions in a heatmap
text = str(parameters).replace(',', '\n')
plt.figure()
plt.imshow(opt_dec_sym,alpha=0.8)
plt.colorbar()
plt.text(0, 0.3, text, transform=plt.gcf().transFigure)
plt.ylabel("x1")
plt.xlabel("x2")
plt.title("Symmetric problem")
plt.subplots_adjust(left=0.3)
plt.savefig("heatmap_opt_dec_sym")

# Overlay with values
for (j,i), label in np.ndenumerate(opt_dec_sym):
    plt.text(i,j,round(label),ha='center',va='center')
plt.show()


#%% Compute optimal actions, asymmetric problem

M = 10 # number of patients in system
opt_dec_asym = np.zeros((M+1,M+1)) # Save optimal decisions

# asymmetric problem
parameters.update({'h1': 2}) 
for x1 in range(M+1):
    for x2 in range(M+1):
        transfers = range(-x2,x1+1,1)
        # transfers = np.sort(np.append(np.linspace(-x2,x1,num=1000),0))
        best_obj = float("inf")
        for I in transfers:
            cost = compute_cost(x1,x2,I,parameters)
            if cost <= best_obj:
                best_obj = cost
                best_action = I
        opt_dec_asym[x1,x2] = best_action
    
    
#%% Show optimal actions in a heatmap
text = str(parameters).replace(',', '\n')
plt.figure()
plt.imshow(opt_dec_asym,alpha=0.8)
plt.colorbar()
plt.text(0, 0.3, text, transform=plt.gcf().transFigure)
plt.ylabel("x1")
plt.xlabel("x2")
plt.title("Asymmetric problem")
plt.subplots_adjust(left=0.3)
plt.savefig("heatmap_opt_dec_asym")

# Overlay with values
for (j,i), label in np.ndenumerate(opt_dec_asym):
    plt.text(i,j,round(label),ha='center',va='center')
plt.show()
