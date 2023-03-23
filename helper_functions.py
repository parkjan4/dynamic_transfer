import numpy as np
import matplotlib.pyplot as plt
import csv

#%% Visualization
def save_heatmap_2D(V,pol,params):   
    # This function generates and saves heatmaps of value functions and optimal policy  
    if len(V)>0:
        text = str(params).replace(',', '\n')
        plt.figure()
        plt.imshow(V)
        plt.colorbar()
        # plt.text(0, 0.3, text, transform=plt.gcf().transFigure)
        # for (j,i), label in np.ndenumerate(V):
        #     plt.text(i,j,round(label),ha='center',va='center')
        # plt.show()
        plt.ylabel("$x_2$",fontsize=14)
        plt.xlabel("$x_1$",fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title("Value functions")
        plt.subplots_adjust(left=0.3)
        plt.tight_layout()
        plt.savefig("heatmap_value_fns", bbox_inches="tight")
        # plt.clf()
    
    fsize = 14
    ofsize = fsize-2 # overlay text font size
    if len(pol)>0:
        plt.figure()
        plt.imshow(pol)
        # plt.clim(0,20)
        cbar = plt.colorbar(location="top",shrink=0.33)
        cbar.ax.tick_params(labelsize=fsize)
        # plt.text(0, 0.3, text, transform=plt.gcf().transFigure)
        for (j,i), label in np.ndenumerate(pol):
            plt.text(i,j,round(label),ha='center',va='center',fontsize=ofsize,color="#cccccc")
        plt.show()
        plt.ylabel("$x_1+x_2$",fontsize=fsize)
        plt.xlabel("$x_2$",fontsize=fsize)
        # n = pol.shape[0]
        # plt.yticks(ticks=np.arange(0,n,1),fontsize=18)
        # plt.xticks(ticks=np.arange(0,n,1),fontsize=18)
        plt.yticks(ticks=np.arange(0,21,1),labels=np.arange(0,21,1),fontsize=fsize)
        plt.xticks(ticks=np.arange(0,21,1),labels=np.arange(0,21,1),fontsize=fsize)
        plt.title("Optimal policy ($I^\star$)",fontsize=fsize)
        plt.subplots_adjust(left=0.3)
        plt.tight_layout()
        plt.savefig("heatmap_policy", bbox_inches="tight")
    
    
def save_heatmap_3D(V,pol,params):
    text = str(params).replace(',', '\n')
    plt.figure()
    

    
def save_to_csv(arr, name):
    # This function saves a given array as a csv file
    with open(name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(arr)
        

def generate_step(params,arrx,arry,xlab,ylab,title):
    text = str(params).replace(',', '\n')
    plt.figure()
    plt.step(arrx,arry,where='post')
    plt.text(0, 0.3, text, transform=plt.gcf().transFigure)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.subplots_adjust(left=0.3)