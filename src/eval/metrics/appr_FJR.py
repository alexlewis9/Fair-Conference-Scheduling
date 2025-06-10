import numpy as np
import math
from src.models.graph import Graph

def run_appr_FJR(n, k, d, M, loss = 'avg'):
    theta=0
    d_local=np.copy(d)
    l =  math.ceil(n/k)
  
    cost=np.zeros((n))

    ##Uncomment for the Average Loss #####################
    if loss == 'avg':
        size=np.zeros((n))
        for i in range(n):
            for j in range(n):
                if M[i]==M[j]:
                    cost[i]=cost[i]+d[i][j]
                    size[i]=size[i]+1
            cost[i]= cost[i]/size[i]
    #####################################################


    ## Max Loss #########################################
    if loss == 'max':
        for i in range(n):
            for j in range(n):
                if M[i]==M[j] and d_local[i][j]>cost[i]:
                    cost[i]=d_local[i][j]

   ######################################################
        


    while (len(d_local)>=l):
    # Find the fist cluster
        smallest_row_index = np.argmin(np.partition(d_local, l-1, axis=1)[:, l-1])
        smallest_row = d_local[smallest_row_index]
        cluster = np.argsort(smallest_row)[:l]
    
        new_cost=np.zeros(len(cluster))
        
        r=0
        ##Uncomment for the Average Loss #####################
        if loss == 'avg':
            for i  in cluster:
                for j in cluster:
                    new_cost[r]= new_cost[r]+ d_local[i][j]
                r=r+1
            new_cost=new_cost/len(cluster)
        ######################################################

        if loss == 'max':
        ## Max Loss #########################################
            for i  in cluster:
                for j in cluster:
                    if(d_local[i][j]>new_cost[r]):
                        new_cost[r]=  d_local[i][j]
                r=r+1
        ######################################################

    
  
        larger_new_cost=np.max(new_cost) 

        #Find smallest current value
        smallest_old_cost = np.min(cost[cluster])
   
        #calcualate theta
        theta=max(theta,smallest_old_cost/larger_new_cost )

        #remove the agent with smallest current cost in the cluster
        remove_agent = cluster[np.argmin(cost[cluster])]

        d_local = np.delete(np.delete(d_local, remove_agent, axis=0), remove_agent, axis=1)
        cost=np.delete(cost, remove_agent)
        
  
    return theta

def get_appr_FJR(graph: Graph, clusters, loss):
    M = graph.flatten_clusters(clusters)
    n = len(M)
    k = graph.k
    d = graph.adj_mat
    return run_appr_FJR(n, k, d, M, loss=loss)
