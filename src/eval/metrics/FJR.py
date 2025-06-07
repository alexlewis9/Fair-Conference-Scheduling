import gurobipy as gp
from gurobipy import GRB
import numpy as np

def run_FJR(n, k, d, M, theta, loss='avg'):

  
    alpha_max=1
    cost=np.zeros((n))
    ##Uncomment for the Average Loss #####################
    if loss=='avg':
        size=np.zeros((n))
        for i in range(n):
            for j in range(n):
                if M[i]==M[j]:
                    cost[i]=cost[i]+d[i][j]
                    size[i]=size[i]+1
            cost[i]= cost[i]/size[i]
    #####################################################

    ### Max Loss #########################################
    if loss=='max':
        for i in range(n):
            for j in range(n):
                if M[i]==M[j] and d[i][j]>cost[i]:
                    cost[i]=d[i][j]
    #####################################################


  
    L1 = 10000 # Fill in L1 value
    L2 = 10000 # Fill in L2 value

    epsilon = 0.01  # Desired precision
    lower_bound = 1.0  # Fill in lower bound for alpha
    upper_bound = theta*4# Fill in upper bound for alpha
    flag=0
    condition_meet=0
    while upper_bound - lower_bound > epsilon and condition_meet==0:
        alpha = (upper_bound + lower_bound) / 2
        # Variables
        model = gp.Model("optimization_problem")
        model.Params.OutputFlag = 0
        x = {}
        for i in range(n):
            x[i] = model.addVar(vtype=GRB.BINARY, name="x_{}".format(i))  # indicating if agent i deviates
        Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

       

        # Constraints
        model.addConstr(gp.quicksum(x[i] for i in range(n)) >= n / k)

         ##Uncomment for the Average Loss #####################
        # for i in range(n):
        #     model.addConstr(gp.quicksum(x[j] * d[i][j] for j in range(n)) <= Z  + L1 * (1 - x[i]))  # cost under the new deviating coalition 
        #     model.addConstr(gp.quicksum(x[j] for j in range(n)) *cost[i]>= Z *  alpha - L2 * (1 - x[i]))  # previous cost 
        #####################################################

        ### Max Loss #########################################
        for i in range(n):
            for j in range(n):
                model.addConstr(x[j] * d[i][j]  <= Z  + L1 * (1 - x[i]))  # cost under the new deviating coalition 

        for i in range(n): 
            model.addConstr(cost[i] - Z * alpha >= - L2 * (1 - x[i]), name=f"cost_constraint_{i}")
        #####################################################
           

        model.optimize()
  
        if model.status == GRB.OPTIMAL:
            flag=1
            alpha_max=max(alpha_max,alpha)
            lower_bound = alpha
        else:
            upper_bound = alpha
    if flag==0:
        alpha=1
    if alpha>10:
        print(alpha)    
    return alpha

def get_FJR(graph, clusters, theta, loss):
    M = graph.flatten_clusters(clusters)
    n = len(M)
    k = graph.k
    d = graph.adj_mat
    return run_FJR(n, k, d, M, theta, loss=loss)

# Example data
# n =4 # Example dimension
# k = 2
# d = [[0, 10, 2 ,2],
#      [10, 0, 2 ,2],
#      [2, 2, 0 ,20],
#      [2, 2, 20 ,0]]  # Example matrix
# M = [1, 1, 0, 0]
# theta=100
# solution = run_FJR(n, k, d, M,1)
