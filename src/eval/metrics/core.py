import gurobipy as gp
from gurobipy import GRB
import numpy as np
from src.models.graph import Graph

def run_core(n, k, d, M, theta, loss='avg'):
    alpha_max = 1
    cost = np.zeros((n))

    if loss == 'avg':
        size=np.zeros((n))
        for i in range(n):
            for j in range(n):
                if M[i]==M[j]:
                    cost[i]=cost[i]+d[i][j]
                    size[i]=size[i]+1
            cost[i]= cost[i]/size[i]
    elif loss == 'max':
        for i in range(n):
            for j in range(n):
                if M[i] == M[j] and d[i][j] > cost[i]:
                    cost[i] = d[i][j]

    L = 1000000  # Fill in L1 value
    epsilon = 0.01  # Desired precision
    lower_bound = 1.0  # Fill in lower bound for alpha
    upper_bound = 4 * theta  # Fill in upper bound for alpha
    flag = 0
    condition_meet = 0
    while upper_bound - lower_bound > epsilon and condition_meet == 0:
        alpha = (upper_bound + lower_bound) / 2
        model = gp.Model("optimization_problem")
        model.Params.OutputFlag = 0
        x = {}
        for i in range(n):
            x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")  # indicating if agent i deviates

        model.addConstr(gp.quicksum(x[i] for i in range(n)) >= n / k)

        ##Uncomment for the Average Loss #####################
        if loss == 'avg':
            for i in range(n):
                model.addConstr(gp.quicksum(x[j] * d[i][j] for j in range(n))*alpha <= cost[i]*gp.quicksum(x[j] for j in range(n))   + L * (1 - x[i]))  # cost under the new deviating coalition
            #####################################################

        ## Max Loss #########################################
        if loss == 'max':
            for i in range(n):
                for j in range(n):
                    model.addConstr(
                        x[j] * d[i][j] * alpha <= cost[i] + L * (1 - x[i]))  # cost under the new deviating coalition
            #####################################################

        model.optimize()
        if model.status == GRB.OPTIMAL:
            flag = 1
            alpha_max = max(alpha_max, alpha)
            lower_bound = alpha
            solution = {}
            for v in model.getVars():
                solution[v.varName] = v.x
        else:
            upper_bound = alpha

    if flag == 0:
        alpha = 1
    return alpha

def get_core(graph: Graph, clusters, theta, loss):
    """ Please run approx_jfr first to get theta."""
    M = graph.flatten_clusters(clusters)
    n = len(M)
    k = graph.k
    d = graph.adj_mat
    return run_core(n, k, d, M, theta, loss=loss)

