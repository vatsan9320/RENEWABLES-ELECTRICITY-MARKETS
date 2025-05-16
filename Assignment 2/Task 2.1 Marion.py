# ##### TASK 2.1 : In Sample Decision Making with ALSO-X and CVaR
#
# In-sample Decision Making: Offering Strategy Under the P90 Requirement:
# Given the P90 requirement of Energinet, determine the optimal reserve capacity bid (in
# kW) of the stochastic load in the FCR-D UP market for the given hour. Utilize both
# ALSO-X and CVaR techniques to solve this problem, as these methods may yield different
# results.
#
# ####

import Task_2_all_data 
import numpy as np
from pyomo.environ import *
import time

start = time.time()


## Data
data=Task_2_all_data.get_data()
nbr_in_sample=100
nbr_seconds=60
epsilon=10/100
## Construction of the In-Sample Scenarios
np.random.seed=42
def create_in_sample():
    in_sample={}
    i=1
    while len(in_sample)<nbr_in_sample:
        
        scen=data["load"][f"Profile{np.random.randint(1, len(data['load']))}"]

        if scen not in in_sample.values():
            in_sample[f"Profile{i}"]=scen
            i+=1

    return in_sample
in_sample_profiles=create_in_sample()
print(in_sample_profiles["Profile1"])

def ALSO_X_MILP(in_sample): #Computationnally expensive when many samples because it is a MILP
    ## Create a model
    model = ConcreteModel()

    ## Define the sets

    model.init_time=RangeSet(nbr_seconds)
    model.init_profiles=RangeSet(nbr_in_sample)

    ## Declare the variables

    #Reserve capacity bid in kW
    model.c_up = Var(within=NonNegativeReals, initialize=0) 
    #Binary variables : 1 per minute per sample
    model.y = Var(model.init_time, model.init_profiles, within=Binary, initialize=0)


    ## Declare parameters
    #model.proba = Param(model.init_profiles, initialize=1/nbr_in_sample)
    # Probability distribution of the FCR-D Up service availability per minute m of the given hour
    model.F = Param(model.init_time, model.init_profiles, initialize={(m, w): in_sample[f"Profile{w}"][m-1] 
                                                for m in model.init_time for w in model.init_profiles})

    M=10000
    q=epsilon*nbr_in_sample*nbr_seconds #Budget for violation

    ## Objective function

    def objective_rule(model):
        return model.c_up

    model.optimal_reserve_cap_bid = Objective(rule=objective_rule, sense= maximize)

    ## Constraints

    def probabilistic(model,m,w):
        return model.c_up - model.F[m,w] <= model.y[m,w]*M
    model.probabilistic_constraint=Constraint(model.init_time, model.init_profiles, rule=probabilistic)

    def violation(model):
        return sum(model.y[m,w] for m in model.init_time for w in model.init_profiles) <= q
    model.violation_constraint=Constraint(rule=violation)


    ## Solve the problem

    #Dual variables
    model.dual = Suffix(direction=Suffix.IMPORT)

    # Create a solver 
    solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

    # Solve the model
    solution = solver.solve(model, tee=True)

    print("Optimal reserve capacity bid (kW):", model.optimal_reserve_cap_bid())

    # Store the results
    results = {"binary":{}, "c_up":{}, "total violation":{}}

    results["c_up"] = value(model.optimal_reserve_cap_bid)
    
    for m in model.init_time :
        results["binary"][f"Minute {m}"] = [(value(model.y[m,w])) for w in model.init_profiles]
        
    results["total violation"]["sum"]=sum(value(model.y[m,w]) for m in model.init_time for w in model.init_profiles)

    return results

def ALSO_X_relaxed(in_sample): #Every binary variable is relaxed between 0 and 1 : it solves several LP until the tolerance is reached
    low_q=0
    up_q=epsilon*nbr_in_sample*nbr_seconds
    delta=10**-5

    while up_q-low_q>=delta:
        mean_q=(up_q+low_q)/2

        ## Create a model
        model = ConcreteModel()

        ## Define the sets

        model.init_time=RangeSet(nbr_seconds)
        model.init_profiles=RangeSet(nbr_in_sample)

        ## Declare the variables

        #Reserve capacity bid in kW
        model.c_up = Var(within=NonNegativeReals, initialize=0) 
        #Binary variables : 1 per minute per sample
        model.y = Var(model.init_time, model.init_profiles, within=NonNegativeReals, bounds=(0,1),  initialize=0)


        ## Declare parameters
        #model.proba = Param(model.init_profiles, initialize=1/nbr_in_sample)
        # Probability distribution of the FCR-D Up service availability per minute m of the given hour
        model.F = Param(model.init_time, model.init_profiles, initialize={(m, w): in_sample[f"Profile{w}"][m-1] 
                                                    for m in model.init_time for w in model.init_profiles})

        M=10000
        q=mean_q #Budget for violation

        ## Objective function

        def objective_rule(model):
            return model.c_up

        model.optimal_reserve_cap_bid = Objective(rule=objective_rule, sense= maximize)

        ## Constraints

        def probabilistic(model,m,w):
            return model.c_up - model.F[m,w] <= model.y[m,w]*M
        model.probabilistic_constraint=Constraint(model.init_time, model.init_profiles, rule=probabilistic)

        def violation(model):
            return sum(model.y[m,w] for m in model.init_time for w in model.init_profiles) <= q
        model.violation_constraint=Constraint(rule=violation)


        ## Solve the problem

        #Dual variables
        model.dual = Suffix(direction=Suffix.IMPORT)

        # Create a solver 
        solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

        # Solve the model
        solution = solver.solve(model, tee=True)

        print("Optimal reserve capacity bid (kW):", model.optimal_reserve_cap_bid())

        # Store the results
        results = {"binary":{}, "c_up":{}, "total violation":{}}

        results["c_up"] = value(model.optimal_reserve_cap_bid)
        
        count=0

        for m in model.init_time :
            results["binary"][f"Minute {m}"] = [(value(model.y[m,w])) for w in model.init_profiles]
            count+=results["binary"][f"Minute {m}"].count(0)
 
        results["total violation"]["sum"]=sum(value(model.y[m,w]) for m in model.init_time for w in model.init_profiles)

        print("Count", count, "proba", count/nbr_in_sample/nbr_seconds)

        if count/(nbr_in_sample*nbr_seconds) > 1-epsilon:
            low_q=mean_q
        else:
            up_q=mean_q

    return results

def CVaR_approximation(in_sample):
    ## Create a model
    model = ConcreteModel()

    ## Define the sets

    model.init_time=RangeSet(nbr_seconds)
    model.init_profiles=RangeSet(nbr_in_sample)

    ## Declare the variables

    # Reserve capacity bid in kW
    model.c_up = Var(within=NonNegativeReals, initialize=0)

    # VaR (ksi) 1 per minute per sample
    model.ksi = Var(model.init_time, model.init_profiles, within=Reals, initialize=0)

    model.beta = Var(within=NonPositiveReals, initialize=0)
    ## Declare parameters
    
    # Probability distribution of the FCR-D Up service availability per minute m of the given hour
    model.F = Param(model.init_time, model.init_profiles, initialize={(m, w): in_sample[f"Profile{w}"][m-1] 
                                                for m in model.init_time for w in model.init_profiles})


    ## Objective function

    def objective_rule(model):
        return model.c_up

    model.optimal_reserve_cap_bid = Objective(rule=objective_rule, sense= maximize)

    ## Constraints

    def max_var(model,m,w):
        return model.c_up - model.F[m,w] <= model.ksi[m,w]
    model.max_var_constraint=Constraint(model.init_time, model.init_profiles, rule=max_var)

    def min_CVaR(model):
        return 1/(nbr_in_sample*nbr_seconds)*sum(model.ksi[m,w] for m in model.init_time for w in model.init_profiles) <= (1-epsilon)*model.beta
    model.min_CVaR_constraint=Constraint(rule=min_CVaR)

    def beta_var(model,m,w):
        return model.beta <= model.ksi[m,w]
    model.beta_var_constraint=Constraint(model.init_time, model.init_profiles, rule=beta_var)

    ## Solve the problem

    #Dual variables
    model.dual = Suffix(direction=Suffix.IMPORT)

    # Create a solver 
    solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

    # Solve the model
    solution = solver.solve(model, tee=True)

    print("Optimal reserve capacity bid (kW):", model.optimal_reserve_cap_bid())

    # Store the results
    results = {"ksi":{}, "c_up":{}, "beta":{}}

    results["c_up"] = round(value(model.optimal_reserve_cap_bid),2)
    results["beta"] = round(value(model.beta),2)
    
    for m in model.init_time :
        results["ksi"][f"Minute {m}"] = [(value(model.ksi[m,w])) for w in model.init_profiles]
        
    
    return results
    

    

ALSO_results=ALSO_X_MILP(in_sample_profiles)
CVaR_approximation_results=CVaR_approximation(in_sample_profiles)

print("Execution time", time.time() - start, "seconds")

print("ALSO-X MILP", ALSO_results["c_up"])
print("CVaR approximation", CVaR_approximation_results["c_up"], "beta", CVaR_approximation_results["beta"])

#ALSO_relaxed_results=ALSO_X_relaxed(in_sample_profiles)
#print("Relaxed LP", ALSO_relaxed_results["c_up"], "MILP", ALSO_results["c_up"])
