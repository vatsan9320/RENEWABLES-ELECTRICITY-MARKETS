# # ##### TASK 2.2 : Verif of the P90 Requirement Using Out-of-Sample Analysis
# #
# Verification of the P90 Requirement Using Out-of-Sample Analysis: Using the
# 200 testing profiles, verify whether the P90 requirement is satisfied for both solution tech-
# niques used. This step does not require solving any optimization problem. Instead, compare
# the optimal reserve bid obtained in Step 2.1 with the actual power consumption under each
# testing profile. For example, if the stochastic load bids 300 kW to the FCR-D UP market
# for a given hour, and its consumption at a certain minute within that hour is 270 kW, this
# would indicate a reserve shortfall of 30 kW for that minute
# # ####

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

def create_out_of_sample(in_sample):
    out_of_sample={}
    i=1
    for j in range(1,len(data['load'])+1):
        if data["load"][f"Profile{j}"] not in in_sample.values():
            out_of_sample[f"Profile{i}"]=data["load"][f"Profile{j}"]
            i+=1
    return out_of_sample

out_of_sample_profiles=create_out_of_sample(in_sample_profiles)



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

#ALSO_relaxed_results=ALSO_X_relaxed(in_sample_profiles)
CVaR_approximation_results=CVaR_approximation(in_sample_profiles)

def P90_requirement():
    ALSO_shortfall=[]
    CVaR_shortfall=[]
    for i in range(len(out_of_sample_profiles)):
        ALSO_shortfall.append(ALSO_results["c_up"]*np.ones(60)-out_of_sample_profiles[f"Profile{i+1}"])
        CVaR_shortfall.append(CVaR_approximation_results["c_up"]*np.ones(60)-out_of_sample_profiles[f"Profile{i+1}"])
    proba_reserve_shortfall_ALSO= np.sum(np.array(ALSO_shortfall) > 0)/len(ALSO_shortfall)/60*100
    proba_reserve_shortfall_CVaR= np.sum(np.array(CVaR_shortfall) > 0)/len(CVaR_shortfall)/60*100

    return (
        f"ALSO-X Reserve Bid: {ALSO_results['c_up']} kW\n"
        f"Violations: {np.sum(np.array(ALSO_shortfall) > 0)} out of {len(ALSO_shortfall) * 60}\n"
        f"Probability of overbidding: {round(proba_reserve_shortfall_ALSO,2)}%\n"
        f"P90 Satisfied: {'Yes' if proba_reserve_shortfall_ALSO <= epsilon*100 else 'No'}\n\n"
        f"CVaR Reserve Bid: {CVaR_approximation_results['c_up']} kW\n"
        f"Violations: {np.sum(np.array(CVaR_shortfall) > 0)} out of {len(CVaR_shortfall) * 60}\n"
        f"Probability of overbidding: {round(proba_reserve_shortfall_CVaR,2)}%\n"
        f"P90 Satisfied (overbidding proba <= 10%): {'Yes' if proba_reserve_shortfall_CVaR <= epsilon*100 else 'No'}")

print(P90_requirement())
print("Execution time", time.time() - start, "seconds")