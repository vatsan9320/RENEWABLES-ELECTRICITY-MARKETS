# # ##### TASK 2.3 Trade-off between optimal reserve bid and expected reserve shortfall

# Energinet Perspective: Take the perspective of Energinet and investigate how varying
# the P90 requirement (e.g., by adjusting the allowed frequency of reserve shortfall between
# 80% and 100%) impacts the optimal reserve bid (in-sample analysis using ALSO-X) and the
# expected reserve shortfall (out-of-sample analysis). Analyze whether there is an observable
# trade-off between these two factors as the P90 requirement is relaxed or tightened.

# # ####

import Task_2_all_data 
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt

## Data
data=Task_2_all_data.get_data()
nbr_in_sample=100
nbr_seconds=60

epsilon=[0, 2.5/100, 5/100, 7.5/100, 10/100, 12.5/100, 15/100, 17.5/100, 20/100]

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


def ALSO_X_MILP(in_sample, epsilon): #Computationnally expensive when many samples because it is a MILP
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


ALSO_results={}
ALSO_optimal_bid=[]
ALSO_shortfall={}
proba_reserve_shortfall={}

for eps in epsilon:
    ALSO_results[f"Epsilon={eps}"]=ALSO_X_MILP(in_sample_profiles, eps)
    ALSO_shortfall[f"Epsilon={eps}"]=[]
    for i in range(len(out_of_sample_profiles)):
        ALSO_shortfall[f"Epsilon={eps}"].append(ALSO_results[f"Epsilon={eps}"]["c_up"]*np.ones(60)-out_of_sample_profiles[f"Profile{i+1}"])
        
    proba_reserve_shortfall[f"Epsilon={eps}"]=np.sum(np.array(ALSO_shortfall[f"Epsilon={eps}"]) > 0)/len(ALSO_shortfall[f"Epsilon={eps}"])/60*100
    
    
   
 

print("Epsilon", epsilon)
print("ALSO-X MILP", [ALSO_results[f"Epsilon={eps}"]["c_up"] for eps in epsilon])
print("Proba of reserve shortfall=overbidding for each eps",[proba_reserve_shortfall[f"Epsilon={eps}"] for eps in epsilon])


# plt.plot(epsilon, [ALSO_results[f"Epsilon={eps}"]["c_up"] for eps in epsilon], label="Optimal Reserve Bid (kW ??)")
# plt.plot(epsilon, [proba_reserve_shortfall[f"Epsilon={eps}"] for eps in epsilon], label="Overbidding probability")
# plt.xlabel("Epsilon")
# plt.legend()
# plt.show()

fig, ax1 = plt.subplots()

# Axis for ALSO_results
color1 = 'tab:blue'
ax1.set_xlabel("Allowed frequence of reserve shortfall (%)")
ax1.set_ylabel("Optimal Reserve Bid (kW)", color=color1)
ax1.plot([eps*100 for eps in epsilon], [ALSO_results[f"Epsilon={eps}"]["c_up"] for eps in epsilon], label="Optimal Reserve Bid", color='tab:blue')
ax1.tick_params(axis='y', labelcolor=color1)

# Axis for proba_reserve_shortfall
ax2 = ax1.twinx()  # Second axis y
color2 = 'tab:orange'
ax2.set_ylabel("Overbidding Probability (%)", color=color2)
ax2.plot([eps*100 for eps in epsilon], [proba_reserve_shortfall[f"Epsilon={eps}"] for eps in epsilon], label="Overbidding Probability", color='tab:orange')
ax2.tick_params(axis='y', labelcolor=color2)

# Add combined legend
fig.tight_layout()  # To avoid legend overlapping
plt.show()
