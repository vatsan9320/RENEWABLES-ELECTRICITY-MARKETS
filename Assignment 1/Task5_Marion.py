from pyomo.environ import *
import all_data
import numpy as np
import matplotlib.pyplot as plt

hour=1
load_curt_price=500 #€/MWh
conv_gen_outage=8 #the conv gen number 8 has an outage
#import data


data=all_data.get_data()
Pmax_D=np.max(data["load"]["System demand (MW)"])
wind_farm_capacity=200

#create a model
model = ConcreteModel()

# Sets
model.init_conv_G = Set(initialize=[e+1 for e in range(12)])
model.init_wind_farm = Set(initialize=[e+1 for e in range(6)])
model.init_demand=Set(initialize=[e+1 for e in range(len(data["node_demand"]["Load #"]))])


#Declare variables

model.p_up_conv_G = Var(model.init_conv_G, within=NonNegativeReals, initialize=0)

model.p_down_conv_G = Var(model.init_conv_G, within=NonNegativeReals, initialize=0)

model.total_curtailment=Var(within=NonNegativeReals, initialize=0)

##Parameters
model.Pmax_convG=Param(model.init_conv_G, initialize={i+1: pmax for i, pmax in enumerate(data["generation_unit"]["Pmax (MW)"])})
model.price_conv=Param(model.init_conv_G, initialize={i+1: price for i, price in enumerate(data["generation_unit"]["Ci"])})

model.price_wind_farm=Param(model.init_wind_farm, initialize = 0)


#### RESULTS FROM TASK 1 during HOUR 9

DA_h9=10.52
PS_h9={'load': [95.683, 85.611, 158.632, 65.467, 62.949, 120.863, 110.791, 151.078, 153.596, 171.222, 234.172, 171.222, 279.495, 88.129, 294.603, 161.15, 113.309], 
    'conventional generators': [0.0, 0.0, 0.0, 0.0, 0.0, 155.0, 155.0, 400.0, 400.0, 300.0, 236.985, 0.0], 
    'wind farms': [163.297, 141.239, 142.903, 139.414, 140.617, 143.52]}  
DS_total=np.sum(PS_h9["load"])

# Objective function : Minimization of balancing cost


def objective_rule(model):
    return (sum((DA_h9+10/100*model.price_conv[k])*model.p_up_conv_G[k] for k in model.init_conv_G) 
            + load_curt_price*model.total_curtailment
            - sum((DA_h9-15/100*model.price_conv[i]) * model.p_down_conv_G[i] for i in model.init_conv_G))
model.balancing_cost = Objective(rule=objective_rule, sense=minimize)




# Constraints
def balance(model):
    return (sum(model.p_up_conv_G[k] - model.p_down_conv_G[k] for k in model.init_conv_G) 
            + model.total_curtailment
            == PS_h9["conventional generators"][conv_gen_outage-1]
             + 15/100*sum(PS_h9["wind farms"][i] for i in range(3))
             - 10/100*sum(PS_h9["wind farms"][j] for j in range(3,6)))
model.balance_constraint = Constraint(rule=balance)

def max_up(model, i):
    return model.p_up_conv_G[i] <= model.Pmax_convG[i] - PS_h9["conventional generators"][i-1]
model.max_up_constraint = Constraint(model.init_conv_G, rule=max_up)

def max_demand_curtailment(model):
    return model.total_curtailment <= DS_total
model.max_curtailment_constraint = Constraint(rule=max_demand_curtailment)

def max_down(model, i):
    return model.p_down_conv_G[i] <= PS_h9["conventional generators"][i-1]
model.max_down_constraint = Constraint(model.init_conv_G, rule=max_down)

model.constraint_fix_p_up_conv_G8 = Constraint(expr=model.p_up_conv_G[8] == 0)
model.constraint_fix_p_down_conv_G8 = Constraint(expr=model.p_down_conv_G[8] == 0)

#Dual variables
model.dual = Suffix(direction=Suffix.IMPORT)

# Create a solver 
solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

# Solve the model
solution = solver.solve(model, tee=True)

model.display()

# Store the results
results = {}
results["curtailment"] = round(value(model.total_curtailment))
results["up"] = [round(value(model.p_up_conv_G[key]),3) for key in model.init_conv_G]
results["down"] = [round(value(model.p_down_conv_G[key]),3) for key in model.init_conv_G]
results["wind farm compared to PS"] = [round(15/100*PS_h9["wind farms"][i],3) for i in range(3)]+[round(-10/100*PS_h9["wind farms"][i],3) for i in range(3,6)]
print(results)
print("Balancing cost =", round(model.balancing_cost(), 3))

#print(solution)

# Extract the market-clearing price 
if solution.solver.termination_condition == TerminationCondition.optimal:
    price = model.dual[model.balance_constraint]
    print(f"Balancing price = : {price:.2f}")
else:
    print("Solver did not find an optimal solution. Balancing price cannot be calculated.")

# Scheduled power for Generator 8 in DA Market
p_scheduled_G8 = PS_h9["conventional generators"][conv_gen_outage - 1]

# Power lost due to outage
p_lost_G8 = p_scheduled_G8

# Penalty: Generator 8 must pay the balancing price for lost power
penalty_G8 = round(p_lost_G8 * price, 3)

# Day-Ahead Market Profits (Fixed)
profit = {}
profit["conventional generators"] = [
    round(prod * DA_h9 - prod * price, 3) if i != conv_gen_outage - 1 else 0
    for i, (prod, price) in enumerate(zip(PS_h9["conventional generators"], data["generation_unit"]["Ci"]))
]
profit["wind farms"] = [round(value * DA_h9, 3) for value in results["wind farm compared to PS"]]

# Balancing Market Profits (Two-Pricing Scheme)
profit_balancing_two_price = {}
profit_balancing_two_price["conventional generators"] = [
    round(prod * (price - DA_h9), 3) if i != conv_gen_outage - 1 else -penalty_G8
    for i, (prod, price) in enumerate(zip(results["down"], data["generation_unit"]["Ci"]))
]
profit_balancing_two_price["wind farms"] = [round(value * price, 3) for value in results["wind farm compared to PS"]]

# Balancing Market Profits (One-Pricing Scheme) - Same price for up/down
profit_balancing_one_price = {}
profit_balancing_one_price["conventional generators"] = [
    round((prod_up - prod_down) * price, 3) if i != conv_gen_outage - 1 else -penalty_G8
    for i, (prod_up, prod_down, price) in enumerate(zip(results["up"], results["down"], data["generation_unit"]["Ci"]))
]
profit_balancing_one_price["wind farms"] = [round(value * price, 3) for value in results["wind farm compared to PS"]]

# Display Profits
print("\nDay-Ahead Market Profits:", profit)
print("\nBalancing Market Profits (One-Pricing Scheme):", profit_balancing_one_price)
print("\nBalancing Market Profits (Two-Pricing Scheme):", profit_balancing_two_price)

# Compare Imbalance Settlement Schemes
one_price_scheme = profit_balancing_one_price
two_price_scheme = profit_balancing_two_price

print("\nComparison of Imbalance Settlement Schemes:")
print("One-Price Scheme:", one_price_scheme)
print("Two-Price Scheme:", two_price_scheme)

# Conclusion
if sum(one_price_scheme["conventional generators"]) > sum(two_price_scheme["conventional generators"]):
    print("One-price scheme benefits balancing service providers more.")
else:
    print("Two-price scheme provides a more stable settlement.")
