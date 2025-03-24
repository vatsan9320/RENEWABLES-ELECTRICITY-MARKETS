from pyomo.environ import *
import all_data
import numpy as np


#Single hour copper plate
hour = 20

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
# model.p_demand= Var(model.init_demand, within=NonNegativeReals, initialize=Pmax_D/2)
# model.p_conv_G = Var(model.init_conv_G, within=NonNegativeReals, initialize=0)
# model.p_wind_farm = Var(model.init_wind_farm, within=NonNegativeReals, initialize=0)

model.rU = Var(model.init_conv_G, within=NonNegativeReals,  initialize=0)  # Upward reserve
model.rD = Var(model.init_conv_G, within=NonNegativeReals, initialize=0)  # Downward reserve
#model.pG = Var(model.init_conv_G, within=NonNegativeReals)  # Generation output


#Parameters

model.Pmax_convG=Param(model.init_conv_G, initialize={i+1: pmax for i, pmax in enumerate(data["generation_unit"]["Pmax (MW)"])})
model.RUmax = Param(model.init_conv_G, initialize={i+1: rumax for i, rumax in enumerate(data["generation_unit"]["R+ (MW)"])})
model.RDmax = Param(model.init_conv_G, initialize={i+1: rdmax for i, rdmax in enumerate(data["generation_unit"]["R- (MW)"])})
model.priceup_conv=Param(model.init_conv_G, initialize={i+1: price for i, price in enumerate(data["generation_unit"]["Cu_i"])})
model.pricedown_conv=Param(model.init_conv_G, initialize={i+1: price for i, price in enumerate(data["generation_unit"]["Cd_i"])})

# model.reserve_cost = Param(model.init_conv_G, initialize={g: data["generation_unit"]["Cu_i"][g-1] for g in active_generators})
# model.demand = Param(model.hour, initialize={hour: data["load"]["System demand (MW)"][hour-1]})  
# model.gen_cost = Param(model.generators, initialize={g: data["generation_unit"]["Ci"][g-1] for g in active_generators})



# Objective function : Minimization of Reserve
def objective_rule(model):
    return sum(model.priceup_conv[k] * model.rU[k] for k in model.init_conv_G) + \
       sum(model.pricedown_conv[k] * model.rD[k] for k in model.init_conv_G)

model.reservemarket = Objective(rule=objective_rule, sense=minimize)

# Constraints

# 1. Upward Reserve Procurement Requirement (15% of demand)
def upward_reserve_req(model):
    return sum(model.rU[g] for g in model.init_conv_G) == 0.15 * data["load"]["System demand (MW)"][hour-1]
model.upward_reserve_constraint = Constraint(rule=upward_reserve_req)

# 2. Downward Reserve Procurement Requirement (10% of demand)
def downward_reserve_req(model):
    return sum(model.rD[g] for g in model.init_conv_G) == 0.10 * data["load"]["System demand (MW)"][hour-1]
model.downward_reserve_constraint = Constraint(rule=downward_reserve_req)

# 3. Generator Capacity Limits (Adjusted for Reserves)
def generator_capacity_limit(model, g):
    return model.rU[g] + model.rD[g] <= model.Pmax_convG[g] 
model.generator_capacity_constraint = Constraint(model.init_conv_G, rule=generator_capacity_limit)

# 4. Ramping Constraints
def max_ru(model, i):
    return model.rU[i] <= model.RUmax[i]
model.max_ru = Constraint(model.init_conv_G, rule=max_ru)

def max_rd(model, i):
    return model.rD[i] <= model.RDmax[i]
model.max_rd = Constraint(model.init_conv_G, rule=max_rd)

#Dual variables
model.dual = Suffix(direction=Suffix.IMPORT)

# Create a solver 
solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

# Solve the model
solution = solver.solve(model, tee=True)

model.display()

# # Store the results
results_reserve = {}
results_reserve["up"] = [round(value(model.rU[key]),3) for key in model.rU]
results_reserve["down"] = [round(value(model.rD[key]),3) for key in model.rD]
# results["wind farms"] = [round(value(model.p_wind_farm[key]),3) for key in model.p_wind_farm]
print(results_reserve)
# print("Social Welfare =", round(model.social_welfare(), 3))

#print(solution)

# Extract the market-clearing price (MCP)
if solution.solver.termination_condition == TerminationCondition.optimal:
    up_price = model.dual[model.upward_reserve_constraint]
    print(f"Up reserve price: {up_price:.2f}")
    down_price = model.dual[model.downward_reserve_constraint]
    print(f"Down reserve price: {down_price:.2f}")
else:
    print("Solver did not find an optimal solution. MCP cannot be calculated.")

######### DAY AHEAD MARKET CLEARING OPTIMIZATION
#create a new model
modelDA = ConcreteModel()

# Sets
modelDA.init_conv_G = Set(initialize=[e+1 for e in range(12)])
modelDA.init_wind_farm = Set(initialize=[e+1 for e in range(6)])
modelDA.init_demand=Set(initialize=[e+1 for e in range(len(data["node_demand"]["Load #"]))])


#Declare variables
modelDA.p_demand= Var(modelDA.init_demand, within=NonNegativeReals, initialize=Pmax_D/2)
modelDA.p_conv_G = Var(modelDA.init_conv_G, within=NonNegativeReals, initialize=0)
modelDA.p_wind_farm = Var(modelDA.init_wind_farm, within=NonNegativeReals, initialize=0)

#Parameters
modelDA.Pmax_convG=Param(modelDA.init_conv_G, initialize={i+1: pmax for i, pmax in enumerate(data["generation_unit"]["Pmax (MW)"])})
modelDA.price_conv=Param(modelDA.init_conv_G, initialize={i+1: price for i, price in enumerate(data["generation_unit"]["Ci"])})

modelDA.price_wind_farm=Param(modelDA.init_wind_farm, initialize = 0)

modelDA.Pmax_demand=Param(modelDA.init_demand, initialize={i+1: data["load"]["System demand (MW)"][hour-1]*percentage/100 for i, percentage in enumerate(data["node_demand"]["percentage of system load"])} )
modelDA.demand_bidding_prices=Param(modelDA.init_demand, initialize={i+1: price for i, price in enumerate(data["node_demand"]["Bid price"])})


# Objective function : Maximization of Social welfare
def objective_rule(modelDA):
    return (sum(modelDA.demand_bidding_prices[k]*modelDA.p_demand[k] for k in modelDA.init_demand)
            - sum(modelDA.price_conv[i] * modelDA.p_conv_G[i] for i in modelDA.init_conv_G) 
            - sum(modelDA.price_wind_farm[j] * modelDA.p_wind_farm[j] for j in modelDA.init_wind_farm))
modelDA.social_welfare = Objective(rule=objective_rule, sense=maximize)

## Constraints

# Constraints modified due to the reserve markets

def max_capacity_rule_conv_G(modelDA, i):
    return modelDA.p_conv_G[i] <= modelDA.Pmax_convG[i]-results_reserve["up"][i-1]
modelDA.max_capacity_conv_G_constraint = Constraint(modelDA.init_conv_G, rule=max_capacity_rule_conv_G)

def min_capacity_rule_conv_G(modelDA, i):
    return  results_reserve["down"][i-1] <= modelDA.p_conv_G[i]
modelDA.min_capacity_conv_G_constraint = Constraint(modelDA.init_conv_G, rule=min_capacity_rule_conv_G)

#Wind capacity
def capacity_rule_wind_farm(modelDA, i):
    return modelDA.p_wind_farm[i] <= data["wind_farm"][f"wind_farm {i}"][hour-1]*wind_farm_capacity
modelDA.capacity_wind_farm_constraint = Constraint(modelDA.init_wind_farm, rule=capacity_rule_wind_farm)

#Demand
def max_load_demand(modelDA, j):
    return modelDA.p_demand[j] <= modelDA.Pmax_demand[j]
modelDA.max_load_demand = Constraint(modelDA.init_demand, rule=max_load_demand)

#Balance
def balance(modelDA):
    return sum(modelDA.p_demand[k] for k in modelDA.init_demand) == sum(modelDA.p_conv_G[i] for i in modelDA.init_conv_G) + sum(modelDA.p_wind_farm[j] for j in modelDA.init_wind_farm)
modelDA.balance_constraint = Constraint(rule=balance)


#Dual variables
modelDA.dual = Suffix(direction=Suffix.IMPORT)

# Create a solver 
solverDA = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

# Solve the model
solutionDA = solverDA.solve(modelDA, tee=True)

modelDA.display()

# Store the results
results_DA = {}
results_DA["load"] = [round(value(modelDA.p_demand[key]),3) for key in modelDA.p_demand]
results_DA["conventional generators"] = [round(value(modelDA.p_conv_G[key]),3) for key in modelDA.p_conv_G]
results_DA["wind farms"] = [round(value(modelDA.p_wind_farm[key]),3) for key in modelDA.p_wind_farm]
print(results_DA)
print("Social Welfare =", round(modelDA.social_welfare(), 3))


# Extract the market-clearing price (MCP)
if solutionDA.solver.termination_condition == TerminationCondition.optimal:
    mcp = modelDA.dual[modelDA.balance_constraint]
    print(f"Market-Clearing Price (MCP): {mcp:.2f}")
else:
    print("Solver did not find an optimal solution. MCP cannot be calculated.")

print(f"Down reserve price: {down_price:.2f}")
print(f"Up reserve price: {up_price:.2f}")