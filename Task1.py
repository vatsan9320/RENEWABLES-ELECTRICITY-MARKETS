from pyomo.environ import *
import all_data
import numpy as np
import matplotlib.pyplot as plt

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
model.p_demand= Var(model.init_demand, within=NonNegativeReals, initialize=Pmax_D/2)
model.p_conv_G = Var(model.init_conv_G, within=NonNegativeReals, initialize=0)
model.p_wind_farm = Var(model.init_wind_farm, within=NonNegativeReals, initialize=0)

#Parameters
model.Pmax_convG=Param(model.init_conv_G, initialize={i+1: pmax for i, pmax in enumerate(data["generation_unit"]["Pmax (MW)"])})
model.price_conv=Param(model.init_conv_G, initialize={i+1: price for i, price in enumerate(data["generation_unit"]["Ci"])})

model.price_wind_farm=Param(model.init_wind_farm, initialize = 0)

model.Pmax_demand=Param(model.init_demand, initialize={i+1: price for i, price in enumerate(data["node_demand"]["Load distribution peak"])} )
model.demand_bidding_prices=Param(model.init_demand, initialize={i+1: price for i, price in enumerate(data["node_demand"]["Bid price"])})


# Objective function : Maximization of Social welfare
def objective_rule(model):
    return (sum(model.demand_bidding_prices[k]*model.p_demand[k] for k in model.init_demand)
            - sum(model.price_conv[i] * model.p_conv_G[i] for i in model.init_conv_G) 
            - sum(model.price_wind_farm[j] * model.p_wind_farm[j] for j in model.init_wind_farm))
model.social_welfare = Objective(rule=objective_rule, sense=maximize)

# Constraints
def capacity_rule_conv_G(model, i):
    return model.p_conv_G[i] <= model.Pmax_convG[i]
model.capacity_conv_G_constraint = Constraint(model.init_conv_G, rule=capacity_rule_conv_G)

def capacity_rule_wind_farm(model, i):
    return model.p_wind_farm[i] <= np.mean(data["wind_farm"][f"wind_farm {i}"])*wind_farm_capacity
model.capacity_wind_farm_constraint = Constraint(model.init_wind_farm, rule=capacity_rule_wind_farm)

def max_load_demand(model, j):
    return model.p_demand[j] <= model.Pmax_demand[j]
model.max_load_demand = Constraint(model.init_demand, rule=max_load_demand)

def balance(model):
    return sum(model.p_demand[k] for k in model.init_demand) == sum(model.p_conv_G[i] for i in model.init_conv_G) + sum(model.p_wind_farm[j] for j in model.init_wind_farm)
model.balance_constraint = Constraint(rule=balance)

#Dual variables
model.dual = Suffix(direction=Suffix.IMPORT)

# Create a solver 
solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

# Solve the model
solution = solver.solve(model, tee=True)

model.display()

# Store the results
results = {}
results["load"] = [round(value(model.p_demand[key]),3) for key in model.p_demand]
results["conventional generators"] = [round(value(model.p_conv_G[key]),3) for key in model.p_conv_G]
results["wind farms"] = [round(value(model.p_wind_farm[key]),3) for key in model.p_wind_farm]
print(results)
print("Social Welfare =", round(model.social_welfare(), 3))

#print(solution)

# Extract the market-clearing price (MCP)
if solution.solver.termination_condition == TerminationCondition.optimal:
    mcp = model.dual[model.balance_constraint]
    print(f"Market-Clearing Price (MCP): {mcp:.2f}")
else:
    print("Solver did not find an optimal solution. MCP cannot be calculated.")

#Profit of each generators
profit= {}
profit["conventional generators"] = [
    round(prod * (mcp - price), 3)
    for prod, price in zip(results["conventional generators"], data["generation_unit"]["Ci"])]
profit["wind farms"] = [round(value * mcp, 3) for value in results["wind farms"]]

#Utility
utility= [round(qtt*(bidprice-mcp), 3) for bidprice, qtt in zip(data["node_demand"]["Bid price"], results["load"])]

print("profit", profit, "utility", utility)


####### Print Supply and Offer curves

# Data

demand_quantities=[2650.5*x/100 for x in [3.8, 3.4, 6.3, 2.6, 2.5, 4.8, 4.4, 6.0, 6.1, 6.8, 9.3, 6.8, 11.1, 3.5, 11.7, 6.4, 4.5, 0]]
demand_bidding_prices=[25, 22, 15.4, 12.5, 13, 14, 24, 15, 12.8, 17.8, 29.3, 28, 16.9, 30, 18, 16, 21, 0]

supply_quantities = [152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 300, 350, 
                     2*66.02668138964893, 2*69.90705836674121, 2*69.09000109053993, 
                     2*60.93791955818926, 2*64.88591062635679, 2*62.98944854324062]
generators_bidding_prices = [13.32, 13.32, 20.7, 20.93, 26.11, 10.52, 6.02, 5.47, 0, 0, 
                  10.52, 10.89, 0, 0, 0, 0, 0, 0]

# Plot the supply curve
sorted_indices = np.argsort(generators_bidding_prices) #Sort in ascending order
sorted_prices = np.array(generators_bidding_prices)[sorted_indices]
sorted_supply= np.array(supply_quantities)[sorted_indices]
sorted_cumulative_supply = np.cumsum(sorted_supply) 

plt.figure(figsize=(10, 6))
plt.step(sorted_cumulative_supply, sorted_prices, where='post', label="Supply", color='orange')

#Plot the demand curve
sorted_indices_dem = np.argsort(demand_bidding_prices)[::-1]
sorted_prices_dem = np.array(demand_bidding_prices)[sorted_indices_dem]
sorted_demand= np.array(demand_quantities)[sorted_indices_dem]
sorted_cumulative_demand = np.cumsum(sorted_demand) 
plt.step(sorted_cumulative_demand, sorted_prices_dem, where='post', label="Demand", color='blue')

# Add legends
plt.ylabel("Price")
plt.xlabel("Quantity (MW)")
plt.title("Supply and Demand")
plt.legend()
plt.grid(True)

plt.show()