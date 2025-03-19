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
model.init_nodes=RangeSet(24)  #The system has 24 nodes/buses
model.init_transmission_lines=Set(initialize=range(len(data["transmission_line"]["From"])))

#Declare variables
model.p_demand= Var(model.init_demand, within=NonNegativeReals, initialize=0)
model.p_conv_G = Var(model.init_conv_G, within=NonNegativeReals, initialize=0)
model.p_wind_farm = Var(model.init_wind_farm, within=NonNegativeReals, initialize=0)
model.theta = Var(model.init_nodes, bounds= [-np.pi, np.pi], initialize=0) 

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

### Constraints
def capacity_rule_conv_G(model, i):
    return model.p_conv_G[i] <= model.Pmax_convG[i]
model.capacity_conv_G_constraint = Constraint(model.init_conv_G, rule=capacity_rule_conv_G)

def capacity_rule_wind_farm(model, i):
    return model.p_wind_farm[i] <= np.mean(data["wind_farm"][f"wind_farm {i}"])*wind_farm_capacity
model.capacity_wind_farm_constraint = Constraint(model.init_wind_farm, rule=capacity_rule_wind_farm)

def max_load_demand(model, j):
    return model.p_demand[j] <= model.Pmax_demand[j]
model.max_load_demand = Constraint(model.init_demand, rule=max_load_demand)

#Power balance at each node : sum demand + power flow = sum generation at each node
def reference(model):
    return model.theta[1]==0
model.reference_constraint = Constraint(rule=reference)


def balance(model, n):
    p_demand_node_n=sum(model.p_demand[k] for k in model.init_demand if data["node_demand"]["Node"][k-1]==n)
    p_conv_G_node_n=sum(model.p_conv_G[i] for i in model.init_conv_G if data["generation_unit"]["Node"][i-1]==n)
    p_wind_farm_node_n= sum(model.p_wind_farm[j] for j in model.init_wind_farm if data["wind_farm"]["Node"][j-1]==n)
    power_flow_from_node_n= sum(data["transmission_line"]["Susceptance"][0]*(model.theta[n]-model.theta[m]) for ind,m in enumerate(data["transmission_line"]["To"]) if data["transmission_line"]["From"][ind]==n)
    power_flow_to_node_n=  sum(data["transmission_line"]["Susceptance"][0]*(model.theta[n]-model.theta[m]) for ind,m in enumerate(data["transmission_line"]["From"]) if data["transmission_line"]["To"][ind]==n)
    
    return (p_demand_node_n + power_flow_from_node_n + power_flow_to_node_n - p_conv_G_node_n - p_wind_farm_node_n == 0)
   
model.balance_constraint = Constraint(model.init_nodes, rule=balance)

#Constraint on the  Transmission limits of the line connecting

def max_transmission_limit(model, l):
    from_node = data["transmission_line"]["From"][l]
    to_node = data["transmission_line"]["To"][l]
    susceptance = data["transmission_line"]["Susceptance"][l]
    capacity = data["transmission_line"]["Capacity (MVA)"][l]
    return susceptance * (model.theta[from_node] - model.theta[to_node]) <= capacity
model.max_transmission_constraint = Constraint(model.init_transmission_lines, rule=max_transmission_limit)

def min_transmission_limit(model, l):
   
    from_node = data["transmission_line"]["From"][l]
    to_node = data["transmission_line"]["To"][l]
    susceptance = data["transmission_line"]["Susceptance"][l]
    capacity = data["transmission_line"]["Capacity (MVA)"][l]
    return susceptance * (model.theta[from_node] - model.theta[to_node]) >= - capacity
model.min_transmission_constraint = Constraint(model.init_transmission_lines, rule=min_transmission_limit)


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
results["theta"]=[round(value(model.theta[key]),3) for key in model.theta]
print(results)
print("Social Welfare =", round(model.social_welfare(), 3))

#print(solution)

for i in model.init_nodes:
    print(f"Constraint for node {i}: {model.balance_constraint[i].expr}")

# Extract the market-clearing price (MCP)
if solution.solver.termination_condition == TerminationCondition.optimal:
    local_node_mcp = []
    for n in model.init_nodes:
        local_node_mcp.append(value(model.dual[model.balance_constraint[n]]))
    
    # Display the MCP for each hour
    for n, price in enumerate(local_node_mcp, start=1):
        print(f"Node {n}: MCP = {price:.2f}")
else:
    print("Solver did not find an optimal solution. MCP cannot be calculated.")

# for l in model.init_transmission_lines:
#     print(f"Line {l}: Transmission flow = {data['transmission_line']['Susceptance'][l] * (model.theta[data['transmission_line']['From'][l]].value - model.theta[data['transmission_line']['To'][l]].value)}")


print("Social Welfare =", round(model.social_welfare(), 3))
# #Profit of each generators
# profit= {}
# profit["conventional generators"] = [
#     round(prod * (mcp - price), 3)
#     for prod, price in zip(results["conventional generators"], data["generation_unit"]["Ci"])]
# profit["wind farms"] = [round(value * mcp, 3) for value in results["wind farms"]]

# #Utility
# utility= [round(qtt*(bidprice-mcp), 3) for bidprice, qtt in zip(data["node_demand"]["Bid price"], results["load"])]

# print("profit", profit, "utility", utility)

