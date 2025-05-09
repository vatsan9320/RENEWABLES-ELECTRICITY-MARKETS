from pyomo.environ import *
import all_data
import numpy as np
import matplotlib.pyplot as plt
import random

data=all_data.get_data()
Pmax_D=np.max(data["load"]["System demand (MW)"])
wind_farm_capacity=200

# Generating random demand bid price and loading demand quantities data
peak_demand = max(data["load"]["System demand (MW)"])
peak_hours = [hour for hour, demand in zip(data["load"]["Hour"], data["load"]["System demand (MW)"]) if demand >= 0.95 * peak_demand]

total_bid_price = {}
total_load_distribution={}
for hour in data["load"]["Hour"]:
    is_peak_hour = hour in peak_hours
    hour_prices = []
    
    for bid_peak in data["node_demand"]["Bid price"]:
        if is_peak_hour:
            # Demand peak hours bid prices
            hour_prices.append(bid_peak)
        else:
            # Random bid prices for non peak hours
            random_price = round(random.uniform(11, bid_peak - 1), 2)
            hour_prices.append(random_price)
    total_bid_price[f"hour {hour}"] = hour_prices
    
    hour_load =[]
    for i in range(len(data["node_demand"]["Load #"])):
        hour_load.append(round(data["node_demand"]["percentage of system load"][i]/100*data["load"]["System demand (MW)"][hour-1], 2))

    total_load_distribution[f"hour {hour}"] = hour_load


#create a model
model = ConcreteModel()

# Sets

model.time=RangeSet(24)
model.init_conv_G = Set(initialize=[e+1 for e in range(12)])
model.init_wind_farm = Set(initialize=[e+1 for e in range(6)])
model.init_demand=Set(initialize=[e+1 for e in range(len(data["node_demand"]["Load #"]))])


#Declare variables
model.p_demand= Var(model.init_demand, model.time, within=NonNegativeReals, initialize=Pmax_D/2)
model.p_conv_G = Var(model.init_conv_G, model.time, within=NonNegativeReals, initialize=0)
model.p_wind_farm = Var(model.init_wind_farm, model.time, within=NonNegativeReals, initialize=0)

model.p_charging = Var(model.time, within=NonNegativeReals, initialize=0)
model.p_discharging = Var(model.time, within=NonNegativeReals, initialize=0)
model.energy_stored = Var(model.time, within=NonNegativeReals ,initialize=0)

#Parameters

model.Pmax_convG=Param(model.init_conv_G, model.time, initialize={(i+1, t): pmax for i, pmax in enumerate(data["generation_unit"]["Pmax (MW)"]) for t in model.time})
model.price_conv=Param(model.init_conv_G, model.time, initialize={(i+1, t): price for i, price in enumerate(data["generation_unit"]["Ci"]) for t in model.time})


model.price_wind_farm=Param(model.init_wind_farm, model.time, initialize = 0)

model.Pmax_demand=Param(model.init_demand, model.time,  
                        initialize={(i+1, t) : qtt for t in model.time for i, qtt in enumerate(total_load_distribution[f"hour {t}"])} )
model.demand_bidding_prices=Param(model.init_demand, model.time, 
                                  initialize={(i+1, t): price for t in model.time for i, price in enumerate(total_bid_price[f"hour {t}"])})

model.Pmax_charging=Param(model.time, initialize= data["battery"]["Charging capacity (MW)"])
model.Pmax_discharging=Param(model.time, initialize= data["battery"]["Discharging capacity (MW)"])
model.battery_capacity=Param(model.time, initialize= data["battery"]["Energy storage capacity (MWh)"])

# Objective function : Maximization of Social welfare (is not alter by the storage unit)
def objective_rule(model):
    return sum((sum(model.demand_bidding_prices[k,t]*model.p_demand[k,t] for k in model.init_demand)
            - sum(model.price_conv[i,t] * model.p_conv_G[i,t] for i in model.init_conv_G) 
            - sum(model.price_wind_farm[j,t] * model.p_wind_farm[j,t] for j in model.init_wind_farm))
            for t in model.time)
model.social_welfare = Objective(rule=objective_rule, sense=maximize)

### Constraints

#Generators
def capacity_rule_conv_G(model, i, t):
    return model.p_conv_G[i,t] <= model.Pmax_convG[i,t]
model.capacity_conv_G_constraint = Constraint(model.init_conv_G, model.time,  rule=capacity_rule_conv_G)

def capacity_rule_wind_farm(model, i, t):
    return model.p_wind_farm[i,t] <= data["wind_farm"][f"wind_farm {i}"][t-1]*wind_farm_capacity
model.capacity_wind_farm_constraint = Constraint(model.init_wind_farm, model.time, rule=capacity_rule_wind_farm)

#Demands
def max_load_demand(model, j, t):
    return model.p_demand[j,t] <= model.Pmax_demand[j,t]
model.max_load_demand = Constraint(model.init_demand, model.time, rule=max_load_demand)

#Battery storage

def capacity_charging(model, t):
    return model.p_charging[t] <= model.Pmax_charging[t]
model.capacity_charging_constraint = Constraint(model.time, rule = capacity_charging)

def capacity_discharging(model, t):
    if t==1:
        return model.p_discharging[t]==0
    else:
        return model.p_discharging[t] <= model.Pmax_discharging[t]
model.capacity_discharging_constraint = Constraint(model.time, rule = capacity_discharging)

def capacity_energy_storage(model, t):
    return model.energy_stored[t] <= model.battery_capacity[t]
model.capacity_energy_constraint = Constraint(model.time, rule = capacity_energy_storage)

def storage_operation(model, t):
    if t>1:
        return model.energy_stored[t] == model.energy_stored[t-1]+model.p_charging[t]*data["battery"]["Charging efficiency"]-model.p_discharging[t]/data["battery"]["Discharging efficiency"]
    elif t==1:
        return model.energy_stored[t]== 0
model.storage_constraint = Constraint(model.time, rule = storage_operation)

#Balance
def balance(model, t):
    return (sum(model.p_demand[k,t] for k in model.init_demand) + model.p_charging[t]
            == sum(model.p_conv_G[i,t] for i in model.init_conv_G) 
            + sum(model.p_wind_farm[j,t] for j in model.init_wind_farm) + model.p_discharging[t])
model.balance_constraint = Constraint(model.time, rule=balance)



#Dual variables
model.dual = Suffix(direction=Suffix.IMPORT)

# Create a solver 
solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

# Solve the model
solution = solver.solve(model, tee=True)

model.display()

# Store the results
results = {"load":{}, "conventional generators":{} ,"wind farms":{}}

for i in model.init_demand : 
    results["load"][f"Load {i}"] = [round(value(model.p_demand[i, t]), 3) for t in model.time]
for i in model.init_conv_G :
    results["conventional generators"][f"Gen {i}"] = [round(value(model.p_conv_G[i, t]),3) for t in model.time]
for i in model.init_wind_farm :
    results["wind farms"][f"WF {i}"] = [round(value(model.p_wind_farm[i, t]),3) for t in model.time]

results["charging power"] = [round(value(model.p_charging[t]),3) for t in model.time]
results["discharging power"] = [round(value(model.p_discharging[t]),3) for t in model.time]
results["energy storage"] = [round(value(model.energy_stored[t]),3) for t in model.time]

print(results)
print("Social Welfare =", round(model.social_welfare(), 3))

# #print(solution)

# Market Clearing Prices
if solution.solver.termination_condition == TerminationCondition.optimal:
    mcp = []
    for t in model.time:
        # Access the dual for each balance_constraint[t]
        mcp.append(value(model.dual[model.balance_constraint[t]]))
    
    # Display the MCP for each hour
    for t, price in enumerate(mcp, start=1):
        print(f"Hour {t}: MCP = {price:.2f}")
else:
    print("Solver did not find an optimal solution. MCP cannot be calculated.")



#Profit of each generator over 24h
profit= {"conventional generators" : {}, "wind farms" : {}}
for i in model.init_conv_G:
    profit["conventional generators"][f"Gen {i}"] = [round(sum((results["conventional generators"][f"Gen {i}"][t-1] * (mcp[t-1] - data["generation_unit"]["Ci"][i-1])) for t in model.time),3)]

for i in model.init_wind_farm:
    profit["wind farms"][f"WF {i}"] = [round(sum((results["wind farms"][f"WF {i}"][t-1] * mcp[t-1]) for t in model.time),3)]

#Profit battery over 24h (sells at MCP)
total_profit_battery=sum((results["discharging power"][t-1] - results["charging power"][t-1])*mcp[t-1] for t in model.time)

#Utility of each demand over 24h
utility={}
for i in model.init_demand :
    utility[f"Load {i}"]=[round(sum(results["load"][f"Load {i}"][t-1]*(total_bid_price[f"hour {t}"][i-1]-mcp[t-1]) for t in model.time),3)]


print("profit", profit, "utility", utility)

# Total_Profit=sum(profit["conventional generators"][f"Gen {i}"][0] for i in model.init_conv_G)+sum(profit["wind farms"][f"WF {i}"][0] for i in model.init_wind_farm)
# print("Total Profit", Total_Profit)
# Total_Utility=sum(utility[f"Load {i}"][0] for i in model.init_demand)
# print("Total Utility", Total_Utility )
print("Total Battery Profit", total_profit_battery)
print("Total discharging", sum((results["discharging power"][t-1] for t in model.time)))
# print("Total charging", sum((results["charging power"][t-1] for t in model.time)))
# print("SW",Total_Utility+Total_Profit)
# print("Social Welfare =", round(model.social_welfare(), 3))


# #### Plot graphs

# Market Clearing Prices
plt.figure()
plt.step(model.time, mcp)
plt.ylabel("Market clearing price")
plt.xlabel("Time")
plt.title("Market Clearing Prices")
plt.grid(True)



#Battery 

plt.figure()
plt.subplot(3,1,1)
plt.step(model.time, results["charging power"], label="Charging power")
plt.legend()

plt.subplot(3,1,2)
plt.step(model.time, results["discharging power"], label="Disharging power")
plt.legend()

plt.subplot(3,1,3)
plt.step(model.time, results["energy storage"], label="Energy stored")
plt.legend()


plt.grid(True)

plt.show()


