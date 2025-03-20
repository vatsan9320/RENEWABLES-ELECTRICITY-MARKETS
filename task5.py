from pyomo.environ import *
import all_data
import numpy as np

# Load data
data = all_data.get_data()

# Extract system parameters
Pmax_D = np.max(data["load"]["System demand (MW)"])
wind_farm_capacity = 200

# Introduce Imbalance Factors
wind_forecast_error = {1: -0.15, 2: 0.10, 3: -0.15, 4: 0.10, 5: -0.15, 6: 0.10}  # Some wind farms under-produce (-15%), others over-produce (+10%)
generator_outage = 4  # Unexpected outage at generator 4

# Create a Pyomo model
model = ConcreteModel()

# Sets
model.init_conv_G = Set(initialize=[e + 1 for e in range(12)])
model.init_wind_farm = Set(initialize=[e + 1 for e in range(6)])
model.init_demand = Set(initialize=[e + 1 for e in range(len(data["node_demand"]["Load #"]))])

# Variables
model.p_demand = Var(model.init_demand, within=NonNegativeReals, initialize=Pmax_D / 2)
model.p_conv_G = Var(model.init_conv_G, within=NonNegativeReals, initialize=0)
model.p_wind_farm = Var(model.init_wind_farm, within=NonNegativeReals, initialize=0)

# Parameters
model.Pmax_convG = Param(model.init_conv_G, initialize={i + 1: pmax for i, pmax in enumerate(data["generation_unit"]["Pmax (MW)"])})
model.price_conv = Param(model.init_conv_G, initialize={i + 1: price for i, price in enumerate(data["generation_unit"]["Ci"])})
model.Pmax_demand = Param(model.init_demand, initialize={i + 1: price for i, price in enumerate(data["node_demand"]["Load distribution peak"])})
model.demand_bidding_prices = Param(model.init_demand, initialize={i + 1: price for i, price in enumerate(data["node_demand"]["Bid price"])})

# Objective: Maximize Social Welfare
def objective_rule(model):
    return (sum(model.demand_bidding_prices[k] * model.p_demand[k] for k in model.init_demand)
            - sum(model.price_conv[i] * model.p_conv_G[i] for i in model.init_conv_G)
            - sum(0 * model.p_wind_farm[j] for j in model.init_wind_farm))  # Wind farms have zero cost
model.social_welfare = Objective(rule=objective_rule, sense=maximize)

# Constraints
def capacity_rule_conv_G(model, i):
    if i == generator_outage:  # Generator outage condition
        return model.p_conv_G[i] == 0
    return model.p_conv_G[i] <= model.Pmax_convG[i]
model.capacity_conv_G_constraint = Constraint(model.init_conv_G, rule=capacity_rule_conv_G)

def capacity_rule_wind_farm(model, i):
    wind_output = np.mean(data["wind_farm"][f"wind_farm {i}"]) * wind_farm_capacity
    wind_output *= (1 + wind_forecast_error[i])  # Apply forecast error
    return model.p_wind_farm[i] <= wind_output
model.capacity_wind_farm_constraint = Constraint(model.init_wind_farm, rule=capacity_rule_wind_farm)

def max_load_demand(model, j):
    return model.p_demand[j] <= model.Pmax_demand[j]
model.max_load_demand = Constraint(model.init_demand, rule=max_load_demand)

def balance(model):
    return sum(model.p_demand[k] for k in model.init_demand) == sum(model.p_conv_G[i] for i in model.init_conv_G) + sum(model.p_wind_farm[j] for j in model.init_wind_farm)
model.balance_constraint = Constraint(rule=balance)

# Dual variables for MCP
model.dual = Suffix(direction=Suffix.IMPORT)

# Solve with Gurobi
solver = SolverFactory("gurobi", solver_io="python")
solution = solver.solve(model, tee=True)

# Store results
results = {
    "load": [round(value(model.p_demand[key]), 3) for key in model.p_demand],
    "conventional generators": [round(value(model.p_conv_G[key]), 3) for key in model.p_conv_G],
    "wind farms": [round(value(model.p_wind_farm[key]), 3) for key in model.p_wind_farm]
}
print(results)
print("Social Welfare =", round(model.social_welfare(), 3))

# Extract Market-Clearing Price
if solution.solver.termination_condition == TerminationCondition.optimal:
    mcp = model.dual[model.balance_constraint]
    print(f"Market-Clearing Price (MCP): {mcp:.2f}")
else:
    print("Solver did not find an optimal solution. MCP cannot be calculated.")

# Balancing Market Price
balancing_price = mcp + 0.1 * np.mean(data["generation_unit"]["Ci"])  # Upward regulation price
print(f"Balancing Market Price: {balancing_price:.2f}")

# Profits Calculation
profit = {}
profit["conventional generators"] = [
    round(prod * (mcp - price), 3)
    for prod, price in zip(results["conventional generators"], data["generation_unit"]["Ci"])
]
profit["wind farms"] = [round(value * mcp, 3) for value in results["wind farms"]]

# Balancing Market Profits (Two-Pricing Scheme)
profit_balancing = {}
profit_balancing["conventional generators"] = [
    round(prod * (balancing_price - price), 3)
    for prod, price in zip(results["conventional generators"], data["generation_unit"]["Ci"])
]
profit_balancing["wind farms"] = [round(value * balancing_price, 3) for value in results["wind farms"]]

# Display Profits
print("Day-Ahead Market Profits:", profit)
print("Balancing Market Profits (Two-Pricing Scheme):", profit_balancing)

# One-Price vs Two-Price Scheme
one_price_scheme = {k: v for k, v in profit_balancing.items()}
two_price_scheme = {k: v for k, v in profit.items()}

# Compare Profitability
print("\nComparison of Imbalance Settlement Schemes:")
print("One-Price Scheme:", one_price_scheme)
print("Two-Price Scheme:", two_price_scheme)

# Conclusion
if sum(one_price_scheme["conventional generators"]) > sum(two_price_scheme["conventional generators"]):
    print("One-price scheme benefits balancing service providers more.")
else:
    print("Two-price scheme provides a more stable settlement.")
