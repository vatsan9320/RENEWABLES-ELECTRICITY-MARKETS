from pyomo.environ import *
import all_data
import numpy as np

# Load data
data = all_data.get_data()
Pmax_D = np.max(data["load"]["System demand (MW)"])

# Select the hour for optimization
hour = 2  # Change this to optimize for a different hour

# List of generators (all running)
active_generators = [g for g in data["generation_unit"]["Unit #"]]

# Create a Pyomo model
model = ConcreteModel()

# Sets
model.generators = Set(initialize=active_generators)
model.hour = Set(initialize=[hour])  # Single-hour market

# Decision Variables
model.rU = Var(model.generators, model.hour, within=NonNegativeReals)  # Upward reserve
model.rD = Var(model.generators, model.hour, within=NonNegativeReals)  # Downward reserve
model.pG = Var(model.generators, model.hour, within=NonNegativeReals)  # Generation output

# Parameters
model.Pmax = Param(model.generators, initialize={g: data["generation_unit"]["Pmax (MW)"][g-1] for g in active_generators})
model.RU = Param(model.generators, initialize={g: data["generation_unit"]["RU (MW/h)"][g-1] for g in active_generators})
model.RD = Param(model.generators, initialize={g: data["generation_unit"]["RD (MW/h)"][g-1] for g in active_generators})
model.reserve_cost = Param(model.generators, initialize={g: data["generation_unit"]["Cu_i"][g-1] for g in active_generators})
model.demand = Param(model.hour, initialize={hour: data["load"]["System demand (MW)"][hour-1]})  
model.gen_cost = Param(model.generators, initialize={g: data["generation_unit"]["Ci"][g-1] for g in active_generators})

# Objective Function: Minimize total cost (reserve + energy market)
def objective_rule(model):
    return sum(model.reserve_cost[g] * (model.rU[g, hour] + model.rD[g, hour]) + model.gen_cost[g] * model.pG[g, hour] 
               for g in model.generators)
model.obj = Objective(rule=objective_rule, sense=minimize)

# Constraints
# 1. Upward Reserve Procurement Requirement (15% of demand)
def upward_reserve_req(model):
    return sum(model.rU[g, hour] for g in model.generators) >= 0.15 * model.demand[hour]
model.upward_reserve_constraint = Constraint(rule=upward_reserve_req)

# 2. Downward Reserve Procurement Requirement (10% of demand)
def downward_reserve_req(model):
    return sum(model.rD[g, hour] for g in model.generators) >= 0.10 * model.demand[hour]
model.downward_reserve_constraint = Constraint(rule=downward_reserve_req)

# 3. Generator Capacity Limits (Adjusted for Reserves)
def generator_capacity_limit(model, g):
    return model.pG[g, hour] + model.rU[g, hour] <= model.Pmax[g] + model.rD[g, hour]
model.generator_capacity_constraint = Constraint(model.generators, rule=generator_capacity_limit)

# 4. Ramping Constraints
def ramping_limit_up(model, g):
    return model.rU[g, hour] <= model.RU[g]
model.ramping_constraint_up = Constraint(model.generators, rule=ramping_limit_up)

def ramping_limit_down(model, g):
    return model.rD[g, hour] <= model.RD[g]
model.ramping_constraint_down = Constraint(model.generators, rule=ramping_limit_down)

# 5. Demand Satisfaction
def demand_satisfaction(model):
    return sum(model.pG[g, hour] for g in model.generators) == model.demand[hour]
model.demand_satisfaction = Constraint(rule=demand_satisfaction)

# Dual variables for price calculations
model.dual = Suffix(direction=Suffix.IMPORT)

# Solve the Model
solver = SolverFactory("gurobi", solver_io="python")  # Ensure Gurobi is installed
solution = solver.solve(model, tee=True)

# Display Results
if solution.solver.termination_condition == TerminationCondition.optimal:
    print("Optimal Reserve and Day-Ahead Market Clearing Achieved!")

    # Extract reserve prices
    reserve_price_up = model.dual[model.upward_reserve_constraint] if model.upward_reserve_constraint.active else 0
    reserve_price_down = model.dual[model.downward_reserve_constraint] if model.downward_reserve_constraint.active else 0

    # Extract electricity price
    electricity_price = model.dual[model.demand_satisfaction] if model.demand_satisfaction.active else 0

    # Print results
    print(f"Upward Reserve Price: {reserve_price_up:.2f}")
    print(f"Downward Reserve Price: {reserve_price_down:.2f}")
    print(f"Electricity Market Price: {electricity_price:.2f}")
else:
    print("Solver did not find an optimal solution.")
