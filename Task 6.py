from pyomo.environ import *
import all_data
import numpy as np

# Load data
data = all_data.get_data()
Pmax_D = np.max(data["load"]["System demand (MW)"])

# List of conventional generators excluding generator 8 (outage)
active_generators = [g for g in data["generation_unit"]["Unit #"] if g != 8]

# Create a Pyomo model
model = ConcreteModel()

# Sets
model.generators = Set(initialize=active_generators)
model.hours = Set(initialize=range(1, 25))  # 24-hour market

# Decision Variables
model.rU = Var(model.generators, model.hours, within=NonNegativeReals)  # Upward reserve
model.rD = Var(model.generators, model.hours, within=NonNegativeReals)  # Downward reserve
model.pG = Var(model.generators, model.hours, within=NonNegativeReals)  # Generation output

# Parameters
model.Pmax = Param(model.generators, initialize={g: data["generation_unit"]["Pmax (MW)"][g-1] for g in active_generators})
model.RU = Param(model.generators, initialize={g: data["generation_unit"]["RU (MW/h)"][g-1] for g in active_generators})
model.RD = Param(model.generators, initialize={g: data["generation_unit"]["RD (MW/h)"][g-1] for g in active_generators})
model.reserve_cost = Param(model.generators, initialize={g: data["generation_unit"]["Cu_i"][g-1] for g in active_generators})
model.demand = Param(model.hours, initialize={t: data["load"]["System demand (MW)"][t-1] for t in range(1, 25)})
model.gen_cost = Param(model.generators, initialize={g: data["generation_unit"]["Ci"][g-1] for g in active_generators})

# Objective Function: Minimize total cost (reserve + energy market)
def objective_rule(model):
    return sum(model.reserve_cost[g] * (model.rU[g, t] + model.rD[g, t]) + model.gen_cost[g] * model.pG[g, t] 
               for g in model.generators for t in model.hours)
model.obj = Objective(rule=objective_rule, sense=minimize)

# Constraints
# 1. Upward Reserve Procurement Requirement (15% of demand)
def upward_reserve_req(model, t):
    return sum(model.rU[g, t] for g in model.generators) >= 0.15 * model.demand[t]
model.upward_reserve_constraint = Constraint(model.hours, rule=upward_reserve_req)

# 2. Downward Reserve Procurement Requirement (10% of demand)
def downward_reserve_req(model, t):
    return sum(model.rD[g, t] for g in model.generators) >= 0.10 * model.demand[t]
model.downward_reserve_constraint = Constraint(model.hours, rule=downward_reserve_req)

# 3. Generator Capacity Limits (Adjusted for Reserves)
def generator_capacity_limit(model, g, t):
    return model.pG[g, t] + model.rU[g, t] <= model.Pmax[g] + model.rD[g, t]
model.generator_capacity_constraint = Constraint(model.generators, model.hours, rule=generator_capacity_limit)

# 4. Ramping Constraints
def ramping_limit_up(model, g, t):
    return model.rU[g, t] <= model.RU[g]
model.ramping_constraint_up = Constraint(model.generators, model.hours, rule=ramping_limit_up)

def ramping_limit_down(model, g, t):
    return model.rD[g, t] <= model.RD[g]
model.ramping_constraint_down = Constraint(model.generators, model.hours, rule=ramping_limit_down)

# 5. Demand Satisfaction
def demand_satisfaction(model, t):
    return sum(model.pG[g, t] for g in model.generators) == model.demand[t]
model.demand_satisfaction = Constraint(model.hours, rule=demand_satisfaction)

# Dual variables for price calculations
model.dual = Suffix(direction=Suffix.IMPORT)

# Solve the Model
solver = SolverFactory("gurobi", solver_io="python")  # Ensure Gurobi is installed
solution = solver.solve(model, tee=True)

# Display Results
if solution.solver.termination_condition == TerminationCondition.optimal:
    print(" Optimal Reserve and Day-Ahead Market Clearing Achieved!")

    # Extract reserve prices
    reserve_prices_up = {t: model.dual[model.upward_reserve_constraint[t]] for t in model.hours}
    reserve_prices_down = {t: model.dual[model.downward_reserve_constraint[t]] for t in model.hours}

    # Extract electricity prices
    electricity_prices = {t: model.dual[model.demand_satisfaction[t]] for t in model.hours}

    # Print reserve market prices
    print("Upward Reserve Prices:", {t: f"{price:.2f}" for t, price in reserve_prices_up.items()})
    print("Downward Reserve Prices:", {t: f"{price:.2f}" for t, price in reserve_prices_down.items()})
    print("Electricity Market Prices:", {t: f"{price:.2f}" for t, price in electricity_prices.items()})
else:
    print("Solver did not find an optimal solution.")
