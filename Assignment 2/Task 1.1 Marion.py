import Task_1_all_data 
import numpy as np
import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt

from utils import create_all_scenarios
import time

start = time.time()

# ### TASK 1.1
# Offering Strategy Under a One-Price Balancing Scheme: Following Lecture 8, formulate and solve the stochastic offering strategy problem 
# for a one-price balancing scheme using in-sample scenarios. 
# Determine the optimal hourly production quantity offers of the # wind farm in the day-ahead market and calculate the expected profit
# ###


## Data

data=Task_1_all_data.get_data()
nbr_in_sample=200

## Create a model
model = ConcreteModel()

## Define the sets
# model.init_time=RangeSet(24)
model.init_time=RangeSet(24)
model.init_scenarios=RangeSet(nbr_in_sample)

# ## Construction of the In-Sample Scenarios
# in_sample={}
# i=1
# while len(in_sample)<nbr_in_sample:
    
#     scen=(data["wind_scenarios"][f"Sc{np.random.randint(1, len(data['wind_scenarios']))}"], 
#           data["DA_price_scenarios"][f"Sc{np.random.randint(1,len(data['DA_price_scenarios']))}"], 
#           data["power_scenarios"][f"Sc{np.random.randint(1,len(data['power_scenarios']))}"] )
    
#     if scen not in in_sample.values():
#         in_sample[f"Sc{i}"]={}   
#         in_sample[f"Sc{i}"]["wind"]=[scen[0][j]*data["misc"]["WF capacity (MW)"] for j in range(len(scen[0]))]
#         in_sample[f"Sc{i}"]["DA_price"]=scen[1]
#         in_sample[f"Sc{i}"]["power"]=scen[2]
        
#         i+=1

###################################################################################################
#### OTHER WAY TO CREATE THE IN SAMPLE : To always have the same ones
all_scenarios=create_all_scenarios(8)
in_sample=all_scenarios["Fold1"]["In Sample"][0]
###################################################################################################


## Declare the variables

#Indexed by t
model.p_DA = Var(model.init_time,  within=NonNegativeReals, initialize=0)

#Indexed by t and w 
model.power_excess = Var(model.init_time, model.init_scenarios, within=NonNegativeReals, initialize=0)
model.power_deficit = Var(model.init_time, model.init_scenarios, within=NonNegativeReals,  initialize=0)
model.imbalance = Var(model.init_time, model.init_scenarios, within=Reals, initialize=0)

## Declare parameters
model.proba = Param(model.init_scenarios, initialize=1/nbr_in_sample)
model.P_nom = Param(model.init_time, initialize=data["misc"]["WF capacity (MW)"])
model.forecasted_DA_price = Param(model.init_time, model.init_scenarios, 
                                  initialize={(t, w): in_sample[f"Sc{w}"]["DA_price"][t-1] 
                                              for t in model.init_time for w in model.init_scenarios})
model.forecasted_wind_power =  Param(model.init_time, model.init_scenarios, 
                                  initialize={(t, w): in_sample[f"Sc{w}"]["wind"][t-1] 
                                              for t in model.init_time for w in model.init_scenarios})
model.realtime_power =  Param(model.init_time, model.init_scenarios, 
                                  initialize={(t, w): in_sample[f"Sc{w}"]["power"][t-1] 
                                              for t in model.init_time for w in model.init_scenarios})

## Objective function

# Ne fonctionne pas, car power excess non borné ? Pourquoi m'avoir fait introduire imbalance si je ne l'utilise jamais ?
# Si je mets des bounds pour mes power excess et deficit genre (0,500), ça fonctionne. Mais je suis pas sûre du tout que ça ait un sens
# En faite je n'arrive pas à comprendre ce que power excess et power deficit représentent : est ce que les 2 peuvent etre différents de 0 en même temps ?
# Si oui ça induit probablement un problème dans mon obj function puisque, je veux "punir" le power excess quand le system est en excess et je veux le reward 
# quand le system est en déficit (realtime_power=0). Or, actuellement, je punis le power deficit quand syst power deficit et je reward le power excess quand
# syst power excess. Il me manque donc les cas où power syst déficit --> je reward le power excess
#                                                 power syst excess  --> je punis le power  
# def objective_rule(model):
#     return sum(model.proba[w]*(model.forecasted_DA_price[t,w]*model.p_DA[t]
#                                    +data["misc"]["Coeff excess"]*model.forecasted_DA_price[t,w]*model.power_excess[t,w]*model.realtime_power[t,w]
#                                    -data["misc"]["Coeff deficit"]*model.forecasted_DA_price[t,w]*model.power_deficit[t,w]*(1-model.realtime_power[t,w])) 
#                                    for t in model.init_time for w in model.init_scenarios )

# model.expected_profit = Objective(rule=objective_rule, sense= maximize)



## Equivalent à la l'objective funciton avec imbalance : prend en compte les REWARDS et PUNITIONS dans les cas où le system est en excès et déficit

def objective_rule(model):
    return sum(model.proba[w]*(model.forecasted_DA_price[t,w]*model.p_DA[t]
                                   +data["misc"]["Coeff excess"]*model.forecasted_DA_price[t,w]*model.power_excess[t,w]*model.realtime_power[t,w]
                                   -data["misc"]["Coeff excess"]*model.forecasted_DA_price[t,w]*model.power_deficit[t,w]*model.realtime_power[t,w]
                                   +data["misc"]["Coeff deficit"]*model.forecasted_DA_price[t,w]*model.power_excess[t,w]*(1-model.realtime_power[t,w])
                                   -data["misc"]["Coeff deficit"]*model.forecasted_DA_price[t,w]*model.power_deficit[t,w]*(1-model.realtime_power[t,w])) 
                                   for t in model.init_time for w in model.init_scenarios )

model.expected_profit = Objective(rule=objective_rule, sense= maximize)




# ## Version avec imbalance au lieu de power excess et power deficit (et donc un signe + au lieu du - qui est équivalente à celle juste au-dessus...)
# def objective_rule(model):
#     return sum(model.proba[w]*(model.forecasted_DA_price[t,w]*model.p_DA[t]
#                                    +data["misc"]["Coeff excess"]*model.forecasted_DA_price[t,w]*model.imbalance[t,w]*model.realtime_power[t,w]
#                                    +data["misc"]["Coeff deficit"]*model.forecasted_DA_price[t,w]*model.imbalance[t,w]*(1-model.realtime_power[t,w])) 
#                                    for t in model.init_time for w in model.init_scenarios )

# model.expected_profit = Objective(rule=objective_rule, sense= maximize)



## Constraints

def max_wind_farm_capacity(model,t):
    return model.p_DA[t]<=data["misc"]["WF capacity (MW)"]
model.wind_farm_capacity_constraint=Constraint(model.init_time, rule=max_wind_farm_capacity)

def imbalance_production(model,t,w):
    return model.imbalance[t,w]== model.forecasted_wind_power[t,w]-model.p_DA[t]
model.imbalance_production_constraint=Constraint(model.init_time, model.init_scenarios, rule=imbalance_production)

def imbalance_excess_deficit(model,t,w):
    return model.imbalance[t,w]== model.power_excess[t,w]-model.power_deficit[t,w]
model.imbalance_exc_def_constraint=Constraint(model.init_time, model.init_scenarios, rule=imbalance_excess_deficit)

## Solve the problem

#Dual variables
model.dual = Suffix(direction=Suffix.IMPORT)

# Create a solver 
solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

# Solve the model
solution = solver.solve(model, tee=True)


print("Execution time", time.time() - start, "seconds")
print("Expected profit:", model.expected_profit())

# print("Modèle infaisable. Génération d’un fichier LP pour analyse...")
# model.write("debug_model.lp", io_options={"symbolic_solver_labels": True})

# for k, sc in list(in_sample.items())[:5]:
#     print(k)
#     print("wind:", sc["wind"])
#     print("DA_price:", sc["DA_price"])
#     print("power:", sc["power"])
#     print("----------")

# Store the results
results = {"p_DA":{}, "imbalance":{}, "power excess":{}, "power deficit":{}}

for t in model.init_time : 
    results["p_DA"][f"Hour {t}"] = (value(model.p_DA[t]))
for t in model.init_time :
    results["imbalance"][f"Hour {t}"] = [(value(model.imbalance[t,w])) for w in model.init_scenarios]
for t in model.init_time :
    results["power excess"][f"Hour {t}"] = [(value(model.power_excess[t,w])) for w in model.init_scenarios]
for t in model.init_time :
    results["power deficit"][f"Hour {t}"] = [(value(model.power_deficit[t,w])) for w in model.init_scenarios]

print("p_DA", results["p_DA"])

for k, sc in list(in_sample.items())[:5]:
    print(k)
    print("wind:", sc["wind"][:5])
    print("DA_price:", sc["DA_price"][:5])
    print("power:", sc["power"][:5])
    print("----------")

### Profit of each scenario
profits=[]
for w in model.init_scenarios:
    profits.append(sum(model.proba[w]*(model.forecasted_DA_price[t,w]*results["p_DA"][f"Hour {t}"]
                                   +data["misc"]["Coeff excess"]*model.forecasted_DA_price[t,w]*results["imbalance"][f"Hour {t}"][w-1]*model.realtime_power[t,w]
                                   +data["misc"]["Coeff deficit"]*model.forecasted_DA_price[t,w]*results["imbalance"][f"Hour {t}"][w-1]*(1-model.realtime_power[t,w])) 
                                   for t in model.init_time))
print("Profit for each scenario",profits[:5])
print("Total profit", model.expected_profit(), sum(profits))  ## C'est bon, les 2 sont égaux

print("Execution time", time.time() - start, "seconds")
### Plot a cumulative distributive of profit across the scenarios
x=np.arange(nbr_in_sample)

plt.hist(profits, density=True, bins=30, cumulative=True, histtype='step')
plt.title('One price : Cumulative distribution function')
plt.xlabel("Expected profit (€)")
plt.ylabel("Probability")
plt.show()

# X2 = np.sort(profits)
# F2 = np.array(range(nbr_in_sample))/float(nbr_in_sample)

# plt.plot(X2, F2)

# plt.title('Cumulative distribution function')

# plt.show()