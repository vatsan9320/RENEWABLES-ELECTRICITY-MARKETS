# ############ TASK 1.4 RISK AVERSION
#
# Risk-Averse Offering Strategy: Following Lecture 9, formulate and solve the risk-
# averse offering strategy problem for the wind farm under both one- and two-price balancing
# schemes (α = 0.90). Gradually increase the value of β from zero and plot a two-dimensional
# figure showing expected profit versus Conditional Value at Risk (CVaR). Explain how the
# offering strategy and profit volatility evolve as β increases. Additionally, discuss how
# the profit distribution across scenarios changes when risk considerations are incorporated.
# Lastly, analyze whether changing the set and number of in-sample scenarios leads to signif-
# icant changes in the risk-averse offering decisions. This task does not require any ex-post
# out-of-sample or cross-validation analyses
# ############
import Task_1_all_data 
import numpy as np
import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt
import time
from utils import create_all_scenarios

start = time.time()


alpha=0.90

## Data

data=Task_1_all_data.get_data()
nbr_in_sample=200

# ## Construction of the In-Sample Scenarios
# def create_in_sample():
#     in_sample={}
#     i=1
#     while len(in_sample)<nbr_in_sample:
        
#         scen=(data["wind_scenarios"][f"Sc{np.random.randint(1, len(data['wind_scenarios']))}"], 
#             data["DA_price_scenarios"][f"Sc{np.random.randint(1,len(data['DA_price_scenarios']))}"], 
#             data["power_scenarios"][f"Sc{np.random.randint(1,len(data['power_scenarios']))}"] )

#         if scen not in in_sample.values():
#             in_sample[f"Sc{i}"]={}   
#             in_sample[f"Sc{i}"]["wind"]=[scen[0][j]*data["misc"]["WF capacity (MW)"] for j in range(len(scen[0]))]
#             in_sample[f"Sc{i}"]["DA_price"]=scen[1]
#             in_sample[f"Sc{i}"]["power"]=scen[2]

#             i+=1
#     return in_sample

# in_sample_scenarios=create_in_sample()

###################################################################################################
#### OTHER WAY TO CREATE THE IN SAMPLE : To always have the same ones
all_scenarios=create_all_scenarios(8)
in_sample_scenarios=all_scenarios["Fold1"]["In Sample"][0]
###################################################################################################

def oneprice_risk_aversion(in_sample, beta:NonNegativeReals):

    ## Create a model
    model1 = ConcreteModel()

    ## Define the sets
    
    model1.init_time=RangeSet(24)
    model1.init_scenarios=RangeSet(nbr_in_sample)


    ## Declare the variables

    model1.VaR = Var(within=NonNegativeReals, initialize=0)
    #Indexed by w
    model1.aux_var = Var(model1.init_scenarios, initialize=0, within=NonNegativeReals)
    #Indexed by t
    model1.p_DA = Var(model1.init_time,  within=NonNegativeReals, initialize=0)

    #Indexed by t and w 
    model1.power_excess = Var(model1.init_time, model1.init_scenarios, within=NonNegativeReals, initialize=0)
    model1.power_deficit = Var(model1.init_time, model1.init_scenarios, within=NonNegativeReals,  initialize=0)
    model1.imbalance = Var(model1.init_time, model1.init_scenarios, within=Reals, initialize=0)

    ## Declare parameters
    model1.proba = Param(model1.init_scenarios, initialize=1/nbr_in_sample)
    model1.P_nom = Param(model1.init_time, initialize=data["misc"]["WF capacity (MW)"])
    model1.forecasted_DA_price = Param(model1.init_time, model1.init_scenarios, 
                                    initialize={(t, w): in_sample[f"Sc{w}"]["DA_price"][t-1] 
                                                for t in model1.init_time for w in model1.init_scenarios})
    model1.forecasted_wind_power =  Param(model1.init_time, model1.init_scenarios, 
                                    initialize={(t, w): in_sample[f"Sc{w}"]["wind"][t-1] 
                                                for t in model1.init_time for w in model1.init_scenarios})
    model1.realtime_power =  Param(model1.init_time, model1.init_scenarios, 
                                    initialize={(t, w): in_sample[f"Sc{w}"]["power"][t-1] 
                                                for t in model1.init_time for w in model1.init_scenarios})

    ## Objective function

    def objective_rule(model1):
        return (sum(model1.proba[w]*(model1.forecasted_DA_price[t,w]*model1.p_DA[t]
                                    +data["misc"]["Coeff excess"]*model1.forecasted_DA_price[t,w]*model1.power_excess[t,w]*model1.realtime_power[t,w]
                                    -data["misc"]["Coeff excess"]*model1.forecasted_DA_price[t,w]*model1.power_deficit[t,w]*model1.realtime_power[t,w]
                                    +data["misc"]["Coeff deficit"]*model1.forecasted_DA_price[t,w]*model1.power_excess[t,w]*(1-model1.realtime_power[t,w])
                                    -data["misc"]["Coeff deficit"]*model1.forecasted_DA_price[t,w]*model1.power_deficit[t,w]*(1-model1.realtime_power[t,w])) 
                                    for t in model1.init_time for w in model1.init_scenarios )
                + beta*(model1.VaR-1/(1-alpha)*sum(model1.proba[w]*model1.aux_var[w] for w in model1.init_scenarios)))

    model1.objective = Objective(rule=objective_rule, sense= maximize)


    ## Constraints

    def max_wind_farm_capacity(model1,t):
        return model1.p_DA[t]<=data["misc"]["WF capacity (MW)"]
    model1.wind_farm_capacity_constraint=Constraint(model1.init_time, rule=max_wind_farm_capacity)

    def imbalance_production(model1,t,w):
        return model1.imbalance[t,w]== model1.forecasted_wind_power[t,w]-model1.p_DA[t]
    model1.imbalance_production_constraint=Constraint(model1.init_time, model1.init_scenarios, rule=imbalance_production)

    def imbalance_excess_deficit(model1,t,w):
        return model1.imbalance[t,w]== model1.power_excess[t,w]-model1.power_deficit[t,w]
    model1.imbalance_exc_def_constraint=Constraint(model1.init_time, model1.init_scenarios, rule=imbalance_excess_deficit)

    ## New constraints compare to the previous optimization expected profit pb
    
    # auxiliary variable positivity already ensures by the within=NonNegativeReals in the Var creation

    def auxiliary_variable(model1,w):
        return (-sum(model1.forecasted_DA_price[t,w]*model1.p_DA[t]
                                    +data["misc"]["Coeff excess"]*model1.forecasted_DA_price[t,w]*model1.power_excess[t,w]*model1.realtime_power[t,w]
                                    -data["misc"]["Coeff excess"]*model1.forecasted_DA_price[t,w]*model1.power_deficit[t,w]*model1.realtime_power[t,w]
                                    +data["misc"]["Coeff deficit"]*model1.forecasted_DA_price[t,w]*model1.power_excess[t,w]*(1-model1.realtime_power[t,w])
                                    -data["misc"]["Coeff deficit"]*model1.forecasted_DA_price[t,w]*model1.power_deficit[t,w]*(1-model1.realtime_power[t,w])
                                    for t in model1.init_time)
                                    + model1.VaR
                                    - model1.aux_var[w]) <= 0
    

    model1.auxiliary_variable_constraint=Constraint(model1.init_scenarios, rule=auxiliary_variable)

    ## Solve the problem

    #Dual variables
    model1.dual = Suffix(direction=Suffix.IMPORT)

    # Create a solver 
    solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

    # Solve the model1
    solution = solver.solve(model1, tee=True)


    # Store the results
    results = {"p_DA":{}, "imbalance":{}, "power excess":{}, "power deficit":{}, "VaR":{}, "Auxiliary Var":{}, "Expected profit":{}, "CVaR":{}}

    for t in model1.init_time : 
        results["p_DA"][f"Hour {t}"] = (value(model1.p_DA[t]))
    for t in model1.init_time :
        results["imbalance"][f"Hour {t}"] = [(value(model1.imbalance[t,w])) for w in model1.init_scenarios]
    for t in model1.init_time :
        results["power excess"][f"Hour {t}"] = [(value(model1.power_excess[t,w])) for w in model1.init_scenarios]
    for t in model1.init_time :
        results["power deficit"][f"Hour {t}"] = [(value(model1.power_deficit[t,w])) for w in model1.init_scenarios]
    results["VaR"]=value(model1.VaR)
    for w in model1.init_scenarios:
        results["Auxiliary Var"][f"Sc{w}"]=value(model1.aux_var[w])

    # To stock the expected profit term and the CVaR term
    results["Expected profit"]=sum(model1.proba[w]*(model1.forecasted_DA_price[t,w]*value(model1.p_DA[t])
                                    +data["misc"]["Coeff excess"]*model1.forecasted_DA_price[t,w]*value(model1.power_excess[t,w])*model1.realtime_power[t,w]
                                    -data["misc"]["Coeff excess"]*model1.forecasted_DA_price[t,w]*value(model1.power_deficit[t,w])*model1.realtime_power[t,w]
                                    +data["misc"]["Coeff deficit"]*model1.forecasted_DA_price[t,w]*value(model1.power_excess[t,w])*(1-model1.realtime_power[t,w])
                                    -data["misc"]["Coeff deficit"]*model1.forecasted_DA_price[t,w]*value(model1.power_deficit[t,w])*(1-model1.realtime_power[t,w])) 
                                    for t in model1.init_time for w in model1.init_scenarios)
    results["CVaR"]=value(model1.VaR)-1/(1-alpha)*sum(model1.proba[w]*value(model1.aux_var[w]) for w in model1.init_scenarios)

    ### Profit of each scenario
    profits=[]
    for w in model1.init_scenarios:
        profits.append(sum(model1.proba[w]*(model1.forecasted_DA_price[t,w]*results["p_DA"][f"Hour {t}"]
                                    +data["misc"]["Coeff excess"]*model1.forecasted_DA_price[t,w]*results["imbalance"][f"Hour {t}"][w-1]*model1.realtime_power[t,w]
                                    +data["misc"]["Coeff deficit"]*model1.forecasted_DA_price[t,w]*results["imbalance"][f"Hour {t}"][w-1]*(1-model1.realtime_power[t,w])) 
                                    for t in model1.init_time))

    # print("VaR", results["VaR"])
    # print("Aux var", results["Auxiliary Var"])
    # print("Expected Profit", results["Expected profit"])
    # print("CVaR", results["CVaR"])
    # print("Expected profit + Beta CVaR:", model1.objective(), "Compared to", results["Expected profit"]+beta*results["CVaR"])

    #return {"results":results, "profits":profits, "Expected profit + beta*CVaR":model1.expected_profit()}

    return {"results":results, "profits":profits, "Expected profit + beta*CVaR":model1.objective()}
    
    


def twoprice_risk_aversion(in_sample, beta:NonNegativeReals):

    ## Create a model
    model2 = ConcreteModel()

    ## Define the sets
    # model2.init_time=RangeSet(24)
    model2.init_time=RangeSet(24)
    model2.init_scenarios=RangeSet(nbr_in_sample)


    ## Declare the variables
    model2.VaR = Var(within=NonNegativeReals, initialize=0)
    #Indexed by w
    model2.aux_var = Var(model2.init_scenarios, initialize=0, within=NonNegativeReals)
    #Indexed by t
    model2.p_DA = Var(model2.init_time,  within=NonNegativeReals, initialize=0)

    #Indexed by t and w 
    model2.power_excess = Var(model2.init_time, model2.init_scenarios, within=NonNegativeReals, initialize=0)
    model2.power_deficit = Var(model2.init_time, model2.init_scenarios, within=NonNegativeReals,  initialize=0)
    model2.imbalance = Var(model2.init_time, model2.init_scenarios, within=Reals, initialize=0)

    ## Declare parameters
    model2.proba = Param(model2.init_scenarios, initialize=1/nbr_in_sample)
    model2.P_nom = Param(model2.init_time, initialize=data["misc"]["WF capacity (MW)"])
    model2.forecasted_DA_price = Param(model2.init_time, model2.init_scenarios, 
                                    initialize={(t, w): in_sample[f"Sc{w}"]["DA_price"][t-1] 
                                                for t in model2.init_time for w in model2.init_scenarios})
    model2.forecasted_wind_power =  Param(model2.init_time, model2.init_scenarios, 
                                    initialize={(t, w): in_sample[f"Sc{w}"]["wind"][t-1] 
                                                for t in model2.init_time for w in model2.init_scenarios})
    model2.realtime_power =  Param(model2.init_time, model2.init_scenarios, 
                                    initialize={(t, w): in_sample[f"Sc{w}"]["power"][t-1] 
                                                for t in model2.init_time for w in model2.init_scenarios})

    ## Objective function

    def objective_rule(model2):
        return ((sum(model2.proba[w]*(model2.forecasted_DA_price[t,w]*model2.p_DA[t]
                                    +data["misc"]["Coeff excess"]*model2.forecasted_DA_price[t,w]*model2.power_excess[t,w]*model2.realtime_power[t,w]
                                    - model2.forecasted_DA_price[t,w]*model2.power_deficit[t,w]*model2.realtime_power[t,w]
                                    + model2.forecasted_DA_price[t,w]*model2.power_excess[t,w]*(1-model2.realtime_power[t,w])
                                    -data["misc"]["Coeff deficit"]*model2.forecasted_DA_price[t,w]*model2.power_deficit[t,w]*(1-model2.realtime_power[t,w])) 
                                    for t in model2.init_time for w in model2.init_scenarios ))
                + beta*(model2.VaR-1/(1-alpha)*sum(model2.proba[w]*model2.aux_var[w] for w in model2.init_scenarios)))

    model2.objective = Objective(rule=objective_rule, sense= maximize)


    ## Constraints

    def max_wind_farm_capacity(model2,t):
        return model2.p_DA[t]<=data["misc"]["WF capacity (MW)"]
    model2.wind_farm_capacity_constraint=Constraint(model2.init_time, rule=max_wind_farm_capacity)

    def imbalance_production(model2,t,w):
        return model2.imbalance[t,w]== model2.forecasted_wind_power[t,w]-model2.p_DA[t]
    model2.imbalance_production_constraint=Constraint(model2.init_time, model2.init_scenarios, rule=imbalance_production)

    def imbalance_excess_deficit(model2,t,w):
        return model2.imbalance[t,w]== model2.power_excess[t,w]-model2.power_deficit[t,w]
    model2.imbalance_exc_def_constraint=Constraint(model2.init_time, model2.init_scenarios, rule=imbalance_excess_deficit)

    ## New constraints compare to the previous optimization expected profit pb
    
    # auxiliary variable positivity already ensures by the within=NonNegativeReals in the Var creation

    def auxiliary_variable(model2,w):
        return (-(sum(model2.forecasted_DA_price[t,w]*model2.p_DA[t]
                                    +data["misc"]["Coeff excess"]*model2.forecasted_DA_price[t,w]*model2.power_excess[t,w]*model2.realtime_power[t,w]
                                    - model2.forecasted_DA_price[t,w]*model2.power_deficit[t,w]*model2.realtime_power[t,w]
                                    + model2.forecasted_DA_price[t,w]*model2.power_excess[t,w]*(1-model2.realtime_power[t,w])
                                    -data["misc"]["Coeff deficit"]*model2.forecasted_DA_price[t,w]*model2.power_deficit[t,w]*(1-model2.realtime_power[t,w])
                                    for t in model2.init_time))
                                    + model2.VaR
                                    - model2.aux_var[w]) <= 0
    model2.auxiliary_variable_constraint=Constraint(model2.init_scenarios, rule=auxiliary_variable)

    ## Solve the problem

    #Dual variables
    model2.dual = Suffix(direction=Suffix.IMPORT)

    # Create a solver 
    solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

    # Solve the model2
    solution = solver.solve(model2, tee=True)

    # Store the results
    results = {"p_DA":{}, "imbalance":{}, "power excess":{}, "power deficit":{}, "VaR":{}, "Auxiliary Var":{}, "Expected profit":{}, "CVaR":{}}

    for t in model2.init_time : 
        results["p_DA"][f"Hour {t}"] = (value(model2.p_DA[t]))
    for t in model2.init_time :
        results["imbalance"][f"Hour {t}"] = [(value(model2.imbalance[t,w])) for w in model2.init_scenarios]
    for t in model2.init_time :
        results["power excess"][f"Hour {t}"] = [(value(model2.power_excess[t,w])) for w in model2.init_scenarios]
    for t in model2.init_time :
        results["power deficit"][f"Hour {t}"] = [(value(model2.power_deficit[t,w])) for w in model2.init_scenarios]

    results["VaR"]=value(model2.VaR)
    for w in model2.init_scenarios:
        results["Auxiliary Var"][f"Sc{w}"]=value(model2.aux_var[w])

    # To stock the expected profit term and the CVaR term
    results["Expected profit"]=sum(model2.proba[w]*(model2.forecasted_DA_price[t,w]*value(model2.p_DA[t])
                                    +data["misc"]["Coeff excess"]*model2.forecasted_DA_price[t,w]*value(model2.power_excess[t,w])*model2.realtime_power[t,w]
                                    - model2.forecasted_DA_price[t,w]*value(model2.power_deficit[t,w])*model2.realtime_power[t,w]
                                    + model2.forecasted_DA_price[t,w]*value(model2.power_excess[t,w])*(1-model2.realtime_power[t,w])
                                    -data["misc"]["Coeff deficit"]*model2.forecasted_DA_price[t,w]*value(model2.power_deficit[t,w])*(1-model2.realtime_power[t,w])) 
                                    for t in model2.init_time for w in model2.init_scenarios )
    results["CVaR"]=value(model2.VaR)-1/(1-alpha)*sum(model2.proba[w]*value(model2.aux_var[w]) for w in model2.init_scenarios)


    ### Profit of each scenario
    profits=[]
    for w in model2.init_scenarios:
        profits.append(sum(model2.proba[w]*(model2.forecasted_DA_price[t,w]*results["p_DA"][f"Hour {t}"]
                                    +data["misc"]["Coeff excess"]*model2.forecasted_DA_price[t,w]*results["power excess"][f"Hour {t}"][w-1]*model2.realtime_power[t,w]
                                    - model2.forecasted_DA_price[t,w]*results["power deficit"][f"Hour {t}"][w-1]*model2.realtime_power[t,w]
                                    + model2.forecasted_DA_price[t,w]*results["power excess"][f"Hour {t}"][w-1]*(1-model2.realtime_power[t,w])
                                    -data["misc"]["Coeff deficit"]*model2.forecasted_DA_price[t,w]*results["power deficit"][f"Hour {t}"][w-1]*(1-model2.realtime_power[t,w])) 
                                    for t in model2.init_time))
        
    # print("VaR", results["VaR"])
    # print("Aux var", results["Auxiliary Var"])
    # print("Expected Profit", results["Expected profit"])
    # print("CVaR", results["CVaR"])
    #print("Expected profit + Beta CVaR:", model2.objective(), "Compared to", results["Expected profit"]+beta*results["CVaR"])

    return {"results":results,  "profits": profits, "Expected profit + beta*CVaR":model2.objective()}

print("Execution time", time.time() - start, "seconds")

oneprice_expected_profit, twoprice_expected_profit=[], []
oneprice_CVaR, twoprice_CVaR=[], []
oneprice_profit_each_sc, twoprice_profit_each_sc=[],[]

beta_list=[0, 0.1, 0.5, 1, 2, 5, 10, 100]

for beta in beta_list:
    oneprice_risk_results=oneprice_risk_aversion(in_sample_scenarios, beta)
    oneprice_expected_profit.append(oneprice_risk_results["results"]["Expected profit"])
    oneprice_CVaR.append(oneprice_risk_results["results"]["CVaR"])
    oneprice_profit_each_sc.append(oneprice_risk_results["profits"])

    twoprice_risk_results=twoprice_risk_aversion(in_sample_scenarios, beta)
    twoprice_expected_profit.append(twoprice_risk_results["results"]["Expected profit"])
    twoprice_CVaR.append(twoprice_risk_results["results"]["CVaR"])
    twoprice_profit_each_sc.append(twoprice_risk_results["profits"])

plt.plot(oneprice_CVaR, oneprice_expected_profit, label="One price", marker='o')
plt.legend()
plt.xlabel('CVaR (€)')
plt.ylabel('Expected profit (€)')
plt.title('One price: Efficient frontier')
# Add the beta value for each point
for i, beta in enumerate(beta_list):
    plt.annotate(f"\u03B2={beta}", (oneprice_CVaR[i], oneprice_expected_profit[i]),
                 textcoords="offset points", xytext=(5, 5), ha='center')
plt.show()


plt.plot(twoprice_CVaR, twoprice_expected_profit, label="Two price", marker='o')
plt.legend()
plt.xlabel('CVaR (€)')
plt.ylabel('Expected profit (€)')
plt.title(' Two price: Efficient frontier')
# Add the beta value for each point
for i, beta in enumerate(beta_list):
    plt.annotate(f"\u03B2={beta}", (twoprice_CVaR[i], twoprice_expected_profit[i]),
                 textcoords="offset points", xytext=(5, 5), ha='center')
plt.show()



##### CDF CURVES
### ONE PRICE CDF
plt.hist(oneprice_profit_each_sc[0], density=True, bins=30, cumulative=True, histtype='step', label="\u03B2=0")
plt.hist(oneprice_profit_each_sc[4], density=True, bins=30, cumulative=True, histtype='step', label="\u03B2=1")
plt.hist(oneprice_profit_each_sc[7], density=True, bins=30, cumulative=True, histtype='step', label="\u03B2=10")
plt.legend()
plt.title("One price Cumulative distribution function")
plt.xlabel("Expected profit (€)")
plt.ylabel("Probability")
plt.show()

### TWO PRICE CDF
plt.hist(twoprice_profit_each_sc[0], density=True, bins=30, cumulative=True, histtype='step', label="\u03B2=0")
plt.hist(twoprice_profit_each_sc[4], density=True, bins=30, cumulative=True, histtype='step', label="\u03B2=1")
plt.hist(twoprice_profit_each_sc[7], density=True, bins=30, cumulative=True, histtype='step', label="\u03B2=10")
plt.legend()
plt.title("Two Cumulative distribution function")
plt.xlabel("Expected profit (€)")
plt.ylabel("Probability")
plt.show()


#ONE PROFIT VS TWO PROFIT
plt.hist(twoprice_profit_each_sc[0], density=True, bins=30, cumulative=True, histtype='step', label="Two price \u03B2=0")

plt.hist(twoprice_profit_each_sc[7], density=True, bins=30, cumulative=True, histtype='step', label="Two price \u03B2=10")
plt.hist(oneprice_profit_each_sc[0], density=True, bins=30, cumulative=True, histtype='step', label="One price \u03B2=0")

plt.hist(oneprice_profit_each_sc[7], density=True, bins=30, cumulative=True, histtype='step', label="One price \u03B2=10")
plt.legend()
plt.title("Cumulative distribution function")
plt.xlabel("Expected profit (€)")
plt.ylabel("Probability")
plt.show()

