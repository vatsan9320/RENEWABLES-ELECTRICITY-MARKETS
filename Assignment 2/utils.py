import Task_1_all_data
import numpy as np
import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from itertools import product

data=Task_1_all_data.get_data()


def create_all_scenarios(k:int):

    listwind=range(1,len(data['wind_scenarios'])+1)
    listprice=range(1,len(data['DA_price_scenarios'])+1)
    listrealpower=range(1,len(data['power_scenarios'])+1)

    list_scenarios = np.array(list(product(listwind, listprice, listrealpower)))  #len =1600

    #k = 8
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    all_scenario={}

    a=1
    for train_index, test_index in kf.split(list_scenarios):
        all_scenario[f"Fold{a}"]={"In Sample":[], "Out of Sample":[]}
        in_sample, out_of_sample={}, {}
        
        for i in range(len(test_index)):  # for all the scenarios of the in-sample in this fold fold  ==200
        
            in_sample[f"Sc{i+1}"]={}
            
            in_sample[f"Sc{i+1}"]["DA_price"]=data["DA_price_scenarios"][f"Sc{list_scenarios[test_index][i][1]}"]
            in_sample[f"Sc{i+1}"]["power"]=data["power_scenarios"][f"Sc{list_scenarios[test_index][i][2]}"]
            in_sample[f"Sc{i+1}"]["wind"]=[data["wind_scenarios"][f"Sc{list_scenarios[test_index][i][0]}"][j]*data["misc"]["WF capacity (MW)"] for j in range(24)]
            
        
        for i in range(len(train_index)):  # for all the scenarios of out of the sample in this fold =1400
        
            out_of_sample[f"Sc{i+1}"]={}
            
            out_of_sample[f"Sc{i+1}"]["DA_price"]=data["DA_price_scenarios"][f"Sc{list_scenarios[train_index][i][1]}"]
            out_of_sample[f"Sc{i+1}"]["power"]=data["power_scenarios"][f"Sc{list_scenarios[train_index][i][2]}"]
            out_of_sample[f"Sc{i+1}"]["wind"]=[data["wind_scenarios"][f"Sc{list_scenarios[train_index][i][0]}"][j]*data["misc"]["WF capacity (MW)"] for j in range(24)]
        
        all_scenario[f"Fold{a}"]["In Sample"].append(in_sample)
        all_scenario[f"Fold{a}"]["Out of Sample"].append(out_of_sample)
        
        a=a+1
        # print(len(out_of_sample))

        
        # print(len(in_sample))
    
    return all_scenario
    

def oneprice(in_sample):


    ## Create a model
    model1 = ConcreteModel()

    ## Define the sets
    # model1.init_time=RangeSet(24)
    model1.init_time=RangeSet(24)
    model1.init_scenarios=RangeSet(len(in_sample))


    ## Declare the variables

    #Indexed by t
    model1.p_DA = Var(model1.init_time,  within=NonNegativeReals, initialize=0)

    #Indexed by t and w 
    model1.power_excess = Var(model1.init_time, model1.init_scenarios, within=NonNegativeReals, initialize=0)
    model1.power_deficit = Var(model1.init_time, model1.init_scenarios, within=NonNegativeReals,  initialize=0)
    model1.imbalance = Var(model1.init_time, model1.init_scenarios, within=Reals, initialize=0)

    ## Declare parameters
    model1.proba = Param(model1.init_scenarios, initialize=1/len(in_sample))
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

   
    ## Equivalent à la l'objective funciton avec imbalance : prend en compte les REWARDS et PUNITIONS dans les cas où le system est en excès et déficit

    def objective_rule(model1):
        return sum(model1.proba[w]*(model1.forecasted_DA_price[t,w]*model1.p_DA[t]
                                    +data["misc"]["Coeff excess"]*model1.forecasted_DA_price[t,w]*model1.power_excess[t,w]*model1.realtime_power[t,w]
                                    -data["misc"]["Coeff excess"]*model1.forecasted_DA_price[t,w]*model1.power_deficit[t,w]*model1.realtime_power[t,w]
                                    +data["misc"]["Coeff deficit"]*model1.forecasted_DA_price[t,w]*model1.power_excess[t,w]*(1-model1.realtime_power[t,w])
                                    -data["misc"]["Coeff deficit"]*model1.forecasted_DA_price[t,w]*model1.power_deficit[t,w]*(1-model1.realtime_power[t,w])) 
                                    for t in model1.init_time for w in model1.init_scenarios )

    model1.expected_profit = Objective(rule=objective_rule, sense= maximize)

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

    ## Solve the problem

    #Dual variables
    model1.dual = Suffix(direction=Suffix.IMPORT)

    # Create a solver 
    solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

    # Solve the model1
    solution = solver.solve(model1, tee=True)

    #print("Expected profit:", model1.expected_profit())

    # Store the results
    results = {"p_DA":{}, "imbalance":{}, "power excess":{}, "power deficit":{}}

    for t in model1.init_time : 
        results["p_DA"][f"Hour {t}"] = (value(model1.p_DA[t]))
    for t in model1.init_time :
        results["imbalance"][f"Hour {t}"] = [(value(model1.imbalance[t,w])) for w in model1.init_scenarios]
    for t in model1.init_time :
        results["power excess"][f"Hour {t}"] = [(value(model1.power_excess[t,w])) for w in model1.init_scenarios]
    for t in model1.init_time :
        results["power deficit"][f"Hour {t}"] = [(value(model1.power_deficit[t,w])) for w in model1.init_scenarios]

    #print("p_DA", results["p_DA"])
    
    
    ### Profit of each scenario
    profits=[]
    for w in model1.init_scenarios:
        profits.append(sum(model1.proba[w]*(model1.forecasted_DA_price[t,w]*results["p_DA"][f"Hour {t}"]
                                    +data["misc"]["Coeff excess"]*model1.forecasted_DA_price[t,w]*results["imbalance"][f"Hour {t}"][w-1]*model1.realtime_power[t,w]
                                    +data["misc"]["Coeff deficit"]*model1.forecasted_DA_price[t,w]*results["imbalance"][f"Hour {t}"][w-1]*(1-model1.realtime_power[t,w])) 
                                    for t in model1.init_time))
    # print("Profit for each scenario",profits[:5])
    # print("Total profit", model1.expected_profit(), sum(profits))  ## C'est bon, les 2 sont égaux

    ### Plot a cumulative distributive of profit across the scenarios
    
    # X2 = np.sort(profits)
    # F2 = np.array(range(nbr_in_sample))/float(nbr_in_sample)

    # plt.plot(X2, F2)

    # plt.title('Cumulative distribution function')

    # plt.show()

    return {"results":results, "profits":profits, "Expected profit":model1.expected_profit()}

def twoprice(in_sample):

    ## Create a model
    model2 = ConcreteModel()

    ## Define the sets
    # model2.init_time=RangeSet(24)
    model2.init_time=RangeSet(24)
    model2.init_scenarios=RangeSet(len(in_sample))


    ## Declare the variables

    #Indexed by t
    model2.p_DA = Var(model2.init_time,  within=NonNegativeReals, initialize=0)

    #Indexed by t and w 
    model2.power_excess = Var(model2.init_time, model2.init_scenarios, within=NonNegativeReals, initialize=0)
    model2.power_deficit = Var(model2.init_time, model2.init_scenarios, within=NonNegativeReals,  initialize=0)
    model2.imbalance = Var(model2.init_time, model2.init_scenarios, within=Reals, initialize=0)

    ## Declare parameters
    model2.proba = Param(model2.init_scenarios, initialize=1/len(in_sample))
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

    ## Equivalent à la l'objective funciton avec imbalance : prend en compte les REWARDS et PUNITIONS dans les cas où le system est en excès et déficit

    def objective_rule(model2):
        return sum(model2.proba[w]*(model2.forecasted_DA_price[t,w]*model2.p_DA[t]
                                    +data["misc"]["Coeff excess"]*model2.forecasted_DA_price[t,w]*model2.power_excess[t,w]*model2.realtime_power[t,w]
                                    - model2.forecasted_DA_price[t,w]*model2.power_deficit[t,w]*model2.realtime_power[t,w]
                                    + model2.forecasted_DA_price[t,w]*model2.power_excess[t,w]*(1-model2.realtime_power[t,w])
                                    -data["misc"]["Coeff deficit"]*model2.forecasted_DA_price[t,w]*model2.power_deficit[t,w]*(1-model2.realtime_power[t,w])) 
                                    for t in model2.init_time for w in model2.init_scenarios )

    model2.expected_profit = Objective(rule=objective_rule, sense= maximize)


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

    ## Solve the problem

    #Dual variables
    model2.dual = Suffix(direction=Suffix.IMPORT)

    # Create a solver 
    solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured

    # Solve the model2
    solution = solver.solve(model2, tee=True)

    #print("Two prices Expected profit:", model2.expected_profit())

    # print("Modèle infaisable. Génération d’un fichier LP pour analyse...")
    # model2.write("debug_model2.lp", io_options={"symbolic_solver_labels": True})

    # for k, sc in list(in_sample.items())[:5]:
    #     print(k)
    #     print("wind:", sc["wind"])
    #     print("DA_price:", sc["DA_price"])
    #     print("power:", sc["power"])
    #     print("----------")

    # Store the results
    results = {"p_DA":{}, "imbalance":{}, "power excess":{}, "power deficit":{}}

    for t in model2.init_time : 
        results["p_DA"][f"Hour {t}"] = (value(model2.p_DA[t]))
    for t in model2.init_time :
        results["imbalance"][f"Hour {t}"] = [(value(model2.imbalance[t,w])) for w in model2.init_scenarios]
    for t in model2.init_time :
        results["power excess"][f"Hour {t}"] = [(value(model2.power_excess[t,w])) for w in model2.init_scenarios]
    for t in model2.init_time :
        results["power deficit"][f"Hour {t}"] = [(value(model2.power_deficit[t,w])) for w in model2.init_scenarios]

    # print("p_DA", results["p_DA"])
    # print("imabalance", results["imbalance"]['Hour 8'],results["imbalance"]['Hour 9'])
    # print("excess", results["power excess"]['Hour 8'])
    # print("deficit",results["power deficit"]['Hour 8'])
    # for k, sc in list(in_sample.items())[:5]:
    #     print(k)
    #     print("wind:", sc["wind"][:5])
    #     print("DA_price:", sc["DA_price"][:5])
    #     print("power:", sc["power"][:5])
    #     print("----------")

    ### Profit of each scenario
    profits=[]
    for w in model2.init_scenarios:
        profits.append(sum(model2.proba[w]*(model2.forecasted_DA_price[t,w]*results["p_DA"][f"Hour {t}"]
                                    +data["misc"]["Coeff excess"]*model2.forecasted_DA_price[t,w]*results["power excess"][f"Hour {t}"][w-1]*model2.realtime_power[t,w]
                                    - model2.forecasted_DA_price[t,w]*results["power deficit"][f"Hour {t}"][w-1]*model2.realtime_power[t,w]
                                    + model2.forecasted_DA_price[t,w]*results["power excess"][f"Hour {t}"][w-1]*(1-model2.realtime_power[t,w])
                                    -data["misc"]["Coeff deficit"]*model2.forecasted_DA_price[t,w]*results["power deficit"][f"Hour {t}"][w-1]*(1-model2.realtime_power[t,w])) 
                                    for t in model2.init_time))
    # print("Profit for each scenario",profits[:5])
    # print("Total profit", model2.expected_profit(), sum(profits))  ## C'est bon, les 2 sont égaux

    ### Plot a cumulative distributive of profit across the scenarios
    

    # X3 = np.sort(profits)
    # F3 = np.array(range(nbr_in_sample))/float(nbr_in_sample)

    # plt.plot(X3, F3)

    # plt.title('Two prices : Cumulative distribution function')

    # plt.show()


    return {"results":results, "profits":profits, "Expected profit":model2.expected_profit()}


def one_price_profit(scenarios, power_imbalance, pDA):
    cost=[]
    
    for t in range(24):
        cost.append(scenarios["DA_price"][t]*pDA[t]
                    +data["misc"]["Coeff excess"]*scenarios["DA_price"][t]*power_imbalance[t]*scenarios["power"][t]
                                +data["misc"]["Coeff deficit"]*scenarios["DA_price"][t]*power_imbalance[t]*(1-scenarios["power"][t]) 
                                )
    return np.sum(cost)  # profit of 1 day
    
def two_price_profit(scenarios, power_imbalance, pDA):
    power_excess, power_deficit=[0]*24, [0]*24
    cost=[]
    for t in range(24):
        if power_imbalance[t]>0:
            power_excess[t]=power_imbalance[t]
        else:
            power_deficit[t]=power_imbalance[t]
        cost.append(scenarios["DA_price"][t]*pDA[t]
                                +data["misc"]["Coeff excess"]*scenarios["DA_price"][t]*power_excess[t]*scenarios["power"][t]
                                +scenarios["DA_price"][t]*power_deficit[t]*scenarios["power"][t] #power deficit est négatif donc pas besoin du signe -
                                +scenarios["DA_price"][t]*power_excess[t]*(1-scenarios["power"][t])
                                +data["misc"]["Coeff deficit"]*scenarios["DA_price"][t]*power_deficit[t]*(1-scenarios["power"][t]))

    return np.sum(cost)