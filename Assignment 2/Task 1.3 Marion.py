import numpy as np
import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from itertools import product


import time

start = time.time()



from utils import *

k=8  # --> In sample scenario = 200
# k=10  # --> In sample scenario = 160
#k=16  # --> In sample scenario = 100
#k=4  # --> In sample scenario = 400
all_scenarios=create_all_scenarios(k)


in_sample_analysis={}
out_sample_analysis={}


for i in range(len(all_scenarios)): # 8 

    ####### In Sample analysis

    in_sample_analysis[f"Fold{i+1}"]={"one price":{}, "two price": {}}
    
    in_sample_analysis[f"Fold{i+1}"]["one price"]=oneprice(all_scenarios[f"Fold{i+1}"]["In Sample"][0]) #[0] bec allscen[Fold][Insample] is a 1-elem list
    in_sample_analysis[f"Fold{i+1}"]["two price"]=twoprice(all_scenarios[f"Fold{i+1}"]["In Sample"][0])
    print(in_sample_analysis[f"Fold{i+1}"]["one price"]["Expected profit"])
    print(in_sample_analysis[f"Fold{i+1}"]["two price"]["Expected profit"])


    ###### Out of sample analysis
    out_sample_analysis[f"Fold{i+1}"]={"one price":{"imbalance":{}, "cost":{}}, "two price": {"imbalance":{}, "cost":{}}}

    
    for j in range(len(all_scenarios["Fold1"]["Out of Sample"][0])): #1400
        # Calculate the imbalance
        out_sample_analysis[f"Fold{i+1}"]["one price"]["imbalance"][f"Sc{j+1}"]=np.array(all_scenarios[f"Fold{i+1}"]["Out of Sample"][0][f"Sc{j+1}"]["wind"])-np.array(list(in_sample_analysis[f"Fold{i+1}"]["one price"]["results"]["p_DA"].values()))
        out_sample_analysis[f"Fold{i+1}"]["two price"]["imbalance"][f"Sc{j+1}"]=np.array(all_scenarios[f"Fold{i+1}"]["Out of Sample"][0][f"Sc{j+1}"]["wind"])-np.array(list(in_sample_analysis[f"Fold{i+1}"]["two price"]["results"]["p_DA"].values()))
        # Calculate the profit for each out of sample scenario
        out_sample_analysis[f"Fold{i+1}"]["one price"]["cost"][f"Sc{j+1}"]=one_price_profit(all_scenarios[f"Fold{i+1}"]["Out of Sample"][0][f"Sc{j+1}"], out_sample_analysis[f"Fold{i+1}"]["one price"]["imbalance"][f"Sc{j+1}"],list(in_sample_analysis[f"Fold{i+1}"]["one price"]["results"]["p_DA"].values()))
        out_sample_analysis[f"Fold{i+1}"]["two price"]["cost"][f"Sc{j+1}"]=two_price_profit(all_scenarios[f"Fold{i+1}"]["Out of Sample"][0][f"Sc{j+1}"], out_sample_analysis[f"Fold{i+1}"]["two price"]["imbalance"][f"Sc{j+1}"],list(in_sample_analysis[f"Fold{i+1}"]["two price"]["results"]["p_DA"].values()))
    
    out_sample_analysis[f"Fold{i+1}"]["one price"]["Expected profit"]=np.mean([cost for cost in out_sample_analysis[f"Fold{i+1}"]["one price"]["cost"].values()])
    out_sample_analysis[f"Fold{i+1}"]["two price"]["Expected profit"]=np.mean([cost for cost in out_sample_analysis[f"Fold{i+1}"]["two price"]["cost"].values()])
    

IS_average_expected_profit_one_price=np.mean([in_sample_analysis[f"Fold{i+1}"]["one price"]["Expected profit"] for i in range(len(all_scenarios))])
IS_average_expected_profit_two_price=np.mean([in_sample_analysis[f"Fold{i+1}"]["two price"]["Expected profit"] for i in range(len(all_scenarios))]) 
print("IS mean one price", IS_average_expected_profit_one_price, "IS mean two prices", IS_average_expected_profit_two_price)

OS_average_expected_profit_one_price=np.mean([out_sample_analysis[f"Fold{i+1}"]["one price"]["Expected profit"] for i in range(len(all_scenarios))])
OS_average_expected_profit_two_price=np.mean([out_sample_analysis[f"Fold{i+1}"]["two price"]["Expected profit"] for i in range(len(all_scenarios))]) 
print("OS mean one price", OS_average_expected_profit_one_price, "OS mean two prices", OS_average_expected_profit_two_price)




print("Execution time", time.time() - start, "seconds")
print(in_sample_analysis["Fold1"]["two price"]["results"]["p_DA"].values())

### Plot a cumulative distributive of profit across the scenarios
x=np.arange(24)

plt.plot(x,in_sample_analysis["Fold1"]["one price"]["results"]["p_DA"].values(), label="One price")
plt.plot(x,in_sample_analysis["Fold1"]["two price"]["results"]["p_DA"].values(), label="Two price")
plt.title('One price VS Two price')
plt.xlabel("Hour")
plt.legend()
plt.ylabel("Wind production (MWh)")
plt.show()