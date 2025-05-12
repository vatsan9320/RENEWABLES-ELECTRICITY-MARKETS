import numpy as np

def get_data():

# Data Generation for Future Stochastic Load: Randomly generate 300 consumption load
# profiles, ensuring that for each profile, the load at every minute falls between 220 kW and 600
# kW. Additionally, the change in consumption between two consecutive minutes must not exceed
# 35 kW. 

## Data Generation for Future Stochastic Load
    nbr_load_profiles=300
    load_data={}
    for i in range(nbr_load_profiles):
        load_data[f"Profile{i+1}"] = [round(np.random.uniform(220, 600),2)] #min 1
        for t in range(59):   #60 minutes length
            load_data[f"Profile{i+1}"].append(round(np.random.uniform(max(220,load_data[f"Profile{i+1}"][t]-35), min(600,load_data[f"Profile{i+1}"][t]+35)),2))     

    return {"load": load_data}


# a=get_data()
# print(len(a["load"]))