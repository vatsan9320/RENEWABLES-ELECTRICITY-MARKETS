import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data():
    
    generating_unit_data = {
    #Technical data of generating units 
    "Unit #": [1, 2, 3, 4,  5,  6,  7,  8,   9, 10, 11, 12],
    "Node":   [1, 2, 7, 13, 15, 15, 16, 18, 21, 22, 23, 23],
    "Pmax (MW)": [152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 300, 350],
    "Pmin (MW)": [30.4, 30.4, 75, 206.85, 12, 54.25, 54.25, 100, 100, 300, 108.5, 140],
    "R+ (MW)": [40, 40, 70, 180, 60, 30, 30, 80, 80, 0, 60, 40],
    "R- (MW)": [40, 40, 70, 180, 60, 30, 30, 80, 80, 0, 60, 40],
    "RU (MW/h)": [120, 120, 350, 240, 60, 155, 155, 280, 300, 0, 180, 240],
    "RD (MW/h)": [120, 120, 350, 240, 60, 155, 155, 280, 300, 0, 180, 240],
    "UT (h)": [8, 8, 8, 12, 4, 8, 8, 8, 0, 0, 8, 8],
    "DT (h)": [4, 4, 8, 10, 2, 8, 8, 8, 0, 0, 8, 8],

    #Costs and initial states of generating units
    "Ci": [13.32, 13.32, 20.7, 20.93, 26.11, 10.52, 6.02, 5.47, 0, 0, 10.52, 10.89],
    "Cu_i": [15, 15, 10, 8, 7, 7, 16, 0, 0, 0, 16, 16],
    "Cd_i": [14, 14, 9, 7, 5, 6, 14, 0, 0, 0, 14, 14],
    "C+_i": [15, 15, 24, 25, 28, 16, 16, 0, 0, 0, 16, 16],
    "C-_i": [11, 11, 16, 17, 23, 7, 7, 0, 0, 0, 8, 8],
    "Csu_i": [1430.4, 1430.4, 1725, 3056.7, 437, 312, 124, 0, 0, 624, 2298, 2298],
    "Pini_i": [76, 76, 0, 0, 0, 0, 312, 0, 0, 0, 240, 280],
    "Uini_i": [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    "Tini_i": [22, 22, -2, -1, -1, -1, 10, -2, -2, -1, 24, 50]
    }

    # Demand data from pdf file
    load_data = {
    "Hour": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "System demand (MW)": [
        1775.835, 1669.815, 1590.3, 1563.795, 1563.795, 1590.3, 1961.37, 2279.43, 
        2517.975, 2544.48, 2544.48, 2517.975, 2517.975, 2517.975, 2464.965, 2464.965, 
        2623.995, 2650.5, 2650.5, 2544.48, 2411.955, 2199.915, 1934.865, 1669.815 ]
        }

    # Node Location and Distribution of the Total System Demand
    node_demand = {
    "Load #": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    "Node":   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 20],
    "percentage of system load": [3.8, 3.4, 6.3, 2.6, 2.5, 4.8, 4.4, 6.0, 6.1, 6.8, 9.3, 6.8, 11.1, 3.5, 11.7, 6.4, 4.5 ],
    "Load distribution peak" : [np.max(load_data["System demand (MW)"])*x/100 for x in [3.8, 3.4, 6.3, 2.6, 2.5, 4.8, 4.4, 6.0, 6.1, 6.8, 9.3, 6.8, 11.1, 3.5, 11.7, 6.4, 4.5]],
    "Bid price" : [25, 22, 15.4, 12.5, 13, 14, 24, 15, 12.8, 17.8, 29.3, 28, 16.9, 30, 18, 16, 21]
    }

    #Transmission Line Mapping and capacities
    transmission_line = {
    "From": [1, 1, 1, 2, 2, 3, 3,  4, 5,  6,  7, 8, 8,  9,  9,  10, 10, 11, 11, 12, 12, 13, 14, 15, 15, 15, 16, 16, 17, 17, 18, 19, 20, 21],
    "To":   [2, 3, 5, 4, 6, 9, 24, 9, 10, 10, 8, 9, 10, 11, 12, 11, 12, 13, 14, 13, 23, 23, 16, 16, 21, 24, 17, 19, 18, 22, 21, 20, 23, 22],
    "Reactance": 0.002,  # p.u. # we assume a uniform reactance of 0.200 for all lines
    "Susceptance": 500*np.ones(34), # p.u. #we assume a uniform susceptance of 500 for all lines
    # #Capacity with 0 transmission
    # "Capacity (MVA)": np.ones(34)*0
    # # #Capacity with bottlenecks
    # "Capacity (MVA)": [175, 175, 350, 175, 175, 175, 400, 175, 350, 175, 350, 175, 175, 400, 400,
    #     400, 400, 500, 500, 500, 500, 250, 250, 500, 400, 500, 500, 500, 500, 500,1000, 1000, 1000, 500]
    # Normal Capacity 
    "Capacity (MVA)": [175, 175, 350, 175, 175, 175, 400, 175, 350, 175, 350, 175, 175, 400, 400,
        400, 400, 500, 500, 500, 500, 500, 500, 500, 1000, 500, 500, 500, 500, 500,1000, 1000, 1000, 500]
    
    }

    
    #Wind farms data : We extract the production of the 24 first hours of each first dataset from each zone (1 zone 1 file)
    wind_farm = {}
    file_path = ["Wind farms data\scen_zone1.out", "Wind farms data\scen_zone2.out","Wind farms data\scen_zone3.out","Wind farms data\scen_zone4.out","Wind farms data\scen_zone5.out","Wind farms data\scen_zone6.out"]
    for i in range(len(file_path)):
        wind_farm[f"wind_farm {i+1}"] = pd.read_csv(file_path[i])["V1"].head(24).tolist()
    wind_farm["Node"]= [3, 5, 7, 16, 21, 23]
    #Battery data
    # #When no battery
    # battery= {
    #     "Charging efficiency": 0.85,
    #     "Discharging efficiency": 0.9,
    #     "Charging capacity (MW)": 0,
    #     "Discharging capacity (MW)": 0,
    #     "Energy storage capacity (MWh)" : 0
    # }
    #when battery
    battery= {
        "Charging efficiency": 0.85,
        "Discharging efficiency": 0.90,
        "Charging capacity (MW)": 70,
        "Discharging capacity (MW)": 80,
        "Energy storage capacity (MWh)" : 400
    }

    return {"generation_unit" : generating_unit_data, "load" : load_data,  "wind_farm" : wind_farm, "node_demand" : node_demand, "transmission_line": transmission_line, "battery": battery}

test=get_data()
# # print(test["wind_farm"][f"wind_farm {1}"])
# # print(test["wind_farm"][f"wind_farm {1}"][0])
# # print(test["node_demand"]["Load distribution peak"])
# print(len(test["transmission_line"]["From"]))
# print(np.ones(4)*0.2)

# # Calcul de la quantité de vent moyenne à chaque heure   ---> il y a très peu de vent pendant l'heure 1
# list=[]
# for t in range(24):
#     list.append([])
#     for i in range(1,7):
#         list[t].append(test["wind_farm"][f"wind_farm {i}"][t])
#     print(f"Hour {t+1}",np.mean(list[t]))

print(test["transmission_line"]["Capacity (MVA)"][33])
print(range(4))