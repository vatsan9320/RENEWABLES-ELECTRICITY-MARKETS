import pandas as pd
import numpy as np

def get_data():

    ## Wind power scenarios
    nbr_wind_power_scenario=20
    wind_data={}
    file_path_wind = "Assignment 2\Wind_data.csv"
    for i in range(nbr_wind_power_scenario):
        wind_data[f"Sc{i+1}"] = pd.read_csv(file_path_wind, sep=";")["Capacity_factor"].iloc[i*24:(i+1)*24].tolist()

    # ## Wind power scenarios ############ Version des 2 compères
    # nbr_wind_power_scenario=20
    # wind_data={}
    # file_path_wind = "Assignment 2\Wind_Scenarios_for_20_Days 1.csv"
    # for i in range(nbr_wind_power_scenario):
    #     wind_data[f"Sc{i+1}"] = pd.read_csv(file_path_wind, sep=",")["offshore"].iloc[i*24:(i+1)*24].tolist()


    ## Day Ahead Market price scenarios
    nbr_DA_price_scenario=20
    price_data={}
    file_path_price = "Assignment 2\DA_price_data copy.csv"
    for i in range(nbr_DA_price_scenario):
        price_data[f"Sc{i+1}"] = pd.read_csv(file_path_price, sep=";")["DA price (€/MWh)"].iloc[i*24:(i+1)*24].tolist()
    

    ## Real-Time Power Scenario Generation (24 random binary (two-state) variables generated using Bernoulli distribution)
    # nbr_power_scenario=4
    # power_data={}
    # for i in range(nbr_power_scenario):
    #     power_data[f"Sc{i+1}"] = np.random.binomial(1, 0.5, 24).tolist()

    ## We fix the power data scenarios to always have the same results
    power_data={'Sc1': [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0], 
                'Sc2': [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1], 
                'Sc3': [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], 
                'Sc4': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]}

    ## Miscellaneous
    misc={}
    misc["WF capacity (MW)"] = 500
    misc["Coeff deficit"] = 1.25
    misc["Coeff excess"] = 0.85


    return {"wind_scenarios" : wind_data, "DA_price_scenarios" : price_data, "power_scenarios" : power_data, "misc" : misc}

