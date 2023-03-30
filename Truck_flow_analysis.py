#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.random import choice
import seaborn as sns



from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import warnings
warnings.filterwarnings("ignore",category = FutureWarning)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
import urllib.request
import csv
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# Ensure compatibility between python 2 and python 3
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import requests
import platform
import os
import stat
import tempfile
import json
import time
import subprocess
import geopandas as gpd
import shutil

def _download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = requests.get(url)
        # write to file
        file.write(response.content)

_inmap_exe = None
_tmpdir = tempfile.TemporaryDirectory()


if _inmap_exe == None:
    ost = platform.system()
    print("Downloading InMAP executable for %s"%ost, end='\r')
    if ost == "Windows":
        _inmap_exe = os.path.join(_tmpdir.name, "inmap_1.7.2.exe")
        _download("https://github.com/spatialmodel/inmap/releases/download/v1.7.2/inmap1.7.2windows-amd64.exe", _inmap_exe)
    elif ost == "Darwin":
        _inmap_exe = os.path.join(_tmpdir.name, "inmap_1.7.2")
        _download("https://github.com/spatialmodel/inmap/releases/download/v1.7.2/inmap1.7.2darwin-amd64", _inmap_exe)
    elif ost == "Linux":
        _inmap_exe = os.path.join(_tmpdir.name, "inmap_1.7.2")
        _download("https://github.com/spatialmodel/inmap/releases/download/v1.7.2/inmap1.7.2linux-amd64", _inmap_exe)
    else:
        raise(OSError("invalid operating system %s"%(ost)))
    os.chmod(_inmap_exe, stat.S_IXUSR|stat.S_IRUSR|stat.S_IWUSR)







def run_sr(emis, model, output_variables, emis_units="tons/year"):
    """
    Run the provided emissions through the specified SR matrix, calculating the
    specified output properties.

    Args:
        emis: The emissions to be calculated, Needs to be a geopandas dataframe.

        model: The SR matrix to use. Allowed values:
            isrm: The InMAP SR matrix
            apsca_q0: The APSCA SR matrix, annual average
            apsca_q1: The APSCA SR matrix, Jan-Mar season
            apsca_q2: The APSCA SR matrix, Apr-Jun season
            apsca_q3: The APSCA SR matrix, Jul-Sep season
            apsca_q4: The APSCA SR matrix, Oct-Dec season

        output_variables: Output variables to be calculated. See
            https://inmap.run/docs/results/ for more information.

        emis_units: The units that the emissions are in. Allowed values:
            'tons/year', 'kg/year', 'ug/s', and 'μg/s'.
    """


    global _tmpdir
    global _inmap_exe

    model_paths = {
        #"isrm": "/data/isrmv121/isrm_v1.2.1.ncf",
        "isrm": "/Users/norahennessy/Desktop/Stanford Research/IMSR/isrm_v1.2.1.ncf",
        "apsca_q0": "/data/apsca/apsca_sr_Q0_v1.2.1.ncf",
        "apsca_q1": "/data/apsca/apsca_sr_Q1_v1.2.1.ncf",
        "apsca_q2": "/data/apsca/apsca_sr_Q2_v1.2.1.ncf",
        "apsca_q3": "/data/apsca/apsca_sr_Q3_v1.2.1.ncf",
        "apsca_q4": "/data/apsca/apsca_sr_Q4_v1.2.1.ncf",
    }
    if model not in model_paths.keys():
        models = ', '.join("{!s}".format(k) for (k) in model_paths.keys())
        msg = 'model must be one of \{{!s}\}, but is `{!s}`'.format(models, model)
        raise ValueError(msg)
    model_path = model_paths[model]

    start = time.time()
    job_name = "run_aqm_%s"%start
    emis_file = os.path.join(_tmpdir.name, "%s.shp"%(job_name))
    emis.to_file(emis_file)
    out_file = os.path.join(_tmpdir.name, "%s_out.shp"%(job_name))
    
    subprocess.check_output([_inmap_exe, "srpredict",
            "--EmissionUnits=%s"%emis_units,
            "--EmissionsShapefiles=%s"%emis_file,
            "--OutputFile=%s"%out_file,
            "--OutputVariables=%s"%json.dumps(output_variables),
            "--SR.OutputFile=%s"%model_path])
    output = gpd.read_file(out_file)
    os.remove(out_file)
    return output




BAs = gpd.read_file("Control_Areas.shp")
BA_codes = pd.read_csv("ba_names.csv")
BA_codes["NAME"]=BA_codes["BANAME"].str.upper()
BAs=BAs.merge(BA_codes,how="left",left_on="NAME",right_on = "NAME")



with open("pm25_twh_consumption_factors",'rb') as f:
    BA_pm25_twh = pickle.load(f)
with open("deathsK_twh_consumption_factors",'rb') as f:
    BA_deathsK_twh = pickle.load(f)  





with open("pm25_twh_consumption_factors_RPS",'rb') as f:
    BA_pm25_twh_RPS = pickle.load(f)
with open("deathsK_twh_consumption_factors_RPS",'rb') as f:
    BA_deathsK_twh_RPS = pickle.load(f)  




with open("pm25_twh_consumption_factors_NREL_mid",'rb') as f:
    BA_pm25_twh_NREL_mid = pickle.load(f)
with open("deathsK_twh_consumption_factors_NREL_mid",'rb') as f:
    BA_deathsK_twh_NREL_mid = pickle.load(f)  



with open("pm25_twh_consumption_factors_NREL_95_2035",'rb') as f:
    BA_pm25_twh_NREL_95_2035 = pickle.load(f)
with open("deathsK_twh_consumption_factors_NREL_95_2035",'rb') as f:
    BA_deathsK_twh_NREL_95_2035 = pickle.load(f)  


with open("pm25_twh_consumption_factors_no_coal",'rb') as f:
    BA_pm25_twh_no_coal = pickle.load(f)
with open("deathsK_twh_consumption_factors_no_coal",'rb') as f:
    BA_deathsK_twh_no_coal = pickle.load(f)  


with open("pm25_twh_consumption_factors_1980_coal",'rb') as f:
    BA_pm25_twh_1980_coal = pickle.load(f)
with open("deathsK_twh_consumption_factors_1980_coal",'rb') as f:
    BA_deathsK_twh_1980_coal = pickle.load(f) 


with open("pm25_twh_consumption_factors_1972_coal",'rb') as f:
    BA_pm25_twh_1972_coal = pickle.load(f)
with open("deathsK_twh_consumption_factors_1972_coal",'rb') as f:
    BA_deathsK_twh_1972_coal = pickle.load(f) 




co2_twh = pd.read_csv("co2_twh_EFs.csv").set_index("Unnamed: 0")
co2_twh_NREL_mid = pd.read_csv("co2_twh_EFs_NREL_mid.csv").set_index("Unnamed: 0")
co2_twh_NREL_95_2035 = pd.read_csv("co2_twh_EFs_NREL_95_2035.csv").set_index("Unnamed: 0")
co2_twh_1980_coal = pd.read_csv("co2_twh_EFs_1980_coal.csv").set_index("Unnamed: 0")
co2_twh_no_coal = pd.read_csv("co2_twh_EFs_no_coal.csv").set_index("Unnamed: 0")




emfac_efs = pd.read_csv("EMFAC_EFs.csv").drop(columns = ["Unnamed: 0"]).set_index("Model Year")

emfac_efs_new = emfac_efs[emfac_efs.index==2020]

emfac_efs_new["weight"] = 1



def get_diesel_efs(n, efs): #n_trucks = 13.479e6
    truck_fleet = choice(efs.index, n, p = efs.weight)
    #n_trucks = 13.479e6 #for now estimating this from Transportation Energy Book. Try to improve the accuracy of this number. 
    year_cts = Counter(truck_fleet)

    nox_ef = np.average(efs.loc[year_cts,"NOx_g/gal"], weights = pd.Series(year_cts))
    so2_ef = np.average(efs.loc[year_cts,"SOx_g/gal"], weights = pd.Series(year_cts))
    pm25_ef = np.average(efs.loc[year_cts,"PM2.5_g/gal"], weights = pd.Series(year_cts))
    voc_ef = np.average(efs.loc[year_cts,"ROG_g/gal"], weights = pd.Series(year_cts))
    nh3_ef = np.average(efs.loc[year_cts,"NH3_g/gal"], weights = pd.Series(year_cts))
    co2_ef = np.average(efs.loc[year_cts,"CO2_g/gal"], weights = pd.Series(year_cts))
    n2o_ef = np.average(efs.loc[year_cts,"N2O_g/gal"], weights = pd.Series(year_cts))
    ch4_ef = np.average(efs.loc[year_cts,"CH4_g/gal"], weights = pd.Series(year_cts))
    
    EF_gal = pd.DataFrame({"nox":nox_ef, "so2":so2_ef, "pm25":pm25_ef,"voc":voc_ef, "nh3":nh3_ef, "co2":co2_ef, "n2o":n2o_ef, "ch4":ch4_ef}, index = [0])
    return EF_gal



demographics = gpd.read_file("inmap_demographics.shp")

dem_colnames = ['ID', 'White_pct', 'Black_pct', 'Native_pct', 'Asian_pct',
       'Pac Islander_pct', 'Two or More_pct', 'Latino_pct',
       'White non-Latino_pct', 'Black non-Latino_pct', 'Asian non-Latino_pct',
       '<10,000', '10,000-14,999', '15,000 - 24,999', '25,000 - 34,999',
       '35,000 - 49,999', '50,000 - 74,999', '75,000 - 99,999',
       '100,000 - 149,999', '150,000 - 199,999', '>200,000',
       'Total Households', 'Median Income', 'index_right', 'REGION',
       'DIVISION', 'STATEFP', 'STATENS', 'GEOID', 'STUSPS', 'NAME', 'LSAD',
       'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER', 'INTPTLAT', 'INTPTLON','geometry']



demographics.columns = dem_colnames

demographics["Region"] = ""
demographics.loc[demographics.DIVISION=='1',"Region"] = "New England"
demographics.loc[demographics.DIVISION=='2',"Region"] = "Middle Atlantic"
demographics.loc[demographics.DIVISION=='3',"Region"] = "East North Central"
demographics.loc[demographics.DIVISION=='4',"Region"] = "West North Central"
demographics.loc[demographics.DIVISION=='5',"Region"] = "South Atlantic"
demographics.loc[demographics.DIVISION=='6',"Region"] = "East South Central"
demographics.loc[demographics.DIVISION=='7',"Region"] = "West South Central"
demographics.loc[demographics.DIVISION=='8',"Region"] = "Mountain"
demographics.loc[demographics.DIVISION=='9',"Region"] = "Pacific"

demographics["Big Region"] = ""
demographics.loc[demographics.DIVISION=='1',"Big Region"] = "Northeast"
demographics.loc[demographics.DIVISION=='2',"Big Region"] = "Northeast"
demographics.loc[demographics.DIVISION=='3',"Big Region"] = "Midwest"
demographics.loc[demographics.DIVISION=='4',"Big Region"] = "Midwest"
demographics.loc[demographics.DIVISION=='5',"Big Region"] = "South"
demographics.loc[demographics.DIVISION=='6',"Big Region"] = "South"
demographics.loc[demographics.DIVISION=='7',"Big Region"] = "South"
demographics.loc[demographics.DIVISION=='8',"Big Region"] = "West"
demographics.loc[demographics.DIVISION=='9',"Big Region"] = "West"

# ## Functions


def vehicle_weight(df, trucks, vehicle, year, commflow, truckflow):
    df[f"lbs_freight_per_truck_{year}"] = df[commflow]*1e3/df[truckflow]/365*2000 #lbs/truck
    df[f"tot_wt_{vehicle}"] = df[f"lbs_freight_per_truck_{year}"]+trucks.loc["base_weight",vehicle]
    
    df.loc[df[f"tot_wt_{vehicle}"] > 80000,truckflow] = np.ceil(df.loc[df[f"tot_wt_{vehicle}"]>80000,commflow]*1e3*2000/365/(80000-trucks.loc["base_weight",vehicle]))
    #recalculate truck weights
    df[f"lbs_freight_per_truck_{year}"] = df[commflow]*1e3/df[truckflow]/365*2000 #lbs/truck
    df[f"tot_wt_{vehicle}"] = df[f"lbs_freight_per_truck_{year}"]+trucks.loc["base_weight",vehicle]

def truck_energy_consumption(df, trucks, vehicle,year, truckflow):
    #print(f"year = {year}")
    # df.ROAD_GRADE = 0
    # df.TIME_UPHILL = 0
    ## Calculate energy consumption in each segment
    vmax = df["AB_FinalSp"]/0.67/2.23694 #m/s --> average speed/0.67 = max speed (based on CARB HDDT drive cycle used in Fan's paper)
    v = vmax*0.67 #m/s
    v_rms  = vmax* 0.77 #m/s
    a = 0.112 #m^2/s
    weight = df[f"tot_wt_{vehicle}"] * 0.453592 #kg
    d = 1609.34 #meters --> 1 mile driving range
    eff_bW_elec = 0.85
    eff_bW_diesel = 0.42 * 0.9
    eff_brk = 0.97
    P_access = (1200+2300+300+1000)/1000 #kW of accessory power
    A_Cd = trucks.loc["A_Cd",vehicle]
    C_rr = trucks.loc["C_rr",vehicle]
    rou = 1.2 #kg/m^3
    g = 9.8 #m/s^2


    #energy per mile #joules/mile
    if vehicle == "Diesel":
        energy_per_mile = ((0.5*rou * A_Cd*v_rms**3 +
                  C_rr * weight * g * v +
                  df.TIME_UPHILL* weight * g * v * df.ROAD_GRADE)/eff_bW_diesel +
                 0.5 * weight * v * a * (1/eff_bW_diesel - eff_bW_diesel * eff_brk) +
                 P_access * 1e3) * d/v 
        
    elif vehicle == "Electric":
        energy_per_mile = ((0.5*rou * A_Cd*v_rms**3 +
                  C_rr * weight * g * v +
                  df.TIME_UPHILL* weight * g * v * df.ROAD_GRADE)/eff_bW_elec +
                 0.5 * weight * v * a * (1/eff_bW_elec - eff_bW_elec * eff_brk) +
                 P_access * 1e3) * d/v 

    #convert to kwh/mile
    energy_per_mile = energy_per_mile/3.6e6
    df[f"kwh_mile_{vehicle}"] = energy_per_mile
    df[f"kwh_{vehicle}_{year}"] = energy_per_mile * df["LENGTH"] * df[truckflow]*365
    df[f"kwh_per_truck_{vehicle}"] = energy_per_mile * df["LENGTH"]
    if vehicle == "Diesel":
        df["total_gal_Diesel_year"] =  df[f"kwh_{vehicle}_{year}"]/40.7 #convert kwh to gallons of diesel to use with EFs
        
        avg_mpg = (df["LENGTH"]*df[truckflow]).sum()*365/df["total_gal_Diesel_year"].sum()
        print(f"avg. mpg: {avg_mpg}")
    #print(df[f"kwh_{vehicle}_{year}"].head())



def create_emissions_file(df):
    df = df.drop(columns = "geometry")
    df = df.rename(columns = {"centroid":"geometry"})
    df.loc[:,"height"] = 0.0
    df = df[["SOx","NOx","NH3","VOC","PM2_5","height","geometry"]]
    df = df.fillna(0)
    return df
    


def run_inmap(emissions_file):
    output_variables = {
        'TotalPM25':'PrimaryPM25 + pNH4 + pSO4 + pNO3 + SOA',
        'deathsK':'(exp(log(1.06)/10 * TotalPM25) - 1) * TotalPop * 1.06115917 * MortalityRate / 100000 * 1.036144578',
        'deathsL':'(exp(log(1.14)/10 * TotalPM25) - 1) * TotalPop * 1.06115917 * MortalityRate / 100000 * 1.036144578',
        'Population': 'TotalPop * 1.06115917',
        'Mortality': 'MortalityRate * 1.036144578',
        'deathsK_pc': 'deathsK/Population',
        'deathsL_pc': 'deathsL/Population'
    }
    
    resultsISRM = run_sr(emissions_file, model="isrm", emis_units="kg/year", output_variables=output_variables)
    return resultsISRM


def get_diesel_efs(n, efs): #n_trucks = 13.479e6
    truck_fleet = choice(efs.index, n, p = efs.weight)
    #n_trucks = 13.479e6 #for now estimating this from Transportation Energy Book. Try to improve the accuracy of this number. 
    year_cts = Counter(truck_fleet)

    nox_ef = np.average(efs.loc[year_cts,"NOx_g/gal"], weights = pd.Series(year_cts))
    so2_ef = np.average(efs.loc[year_cts,"SOx_g/gal"], weights = pd.Series(year_cts))
    pm25_ef = np.average(efs.loc[year_cts,"PM2.5_g/gal"], weights = pd.Series(year_cts))
    voc_ef = np.average(efs.loc[year_cts,"ROG_g/gal"], weights = pd.Series(year_cts))
    nh3_ef = np.average(efs.loc[year_cts,"NH3_g/gal"], weights = pd.Series(year_cts))
    co2_ef = np.average(efs.loc[year_cts,"CO2_g/gal"], weights = pd.Series(year_cts))
    n2o_ef = np.average(efs.loc[year_cts,"N2O_g/gal"], weights = pd.Series(year_cts))
    ch4_ef = np.average(efs.loc[year_cts,"CH4_g/gal"], weights = pd.Series(year_cts))
    
    EF_gal = pd.DataFrame({"nox":nox_ef, "so2":so2_ef, "pm25":pm25_ef,"voc":voc_ef, "nh3":nh3_ef, "co2":co2_ef, "n2o":n2o_ef, "ch4":ch4_ef}, index = [0])
    return EF_gal


def diesel_emissions(df, diesel_efs):
    df_ap = df[["total_gal_Diesel_year", "geometry","LENGTH"]].copy()
    df_ap["PM2_5"] = diesel_efs["pm25"].item()*df_ap["total_gal_Diesel_year"]/1000
    df_ap["VOC"] = diesel_efs["voc"].item()*df_ap["total_gal_Diesel_year"]/1000
    df_ap["SOx"] = diesel_efs["so2"].item()*df_ap["total_gal_Diesel_year"]/1000
    df_ap["NOx"] = diesel_efs["nox"].item()*df_ap["total_gal_Diesel_year"]/1000
    df_ap["NH3"] = diesel_efs["nh3"].item()*df_ap["total_gal_Diesel_year"]/1000
    df_ap = df_ap.drop(columns = ["total_gal_Diesel_year","LENGTH"])
    
    return df_ap


def diesel_co2(df, diesel_efs):
    df_ap = df[["total_gal_Diesel_year", "geometry","LENGTH"]].copy()
    df_ap["CO2"] = diesel_efs["co2"].item()*df_ap["total_gal_Diesel_year"]/1000
    
    return df_ap
    
def commodity_emissions_diesel(faf_df,trucks, vehicle, year, commflow, truckflow,var, efs):
    df = faf_df.copy()
    vehicle_weight(df, trucks, vehicle, year, commflow, truckflow)
    truck_energy_consumption(df, trucks, vehicle,year, truckflow)
    #print(df.kwh_Diesel_2017.sum())
    diesel_efs = get_diesel_efs(5,efs) #13479000
    #print(diesel_efs)
    diesel_ap = diesel_emissions(df, diesel_efs)
    co2 = diesel_co2(df, diesel_efs)
    # print(diesel_ap.NOx.sum())
    diesel_ap = diesel_ap.reset_index().drop(columns = "index").to_crs(4326)
    return diesel_ap, co2



def get_ba_electricity(df, BAs, trucks,vehicle, year, commflow, truckflow, var):
    vehicle_weight(df, trucks, vehicle, year, commflow, truckflow)
    truck_energy_consumption(df, trucks, vehicle, year, truckflow)    
    faf_ba = df.sjoin(BAs.to_crs(df.crs), predicate = "within")
    #Account for overlapping BAs to avoid double counting electricity
    count_dups = faf_ba.groupby("OBJECTID_left").count()["ID_left"]
    for index, row in faf_ba.iterrows():
        lookup = row["OBJECTID_left"]
        n = count_dups[lookup]
        faf_ba.loc[faf_ba.OBJECTID_left == lookup,var] = faf_ba.loc[faf_ba.OBJECTID_left == lookup,var]/n

    ba_electricity = faf_ba.groupby("BACODE").agg({var:"sum"})
    #Reassign consumpgion in GRIS to PNM (closest BA)
    if "GRIS" in ba_electricity.index:
        ba_electricity.loc["PNM",var] += ba_electricity.loc["GRIS",var]
        ba_electricity = ba_electricity.drop("GRIS")
    return ba_electricity



def get_elec_deaths_basic(ba_electricity, deaths_twh, var): #BA_deathsK_twh
    BA_deathsK = {}
    for ba in ba_electricity.index:
        # print(ba)
        BA_deathsK[ba] = deaths_twh[ba]
        BA_deathsK[ba]["TotalDeathsK"] = BA_deathsK[ba]["deathsK_twh"]*ba_electricity.loc[ba,var]/1e9
    
    BAs = list(BA_deathsK.keys())
    print("BAs[0]:",BAs[0])
    total_BA_deathsK = BA_deathsK[BAs[0]].copy()
    for ba in BAs[1:]:
        #print(ba)
        total_BA_deathsK["TotalDeathsK"]+=BA_deathsK[ba]["TotalDeathsK"]
    return total_BA_deathsK




def get_elec_co2(ba_electricity, co2_twh, var):
    ba_co2 = pd.DataFrame(index = ba_electricity.index, columns = ["co2"])
    #print(ba_electricity.head())
    for ba in ba_co2.index:
        ba_co2.loc[ba, "co2"] = co2_twh.loc[ba,"CO2_tonnes_twh"]*ba_electricity.loc[ba,var]/1e9
    total_co2 = ba_co2["co2"].sum()
    return total_co2



def run_diesel_analysis(df, trucks, vehicle, year,commflow, truckflow, var, efs, name ):
    ap_test, diesel_co2 = commodity_emissions_diesel(df,trucks,vehicle, year, commflow, truckflow,var, efs)
    ap_test["centroid"] = ap_test["geometry"].to_crs('+proj=cea').centroid.to_crs(4326)
    diesel_ef = create_emissions_file(ap_test)
    diesel_deaths = run_inmap(diesel_ef)
    with open(f"Results/diesel_deaths_{name}_{commflow}",'wb') as f:
        pickle.dump(diesel_deaths, f)
    with open(f"Results/diesel_co2_{name}_{commflow}",'wb') as f:
        pickle.dump(diesel_co2, f)
    return diesel_deaths, diesel_co2



def run_elec_analysis(df, BAs, trucks,vehicle, year, commflow, truckflow, var, deaths_twh, name, co2_twh):
    #print(year)
    ba_elec = get_ba_electricity(df, BAs, trucks,vehicle, year, commflow, truckflow, var)
    elec_deaths = get_elec_deaths_basic(ba_elec, deaths_twh, var)
    elec_deaths["deathsK"] = elec_deaths["TotalDeathsK"]
    elec_co2 = get_elec_co2(ba_elec, co2_twh, var)
    with open(f"Results/elec_deaths_{name}_{commflow}",'wb') as f:
        pickle.dump(elec_deaths, f)
    with open(f"Results/elec_co2_{name}_{commflow}",'wb') as f:
        pickle.dump(elec_co2, f)
    
    return elec_deaths, elec_co2




def net_benefits(diesel_deaths, elec_deaths):
    net_benefit= diesel_deaths[["Mortality","Population","geometry"]].copy()
    net_benefit["net_deathsK"] = elec_deaths["TotalDeathsK"] - diesel_deaths["deathsK"]
    net_benefit["net_deathsK_pc"] = net_benefit["net_deathsK"]/net_benefit["Population"]*100000
    net_benefit["deathsK"] = net_benefit["net_deathsK"]
    return net_benefit
    



def plot_net_benefits(net_benefit, commflow):
    
    f, ax = plt.subplots(1,1, figsize = (15,15))
    max_val = net_benefit.net_deathsK_pc.max()
    min_val = net_benefit.net_deathsK_pc.min()
    lim = max(abs(max_val), abs(min_val))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    net_benefit.plot(column = "net_deathsK_pc", cmap = "bwr", legend = True, ax = ax,vmin = -1*lim, vmax = lim, cax = cax)
    comm = commflow.split("-")
    ax.set_title(f"Net Deaths per Capita from Full Fleet Electrification (compared to new Diesel Fleet): {comm[0]}")
    ax.set_xticks([])
    ax.set_yticks([])   
    
    f.savefig(f"net_benefit_{comm[0]}.png")



def plot_all(diesel_deaths, elec_deaths, net_benefit, commflow):
    diesel_deaths = diesel_deaths.to_crs(elec_deaths.crs)
    net_benefit = net_benefit.to_crs(elec_deaths.crs)
    
    tot_diesel = round(diesel_deaths.deathsK.sum(),0)
    tot_elec = round(elec_deaths.TotalDeathsK.sum(),0)
    tot_net = round(tot_elec - tot_diesel,0)
    elec_deaths["deathsK_pc"] = elec_deaths["TotalDeathsK"]/elec_deaths["Population"]*100000
    diesel_deaths["deathsK_pc"] = diesel_deaths["deathsK"]/diesel_deaths["Population"]*100000
                                     
    max_val = max(diesel_deaths.deathsK_pc.max(),elec_deaths.deathsK_pc.max())
    
    f, ax = plt.subplots(1,3, figsize = (20,4))
    ax = ax.flatten()
    comm= commflow.split("-")
    f.suptitle(f"Deaths/100,000 people for {comm[0]}")
    
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    diesel_deaths[diesel_deaths["deathsK_pc"]>0].plot(column = "deathsK_pc", cmap = "gist_heat_r", ax = ax[0], legend = True, cax = cax, vmax = max_val, vmin = 0)
    ax[0].set_title(f"Diesel: {tot_diesel} deaths")
    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    elec_deaths[elec_deaths["deathsK_pc"]>0].plot(column = "deathsK_pc", cmap = "gist_heat_r", ax = ax[1], legend = True, cax = cax, vmax = max_val, vmin = 0)
    ax[1].set_title(f"Electric: {tot_elec} deaths")
    
    
    max_val = net_benefit.net_deathsK_pc.max()
    min_val = net_benefit.net_deathsK_pc.min()
    lim = max(abs(max_val), abs(min_val))
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    net_benefit.plot(column = "net_deathsK_pc", cmap = "seismic", legend = True, ax = ax[2],vmin = -1*lim, vmax = lim, cax = cax)
    ax[2].set_title(f"Net Deaths (Electric - Diesel): {tot_net} deaths")

    for a in ax:
        a.set_xticks([])
        a.set_yticks([]) 
        
    plt.subplots_adjust(wspace=None, hspace=None)

def attach_demographics(df, demographics):
    df = df.to_crs(4326)
    result = df.merge(demographics, how = "inner", left_index = True, right_index = True)
    for c in ['White_pct', 'Black_pct', 'Native_pct', 'Asian_pct',
       'Pac Islander_pct', 'Two or More_pct', 'Latino_pct',
       'White non-Latino_pct', 'Black non-Latino_pct', 'Asian non-Latino_pct',
       '<10,000', '10,000-14,999', '15,000 - 24,999', '25,000 - 34,999',
       '35,000 - 49,999', '50,000 - 74,999', '75,000 - 99,999',
       '100,000 - 149,999', '150,000 - 199,999', '>200,000']:
        result[f"{c}_deathsK"] = result["deathsK"] * result[c]/100
        result[f"{c}_pop"] = result[c]/100*result["Population"]
        result[f"{c}_deathsK_pc"] = result[f"{c}_deathsK"]/result[f"{c}_pop"]*100000
    result = result.set_geometry("geometry_x", drop = True)
    result = result.drop(columns = ["geometry_y"])
    
    return result



def demographics_deaths_table(df): #df should be a dataframe with deaths and demographics
    dem_deaths = pd.DataFrame(columns = ["deaths","pop","deaths_pc","pop_pct"], index = ['White_pct', 'Black_pct', 'Native_pct', 'Asian_pct',
           'Pac Islander_pct', 'Two or More_pct', 'Latino_pct',
           'White non-Latino_pct', 'Black non-Latino_pct', 'Asian non-Latino_pct',
           '<10,000', '10,000-14,999', '15,000 - 24,999', '25,000 - 34,999',
           '35,000 - 49,999', '50,000 - 74,999', '75,000 - 99,999',
           '100,000 - 149,999', '150,000 - 199,999', '>200,000'])
    for row in dem_deaths.index:
        dem_deaths.loc[row,"deaths"] = df[f"{row}_deathsK"].sum()
        dem_deaths.loc[row,"pop"] = df[f"{row}_pop"].sum()
        dem_deaths.loc[row,"deaths_pc"] = dem_deaths.loc[row,"deaths"]/dem_deaths.loc[row,"pop"]*100000
        dem_deaths.loc[row,"pop_pct"] = dem_deaths.loc[row,"pop"]/df["Population"].sum()*100

    totals = pd.Series({"deaths":df.deathsK.sum(), 
              "pop":df.Population.sum(),
              "deaths_pc":df.deathsK.sum()/df.Population.sum()*100000, "pop_pct":100})
    totals.name = "Totals"
    dem_deaths = dem_deaths.append(totals)
    dem_deaths["deaths_pct"] = dem_deaths["deaths"]/dem_deaths.loc["Totals","deaths"]*100
    
    return dem_deaths


def make_big_region_death_tables(df): #df should be a dataframe with deaths and demographics
    big_regional_deaths = pd.DataFrame(columns = demographics["Big Region"].unique(), index = ['White_pct', 'Black_pct', 'Native_pct', 'Asian_pct',
       'Pac Islander_pct', 'Two or More_pct', 'Latino_pct',
       'White non-Latino_pct', 'Black non-Latino_pct', 'Asian non-Latino_pct',
       '<10,000', '10,000-14,999', '15,000 - 24,999', '25,000 - 34,999',
       '35,000 - 49,999', '50,000 - 74,999', '75,000 - 99,999',
       '100,000 - 149,999', '150,000 - 199,999', '>200,000'])
    for region in big_regional_deaths.columns:
        for dem in big_regional_deaths.index:
            big_regional_deaths.loc[dem, region] = df.loc[df["Big Region"]==region,f"{dem}_deathsK"].sum()
    totals = pd.Series({region: df.loc[df["Big Region"]==region,"deathsK"].sum() for region in big_regional_deaths.columns})
    totals.name = "Totals"
    big_regional_deaths = big_regional_deaths.append(totals)


    big_regional_deaths_pc = pd.DataFrame(columns = demographics["Big Region"].unique(), index = ['White_pct', 'Black_pct', 'Native_pct', 'Asian_pct',
       'Pac Islander_pct', 'Two or More_pct', 'Latino_pct',
       'White non-Latino_pct', 'Black non-Latino_pct', 'Asian non-Latino_pct',
       '<10,000', '10,000-14,999', '15,000 - 24,999', '25,000 - 34,999',
       '35,000 - 49,999', '50,000 - 74,999', '75,000 - 99,999',
       '100,000 - 149,999', '150,000 - 199,999', '>200,000'])
    for region in big_regional_deaths_pc.columns:
        for dem in big_regional_deaths_pc.index:
            big_regional_deaths_pc.loc[dem, region] = df.loc[df["Big Region"]==region,f"{dem}_deathsK"].sum()/df.loc[df["Big Region"]==region,f"{dem}_pop"].sum()*100000
    totals = pd.Series({region: df.loc[df["Big Region"]==region,"deathsK"].sum()/df.loc[df["Big Region"]==region,"Population"].sum()*100000 for region in big_regional_deaths_pc.columns})
    totals.name = "Totals"
    big_regional_deaths_pc = big_regional_deaths_pc.append(totals)
    
    return big_regional_deaths, big_regional_deaths_pc


def make_region_death_tables(df): #df should be a dataframe with deaths and demographics
    big_regional_deaths = pd.DataFrame(columns = demographics["Region"].unique(), index = ['White_pct', 'Black_pct', 'Native_pct', 'Asian_pct',
       'Pac Islander_pct', 'Two or More_pct', 'Latino_pct',
       'White non-Latino_pct', 'Black non-Latino_pct', 'Asian non-Latino_pct',
       '<10,000', '10,000-14,999', '15,000 - 24,999', '25,000 - 34,999',
       '35,000 - 49,999', '50,000 - 74,999', '75,000 - 99,999',
       '100,000 - 149,999', '150,000 - 199,999', '>200,000'])
    for region in big_regional_deaths.columns:
        for dem in big_regional_deaths.index:
            big_regional_deaths.loc[dem, region] = df.loc[df["Region"]==region,f"{dem}_deathsK"].sum()
    totals = pd.Series({region: df.loc[df["Region"]==region,"deathsK"].sum() for region in big_regional_deaths.columns})
    totals.name = "Totals"
    big_regional_deaths = big_regional_deaths.append(totals)


    big_regional_deaths_pc = pd.DataFrame(columns = demographics["Region"].unique(), index = ['White_pct', 'Black_pct', 'Native_pct', 'Asian_pct',
       'Pac Islander_pct', 'Two or More_pct', 'Latino_pct',
       'White non-Latino_pct', 'Black non-Latino_pct', 'Asian non-Latino_pct',
       '<10,000', '10,000-14,999', '15,000 - 24,999', '25,000 - 34,999',
       '35,000 - 49,999', '50,000 - 74,999', '75,000 - 99,999',
       '100,000 - 149,999', '150,000 - 199,999', '>200,000'])
    for region in big_regional_deaths_pc.columns:
        for dem in big_regional_deaths_pc.index:
            big_regional_deaths_pc.loc[dem, region] = df.loc[df["Region"]==region,f"{dem}_deathsK"].sum()/df.loc[df["Region"]==region,f"{dem}_pop"].sum()*100000
    totals = pd.Series({region: df.loc[df["Region"]==region,"deathsK"].sum()/df.loc[df["Region"]==region,"Population"].sum()*100000 for region in big_regional_deaths_pc.columns})
    totals.name = "Totals"
    big_regional_deaths_pc = big_regional_deaths_pc.append(totals)
    
    return big_regional_deaths, big_regional_deaths_pc


def make_state_death_tables(df):
    state_deaths_pc = pd.DataFrame(columns = demographics["STATEFP"].unique(), index = ['White_pct', 'Black_pct', 'Native_pct', 'Asian_pct',
       'Pac Islander_pct', 'Two or More_pct', 'Latino_pct',
       'White non-Latino_pct', 'Black non-Latino_pct', 'Asian non-Latino_pct',
       '<10,000', '10,000-14,999', '15,000 - 24,999', '25,000 - 34,999',
       '35,000 - 49,999', '50,000 - 74,999', '75,000 - 99,999',
       '100,000 - 149,999', '150,000 - 199,999', '>200,000'])
    for state in state_deaths_pc.columns:
        for dem in state_deaths_pc.index:
            state_deaths_pc.loc[dem, state] = df.loc[df["STATEFP"]==state,f"{dem}_deathsK"].sum()/df.loc[df["STATEFP"]==state,f"{dem}_pop"].sum()*100000
    totals = pd.Series({state: df.loc[df["STATEFP"]==state,"deathsK"].sum()/df.loc[df["STATEFP"]==state,"Population"].sum()*100000 for state in state_deaths_pc.columns})
    totals.name = "Totals"
    state_deaths_pc = state_deaths_pc.append(totals)
    
    
    state_deaths = pd.DataFrame(columns = demographics["STATEFP"].unique(), index = ['White_pct', 'Black_pct', 'Native_pct', 'Asian_pct',
       'Pac Islander_pct', 'Two or More_pct', 'Latino_pct',
       'White non-Latino_pct', 'Black non-Latino_pct', 'Asian non-Latino_pct',
       '<10,000', '10,000-14,999', '15,000 - 24,999', '25,000 - 34,999',
       '35,000 - 49,999', '50,000 - 74,999', '75,000 - 99,999',
       '100,000 - 149,999', '150,000 - 199,999', '>200,000'])
    for state in state_deaths.columns:
        for dem in state_deaths.index:
            state_deaths.loc[dem, state] = df.loc[df["STATEFP"]==state,f"{dem}_deathsK"].sum()
    totals = pd.Series({state: df.loc[df["STATEFP"]==state,"deathsK"].sum() for state in state_deaths.columns})
    totals.name = "Totals"
    
    return state_deaths_pc, state_deaths


def make_state_deaths_race(state_deaths_c):
    state_deaths_race = state_deaths_c.loc[['White_pct', 'Black_pct', 'Native_pct', 'Asian_pct',
       'Pac Islander_pct', 'Two or More_pct', 'Latino_pct'],:].T[1:]
    return state_deaths_race




def add_highest_race(df):
    df = df.copy()
    # df["highest_race"] = ""
    df["max_deaths_pc"] = df.max(axis = 1)
    for state in df.index:
        # print(state)
        # print(df.loc[state,:])
        if df.loc[state,:].max() == df.loc[state,"White_pct"]:
            df.loc[state,"highest_race"] = "White"
        elif df.loc[state,:].max() == df.loc[state,"Black_pct"]:
            df.loc[state,"highest_race"] = "Black"
        elif df.loc[state,:].max() == df.loc[state,"Native_pct"]:
            df.loc[state,"highest_race"] = "Native"
        elif df.loc[state,:].max() == df.loc[state,"Asian_pct"]:
            df.loc[state,"highest_race"] = "Asian"
        elif df.loc[state,:].max() == df.loc[state,"Pac Islander_pct"]:
            df.loc[state,"highest_race"] = "Pac Islander"
        elif df.loc[state,:].max() == df.loc[state,"Two or More_pct"]:
            df.loc[state,"highest_race"] = "Two or More"
        elif df.loc[state,:].max() == df.loc[state,"Latino_pct"]:
            df.loc[state,"highest_race"] = "Latino"
    
    return df
            



def make_deaths_pc_diff_table_states(df, statefps): #df is the demographic table
    deaths_pc_diff = pd.DataFrame(columns = ['White_pct', 'Black_pct', 'Native_pct', 'Asian_pct',
       'Pac Islander_pct', 'Two or More_pct', 'Latino_pct'], index = statefps, dtype = float)
    deaths_pc_diff_pct = pd.DataFrame(columns = ['White_pct', 'Black_pct', 'Native_pct', 'Asian_pct',
       'Pac Islander_pct', 'Two or More_pct', 'Latino_pct'], index = statefps, dtype = float)
    for state in deaths_pc_diff.index:
        for race in deaths_pc_diff.columns:
            race_deaths_pc = df.loc[df.STATEFP==state,f"{race}_deathsK"].sum()/df.loc[df.STATEFP==state,f"{race}_pop"].sum()*100000
            total_deaths_pc = df.loc[df.STATEFP==state,"deathsK"].sum()/df.loc[df.STATEFP==state,"Population"].sum()*100000
            diff = race_deaths_pc - total_deaths_pc
            diff_pct = diff/total_deaths_pc*100
            deaths_pc_diff.loc[state,race] = diff
            deaths_pc_diff_pct.loc[state,race] = diff_pct
            
    deaths_pc_diff = states.merge(deaths_pc_diff, how = "right", left_on = "STATEFP", right_index = True)
    deaths_pc_diff_pct = states.merge(deaths_pc_diff_pct, how = "right", left_on = "STATEFP", right_index = True)
    return deaths_pc_diff, deaths_pc_diff_pct

def make_net_benefit_regions(summary_table, base_scenario, decarb_scenario, race):

    net_benefit_base = pd.DataFrame(columns = ["commodity","region","deaths","co2","deaths_val","co2_value"])
    for c in summary_table.commodity.unique():
        for r in summary_table.region.unique():
            if r!="All":
                net_deaths = (summary_table.loc[(summary_table.region==r) &
                                                (summary_table.commodity==c) &
                      (summary_table.race==race) &
                      (summary_table.income=="All") &
                      (summary_table.scenario == decarb_scenario),"deaths"].sum() - 
                summary_table.loc[(summary_table.region==r) &
                                                (summary_table.commodity==c) &
                      (summary_table.race==race) &
                      (summary_table.income=="All") &
                      (summary_table.scenario == base_scenario),"deaths"].sum())

                net_deaths_val = (summary_table.loc[(summary_table.region==r) &
                                                (summary_table.commodity==c) &
                      (summary_table.race==race) &
                      (summary_table.income=="All") &
                      (summary_table.scenario == decarb_scenario),"deaths_val"].sum() - 
                summary_table.loc[(summary_table.region==r) &
                                                (summary_table.commodity==c) &
                      (summary_table.race==race) &
                      (summary_table.income=="All") &
                      (summary_table.scenario == base_scenario),"deaths_val"].sum())

                net_co2 = (summary_table.loc[(summary_table.region==r) &
                                                (summary_table.commodity==c) &
                      (summary_table.race==race) &
                      (summary_table.income=="All") &
                      (summary_table.scenario == decarb_scenario),"co2"].sum() - 
                summary_table.loc[(summary_table.region==r) &
                                                (summary_table.commodity==c) &
                      (summary_table.race==race) &
                      (summary_table.income=="All") &
                      (summary_table.scenario == base_scenario),"co2"].sum())

                net_co2_val = (summary_table.loc[(summary_table.region==r) &
                                                (summary_table.commodity==c) &
                      (summary_table.race==race) &
                      (summary_table.income=="All") &
                      (summary_table.scenario == decarb_scenario),"co2_value"].sum() - 
                summary_table.loc[(summary_table.region==r) &
                                                (summary_table.commodity==c) &
                      (summary_table.race==race) &
                      (summary_table.income=="All") &
                      (summary_table.scenario == base_scenario),"co2_value"].sum())
                net_benefit_base = net_benefit_base.append({
                    "commodity":c,
                    "region":r,
                    "deaths": net_deaths,
                    "deaths_val": net_deaths_val,
                    "co2": net_co2,
                    "co2_value": net_co2_val
                }, ignore_index = True)
    return net_benefit_base

