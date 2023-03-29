## Import libraries
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
%load_ext autoreload
%autoreload 2

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





## Load InMAP files -- see https://inmap.run/blog/2019/04/20/sr/ for details

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




########################### LOAD DATA ####################################

FAF5 = gpd.read_file("FAF5_Data.shp")
FAF5 = FAF5.set_index("ID")

## Attach terrain data
FAF5_terrain = pd.read_csv("FAF5_terrain_match.csv")
FAF5_terrain = FAF5_terrain.set_index("ID")

FAF5["TERRAIN"] = 1
for row in FAF5_terrain.index:
    FAF5.loc[row,"TERRAIN"] = FAF5_terrain.loc[row,"TERRAIN"]
    
## Load imports data
faf_2022_imports = pd.read_csv("FAF5 Import Truck Flows by Commodity_2022.csv")
FAF_db22_i = FAF5.merge(faf_2022_imports, how = "inner", right_on = "ID", left_on = "ID")
FAF_db22_i = FAF_db22_i[~FAF_db22_i.STATE.isin(["AK", "AB","HI", "BC", "YT", "NB", "QC","ON"])]
FAF_db22_i["ROAD_GRADE"] = 0
FAF_db22_i.loc[FAF_db22_i.TERRAIN == 2,"ROAD_GRADE"] = 0.01
FAF_db22_i.loc[FAF_db22_i.TERRAIN == 3,"ROAD_GRADE"] = 0.02

FAF_db22_i["TIME_UPHILL"] = 0
FAF_db22_i.loc[FAF_db22_i.TERRAIN == 2,"TIME_UPHILL"] = 0.3
FAF_db22_i.loc[FAF_db22_i.TERRAIN == 3,"TIME_UPHILL"] = 0.5

## Load exports data
faf_2022_exports = pd.read_csv("FAF5 Export Truck Flows by Commodity_2022.csv")
FAF_db22_e = FAF5.merge(faf_2022_exports, how = "inner", right_on = "ID", left_on = "ID")
FAF_db22_e = FAF_db22_e[~FAF_db22_e.STATE.isin(["AK", "AB","HI", "BC", "YT", "NB", "QC","ON"])]
FAF_db22_e["ROAD_GRADE"] = 0
FAF_db22_e.loc[FAF_db22_e.TERRAIN == 2,"ROAD_GRADE"] = 0.01
FAF_db22_e.loc[FAF_db22_e.TERRAIN == 3,"ROAD_GRADE"] = 0.02

FAF_db22_e["TIME_UPHILL"] = 0
FAF_db22_e.loc[FAF_db22_e.TERRAIN == 2,"TIME_UPHILL"] = 0.3
FAF_db22_e.loc[FAF_db22_e.TERRAIN == 3,"TIME_UPHILL"] = 0.5

## Load domestic flows data
faf_2022_domestic = pd.read_csv("FAF5 Domestic Truck Flows by Commodity_2022.csv")
FAF_db22_d = FAF5.merge(faf_2022_domestic, how = "inner", right_on = "ID", left_on = "ID")
FAF_db22_d = FAF_db22_d[~FAF_db22_d.STATE.isin(["AK", "AB","HI", "BC", "YT", "NB", "QC","ON"])]
FAF_db22_d["ROAD_GRADE"] = 0
FAF_db22_d.loc[FAF_db22_d.TERRAIN == 2,"ROAD_GRADE"] = 0.01
FAF_db22_d.loc[FAF_db22_d.TERRAIN == 3,"ROAD_GRADE"] = 0.02

FAF_db22_d["TIME_UPHILL"] = 0
FAF_db22_d.loc[FAF_db22_d.TERRAIN == 2,"TIME_UPHILL"] = 0.3
FAF_db22_d.loc[FAF_db22_d.TERRAIN == 3,"TIME_UPHILL"] = 0.5

## Load all flows data
FAF_22_all_flows = pd.read_csv("FAF5 Total Truck Flows by Commodity_2022.csv")
FAF_db22_t = FAF5.merge(FAF_22_all_flows, how = "inner", right_on = "ID", left_on = "ID")
FAF_db22_t = FAF_db22_t[~FAF_db22_t.STATE.isin(["AK", "AB","HI", "BC", "YT", "NB", "QC","ON"])]
FAF_db22_t["ROAD_GRADE"] = 0
FAF_db22_t.loc[FAF_db22_t.TERRAIN == 2,"ROAD_GRADE"] = 0.01
FAF_db22_t.loc[FAF_db22_t.TERRAIN == 3,"ROAD_GRADE"] = 0.02

FAF_db22_t["TIME_UPHILL"] = 0
FAF_db22_t.loc[FAF_db22_t.TERRAIN == 2,"TIME_UPHILL"] = 0.3
FAF_db22_t.loc[FAF_db22_t.TERRAIN == 3,"TIME_UPHILL"] = 0.5

## Load electricity data
BAs = gpd.read_file("Control_Areas.shp")
BA_codes = pd.read_csv("ba_tz.csv")
BA_codes["NAME"]=BA_codes["BANAME"].str.upper()
BAs=BAs.merge(BA_codes,how="left",left_on="NAME",right_on = "NAME")

#Consumption-based PM2_5 Concentration factors
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
    
#CO2 Emission factors
co2_twh = pd.read_csv("co2_twh_EFs.csv").set_index("Unnamed: 0")
co2_twh_NREL_mid = pd.read_csv("co2_twh_EFs_NREL_mid.csv").set_index("Unnamed: 0")
co2_twh_NREL_95_2035 = pd.read_csv("co2_twh_EFs_NREL_95_2035.csv").set_index("Unnamed: 0")
co2_twh_1980_coal = pd.read_csv("co2_twh_EFs_1980_coal.csv").set_index("Unnamed: 0")
co2_twh_no_coal = pd.read_csv("co2_twh_EFs_no_coal.csv").set_index("Unnamed: 0")

## Define Truck Characteristics
## Create a dataframe with truck characteristics
trucks = pd.DataFrame(index = ["A_Cd", "C_rr","base_weight", "energy_capacity_kwh"], columns = ["Diesel"])
trucks["Diesel"] = [5.95,6.4725/1000, 19000+13500, 6105]
trucks["Electric"] = [5.95,6.4725/1000, 19000+13500+1000*1000/(250*.64)*2.2, 1000]

#Diesel emission factors
emfac_efs = pd.read_csv("EMFAC_EFs2.csv").drop(columns = ["Unnamed: 0"]).set_index("Model Year")
emfac_efs_new = emfac_efs[emfac_efs.index==2020]
emfac_efs_new["weight"] = 1

#Demographic data
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


############### DEFINE FUNCTIONS ###############################

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
    df["truck_miles"] = df["LENGTH"]*df[truckflow]*365
    
def create_emissions_file(df):
    df = df.drop(columns = "geometry")
    df = df.rename(columns = {"centroid":"geometry"})
    df.loc[:,"height"] = 0.0
    df = df[["SOx","NOx","NH3","VOC","PM2_5","height","geometry"]]
    df = df.fillna(0)
    return df

def create_emissions_file_elec(df):
    df = df.drop(columns = "geometry")
    df = df.rename(columns = {"centroid":"geometry"})
    df.loc[:,"height"] = 0.0
    df = df[["PM2_5","height","geometry"]]
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

def get_diesel_efs(n, efs): 
    nox_ef = np.average(efs["NOx_g/gal"], weights = efs["weight"])
    so2_ef = np.average(efs["SOx_g/gal"], weights = efs["weight"])
    pm25_ef = np.average(efs["PM2.5_g/gal"], weights = efs["weight"])
    voc_ef = np.average(efs["ROG_g/gal"], weights = efs["weight"])
    nh3_ef = np.average(efs["NH3_g/gal"], weights = efs["weight"])
    co2_ef = np.average(efs["CO2_g/gal"], weights = efs["weight"])
    n2o_ef = np.average(efs["N2O_g/gal"], weights = efs["weight"])
    ch4_ef = np.average(efs["CH4_g/gal"], weights = efs["weight"])
    pm25_tbw_ef = np.average(efs["PM2.5TBW_g/mi"], weights = efs["weight"])
    
    EF_gal = pd.DataFrame({"nox":nox_ef, "so2":so2_ef, "pm25":pm25_ef,"voc":voc_ef, "nh3":nh3_ef, "co2":co2_ef, "n2o":n2o_ef, "ch4":ch4_ef, "pm25_tbw":pm25_tbw_ef}, index = [0])
    return EF_gal
def diesel_emissions(df, diesel_efs):
    df_ap = df[["total_gal_Diesel_year", "geometry","LENGTH","truck_miles"]].copy()
    df_ap["PM2_5"] = diesel_efs["pm25"].item()*df_ap["total_gal_Diesel_year"]/1000
    df_ap["VOC"] = diesel_efs["voc"].item()*df_ap["total_gal_Diesel_year"]/1000
    df_ap["SOx"] = diesel_efs["so2"].item()*df_ap["total_gal_Diesel_year"]/1000
    df_ap["NOx"] = diesel_efs["nox"].item()*df_ap["total_gal_Diesel_year"]/1000
    df_ap["NH3"] = diesel_efs["nh3"].item()*df_ap["total_gal_Diesel_year"]/1000
    df_ap["PM2_5"]+=diesel_efs["pm25_tbw"].item()*df_ap["truck_miles"]/1000
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

def get_elec_deaths_basic(df, ba_electricity, pm25_twh, var): #BA_deathsK_twh
    BA_deathsK = {}
    for ba in ba_electricity.index:
        #print(ba)
        BA_deathsK[ba] = pm25_twh[ba]
        #print(BA_deathsK[ba])
        BA_deathsK[ba]["TotalPM25"] = BA_deathsK[ba]["pm25_twh"]*ba_electricity.loc[ba,var]/1e9
    
    
    #tire and brakeware pm2.5
    pm_emis = df[["geometry","truck_miles"]]
    pm_emis["PM2_5"] = pm_emis["truck_miles"]*0.0350193277992643/1000 #kg pm
    pm_emis = pm_emis.drop(columns = ["truck_miles"])
    
    pm_emis["centroid"] = pm_emis["geometry"].to_crs('+proj=cea').centroid.to_crs(4326)
    elec_ef = create_emissions_file_elec(pm_emis)
    elec_deaths = run_inmap(elec_ef)
    
    BAs = list(BA_deathsK.keys())
    #print("BAs[0]:",BAs[0])
    total_BA_deathsK = elec_deaths#BA_deathsK[BAs[0]].copy()
    #total_BA_deathsK["TotalPM25"] = 0
    for ba in BAs[0:]:
        #print(ba)
        total_BA_deathsK["TotalPM25"]+=BA_deathsK[ba]["TotalPM25"]
    total_BA_deathsK["TotalDeathsK"] = ((np.exp(np.log(1.06)/10 * 
                                                total_BA_deathsK["TotalPM25"]) - 1) * 
                                        total_BA_deathsK["Population"] * 
                                        total_BA_deathsK["Mortality"] / 100000 )
    
    return total_BA_deathsK

def get_elec_co2(ba_electricity, co2_twh, var):
    ba_co2 = pd.DataFrame(index = ba_electricity.index, columns = ["co2"])
    #print(ba_electricity.head())
    for ba in ba_co2.index:
        ba_co2.loc[ba, "co2"] = co2_twh.loc[ba,"CO2_tonnes_twh"]*ba_electricity.loc[ba,var]/1e9
    total_co2 = ba_co2["co2"].sum()
    return total_co2

def run_diesel_analysis(df1, trucks, vehicle, year,commflow, truckflow, var, efs, name ):
    df = df1.copy()
    ap_test, diesel_co2 = commodity_emissions_diesel(df,trucks,vehicle, year, commflow, truckflow,var, efs)
    ap_test["centroid"] = ap_test["geometry"].to_crs('+proj=cea').centroid.to_crs(4326)
    diesel_ef = create_emissions_file(ap_test)
    diesel_deaths = run_inmap(diesel_ef)
    with open(f"Results/diesel_deaths_{name}_{commflow}",'wb') as f:
        pickle.dump(diesel_deaths, f)
    with open(f"Results/diesel_co2_{name}_{commflow}",'wb') as f:
        pickle.dump(diesel_co2, f)
    return diesel_deaths, diesel_co2

def run_elec_analysis(df1, BAs, trucks,vehicle, year, commflow, truckflow, var, deaths_twh, name, co2_twh):
    df = df1.copy()
    #print(year)
    ba_elec = get_ba_electricity(df1, BAs, trucks,vehicle, year, commflow, truckflow, var)
    elec_deaths = get_elec_deaths_basic(df1, ba_elec, deaths_twh, var)
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





