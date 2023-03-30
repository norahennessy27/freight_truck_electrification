This code supports the publication E.M. Hennessy, C.D. Scown and  I.M.L. Azevedo. (2023). *Health, Climate, and Equity Benefits of Freight Truck Electrification.* 

The `Truck_electrification_model.py` file contains data inputs and functions to run the truck electrification model. 

The `Truck_electrification_run_model.ipynb` file contains code to run the model for each commodity, for all flows, and for flows of imported goods, exported goods, and domestic goods.

The `Truck_electrification_plotting.ipynb` file contains code to generate all plots in the paper.

**Data Sources and Inputs:**

*The following data sources are publicly available:*

* FAF5 truck and commodity flows can be downloaded from the FAF website: <https://ops.fhwa.dot.gov/freight/freight_analysis/faf/>. Download the FAF Highway Assignment Results (truck and commodity flows) and the FAF Model Highway Network (road network)

* InMAP Source Receptor Matrix (isrm) can be downloaded from Zenodo: <https://zenodo.org/record/2589760#.ZCW7AuzMLA0>

* Balancing Area shapefiles can be downloaded from Homeland Infrastructure Foundation-level Data: <https://hifld-geoplatform.opendata.arcgis.com/datasets/geoplatform::control-areas/explore>.

* State boundaries (for plotting) can be downloaded from the US Census TIGER/Line website: <https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html>
* US primary road network (for plotting) can be downloaded from the US Census TIGER/Line website: <https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2022&layergroup=Roads>

*Preprocessed input files can be found in the `Data` folder:*

* Conusmption-based PM2.5 concentration, deaths/TWh, and co2 emissions factors (from Hennessy et al. 2022 <https://iopscience.iop.org/article/10.1088/1748-9326/ac6cfa>) for each scenario
*  Diesel age-specific emission factors (extracted from CARB's EMFAC database <https://arb.ca.gov/emfac/fleet-db/28ab505932e5beeed459f365a6966cc2a9419b9e>)
*  Preprocessed demographic data matched to the InMAP Source Receptor Matrix grid cells, originally from the American Community Survey 2019 5-year estimates <https://www.census.gov/programs-surveys/acs/data.html>
*  Commodity values (originally downloaded from FAF5 using their data tabulation tool)
*  ton-miles shipped on interstate highways (calculated from FAF5 data)
*  ton-miles shipped by commodity group (calculated from FAF5 data)
*  Total tons and ton-miles by commodity (downloaded from FAF5 data tabulation tool)




