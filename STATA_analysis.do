/*
Created on Mon May 16 09:44:04 2022

@authors: Giovanni Scabbia*, Antonio Sanfilippo*, Annamaria Mazzoni*, 
          Dunia Bachour*, Daniel Perez-Astudillo*, Veronica Bermudez*, 
          Etienne Wey^, Mathilde Marchand-Lasserre^, Laurent Saboret^
          
        * Qatar Environment and Energy Research Institute (QEERI), 
        Hamad Bin Khalifa University (HBKU), Qatar Foundation, 
        P.O. Box 34110, Doha, Qatar
        
        ^ Transvalor S.A, Sophia Antipolis, France
        
        Corresponding authors: asanfilippo@hbku.edu.qa
        
@project-title: Data-driven Estimation and Forecasting of COVID-19 Rates (DEFCOV)

@Title of the study: Does climate help modeling COVID-19 risk and to what extent?
    
@file description: STATA do file for performing the econometric analysis
            
            1. Data preparation
            2. Stationarity test
            3. Panel data Fixed Effects model - Testing the meteo correlation
            4. Panel data Fixed Effects model - Testing the effect of restrictions.
            

    Input data is loaded from ./input_data/
    Output results as doc are stored in ./results_stata/
    
*/

clear 

set more off

set linesize 100
set matsize 10000

*rmsg = return message shows how long the command took to execute
set rmsg off

*Change the working directory
cd "... add link to working directory before running the code ... " 
* example: C:\Users\xxx\Desktop\folder\DEFCOV_folder

* -----------------------------------------------------------------------------

eststo clear

* Load the data
use ".\input_data\DEFCOV_dataset_STATA.dta", clear

* drop missing values
drop if daily_cases >= .

* drop if daily cases < 10
drop if daily_cases < 10

* generate panel variables
egen panel_id = group(name)
egen time_var = group(date)
sort name date

* remove duplicates
quietly by name date:  gen dup = cond(_N==1,0,_n)
drop if dup>1

***********************************
*** 1. data preparation

* log transform the daily_cases variable
gen ln_daily_cases = ln(daily_cases)

* variable to count the number of observations for each location
egen num_obs = count(ln_daily_cases), by (name)

* declare the data in memory to be a panel dataset
tsset panel_id time_var

* create the dummy variables for the analysis of the intervention policies effect
* 60% is the median value

bys name : g stringency_7 = stringencyindex_7[_n-7]
gen high_stringency_7 = 0
replace high_stringency_7 = 1 if stringency_7 > 60

bys name : g stringency_14 = stringencyindex_14[_n-14]
gen high_stringency_14 = 0
replace high_stringency_14 = 1 if stringency_14 > 60

bys name : g containment_7 = containmenthealthindex_7[_n-7]
gen high_containment_7 = 0
replace high_containment_7 = 1 if containment_7 > 60

bys name : g containment_14 = containmenthealthindex_14[_n-14]
gen high_containment_14 = 0
replace high_containment_14 = 1 if containment_14 > 60

***********************************
*** 2. Stationarity test

* to run the test for all the other variables comment/uncomment the selected one

*xtunitroot fisher ln_daily_cases, dfuller drift lags(0)
*xtunitroot fisher ln_daily_cases, dfuller drift lags(1)

*xtunitroot fisher temperature_7, dfuller drift lags(0)
*xtunitroot fisher temperature_7, dfuller drift lags(1)

*xtunitroot fisher absolutehumidity_7, dfuller drift lags(0)
*xtunitroot fisher absolutehumidity_7, dfuller drift lags(1)

*xtunitroot fisher pressure_7, dfuller drift lags(0)
*xtunitroot fisher pressure_7, dfuller drift lags(1)

*xtunitroot fisher windspeed_7, dfuller drift lags(0)
*xtunitroot fisher windspeed_7, dfuller drift lags(1)

*xtunitroot fisher rainfall_7, dfuller drift lags(0)
*xtunitroot fisher rainfall_7, dfuller drift lags(1)

*xtunitroot fisher shortwaveirradiation_7, dfuller drift lags(0)
*xtunitroot fisher shortwaveirradiation_7, dfuller drift lags(1)

*xtunitroot fisher pm2p5_7, dfuller drift lags(0)
*xtunitroot fisher pm2p5_7, dfuller drift lags(1)

*xtunitroot fisher pm10_7, dfuller drift lags(0)
*xtunitroot fisher pm10_7, dfuller drift lags(1)

*xtunitroot fisher uv_7, dfuller drift lags(0)
*xtunitroot fisher uv_7, dfuller drift lags(1)

*xtunitroot fisher high_stringency_7, dfuller drift lags(0)
*xtunitroot fisher high_stringency_14, dfuller drift lags(1)

*xtunitroot fisher high_containment_7, dfuller drift lags(0)
*xtunitroot fisher high_containment_14, dfuller drift lags(1)


***********************************
*** 3. Panel data Fixed Effects model - Testing the meteo correlation

** uncomment to run: keep uncommented either #3 or #4. eststo otherwise will create conflict between the two tables

/*

drop if iso == "LTU"
drop if iso == "GRL"
drop if iso == "TLS"
drop if iso == "VIR"
drop if iso == "REU"

* consider only locations with at least 3-months worth of data
drop if num_obs < 90

* lag 5
gen temperature_T = temperature_5
gen absolutehumidity_T = absolutehumidity_5
gen pressure_T = pressure_5
gen windspeed_T = windspeed_5
gen rainfall_T = rainfall_5
gen shortwaveirradiation_T = shortwaveirradiation_5
gen pm2p5_T = pm2p5_5
gen pm10_T = pm10_5
gen uv_T = uv_5

eststo: qui reg ln_daily_cases days_from_start temperature_T absolutehumidity_T pressure_T windspeed_T rainfall_T shortwaveirradiation_T pm2p5_T pm10_T uv_T i.panel_id , cluster(panel_id)

* lag 7
replace temperature_T = temperature_7
replace absolutehumidity_T = absolutehumidity_7
replace pressure_T = pressure_7
replace windspeed_T = windspeed_7
replace rainfall_T = rainfall_7
replace shortwaveirradiation_T = shortwaveirradiation_7
replace pm2p5_T = pm2p5_7
replace pm10_T = pm10_7
replace uv_T = uv_7

eststo: qui reg ln_daily_cases days_from_start temperature_T absolutehumidity_T pressure_T windspeed_T rainfall_T shortwaveirradiation_T pm2p5_T pm10_T uv_T i.panel_id , cluster(panel_id)

* lag 10
replace temperature_T = temperature_10
replace absolutehumidity_T = absolutehumidity_10
replace pressure_T = pressure_10
replace windspeed_T = windspeed_10
replace rainfall_T = rainfall_10
replace shortwaveirradiation_T = shortwaveirradiation_10
replace pm2p5_T = pm2p5_10
replace pm10_T = pm10_10
replace uv_T = uv_10

eststo: qui reg ln_daily_cases days_from_start temperature_T absolutehumidity_T pressure_T windspeed_T rainfall_T shortwaveirradiation_T pm2p5_T pm10_T uv_T i.panel_id , cluster(panel_id)

* lag 12
replace temperature_T = temperature_12
replace absolutehumidity_T = absolutehumidity_12
replace pressure_T = pressure_12
replace windspeed_T = windspeed_12
replace rainfall_T = rainfall_12
replace shortwaveirradiation_T = shortwaveirradiation_12
replace pm2p5_T = pm2p5_12
replace pm10_T = pm10_12
replace uv_T = uv_12

eststo: qui reg ln_daily_cases days_from_start temperature_T absolutehumidity_T pressure_T windspeed_T rainfall_T shortwaveirradiation_T pm2p5_T pm10_T uv_T i.panel_id , cluster(panel_id)

* lag 114
replace temperature_T = temperature_14
replace absolutehumidity_T = absolutehumidity_14
replace pressure_T = pressure_14
replace windspeed_T = windspeed_14
replace rainfall_T = rainfall_14
replace shortwaveirradiation_T = shortwaveirradiation_14
replace pm2p5_T = pm2p5_14
replace pm10_T = pm10_14
replace uv_T = uv_14

eststo: qui reg ln_daily_cases days_from_start temperature_T absolutehumidity_T pressure_T windspeed_T rainfall_T shortwaveirradiation_T pm2p5_T pm10_T uv_T i.panel_id , cluster(panel_id)

esttab using FE_res.doc, title (Panel data Fixed Effects model) se(2) r2 ar2 label replace rtf b(3) star (* 0.10 ** 0.05 *** 0.01) drop(*.panel_id)

*/


*** 4. Panel data Fixed Effects model - Testing the effect of restrictions. 

** uncomment to run: keep uncommented either #3 or #4. eststo otherwise will create conflict between the two tables

/*

* consider only the admin-level locations in USA and Canada
keep if iso == "USA" || iso == "CAN"
drop if state == ""

drop if num_obs < 90

eststo: qui reg ln_daily_cases days_from_start temperature_7 absolutehumidity_7 pressure_7 windspeed_7 rainfall_7 shortwaveirradiation_7 pm2p5_7 pm10_7 uv_7 i.panel_id , cluster(panel_id)

eststo: qui reg ln_daily_cases days_from_start temperature_7 absolutehumidity_7 pressure_7 windspeed_7 rainfall_7 shortwaveirradiation_7 pm2p5_7 pm10_7 uv_7 high_stringency_7 i.panel_id , cluster(panel_id)

eststo: qui reg ln_daily_cases days_from_start temperature_7 absolutehumidity_7 pressure_7 windspeed_7 rainfall_7 shortwaveirradiation_7 pm2p5_7 pm10_7 uv_7 high_stringency_14 i.panel_id , cluster(panel_id)

eststo: qui reg ln_daily_cases days_from_start temperature_7 absolutehumidity_7 pressure_7 windspeed_7 rainfall_7 shortwaveirradiation_7 pm2p5_7 pm10_7 uv_7 high_containment_7 i.panel_id , cluster(panel_id)

eststo: qui reg ln_daily_cases days_from_start temperature_7 absolutehumidity_7 pressure_7 windspeed_7 rainfall_7 shortwaveirradiation_7 pm2p5_7 pm10_7 uv_7 high_containment_14 i.panel_id , cluster(panel_id)

esttab using FE_policies_res.doc, title (Panel data Fixed Effects model - Testing the effect of restrictions.) se(2) r2 ar2 label replace rtf b(3) star (* 0.10 ** 0.05 *** 0.01) drop(*.panel_id)

*/

