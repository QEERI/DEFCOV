clear 

set more off

set linesize 100
set matsize 1000

*rmsg = return message shows how long the command took to execute
set rmsg off

*Change the working directory
cd "C:\Users\gscabbia\OneDrive - Hamad bin Khalifa University\Desktop\QEERI\DEFCOV - QNRF - Covid\DEFCOV_software"

* -----------------------------------------------------------------------------

/*
import delimited using "C:\Users\gscabbia\OneDrive - Hamad bin Khalifa University\Desktop\Covid\data\stata analysis\data_stata\data_stata_2020-08-15", clear

save "data_stata.dta", replace

*/


eststo clear

use ".\input_data\DEFCOV_dataset_merra_STATA.dta", clear

drop if daily_cases >= .
drop if daily_cases < 10

* select only observation when borders were closed
*keep if int_trav_rest == 1

g newdate = date( date, "YMD")
format newdate %td

egen panel_id = group(name)
egen time_var = group(date)
sort name date

* remove duplicates
quietly by name date:  gen dup = cond(_N==1,0,_n)
drop if dup>1

*** data processing


* log transform
gen ln_daily_cases = ln(daily_cases)

*gen ln_pm2p5_7 = ln(pm2p5_7)
*gen ln_pm10_7 = ln(pm10_7)
****

tsset panel_id time_var

***********************************
* Stationarity test

* lag 7
*replace pm2p5 = pm2p5_7
*replace pm10 = pm10_7
*replace temperature = temperature_7
*replace absolutehumidity = absolutehumidity_7
*replace windspeed = windspeed_7
*replace rainfall = rainfall_7

*************************************************

*reg ln_daily_cases days_from_start pm2p5_7 pm10_7 temperature_7 absolutehumidity_7 windspeed_7 rainfall_7 uv_7 i.panel_id, cluster(panel_id)

keep if iso == "USA" || iso == "CAN"
drop if state == ""

bys name : g containment_14 = containmenthealthindex_14[_n-14]
gen high_containment_diff = 0
replace high_containment_diff = 1 if containment_14 > 60

reg ln_daily_cases days_from_start pm2p5_7 pm10_7 temperature_7 absolutehumidity_7 windspeed_7 rainfall_7 uv_7 high_containment_diff i.panel_id, cluster(panel_id)

* 

*xtunitroot fisher ln_daily_cases, dfuller drift lags(0)




