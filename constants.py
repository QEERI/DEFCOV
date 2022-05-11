# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:56:06 2022

@authors: Giovanni Scabbia*, Antonio Sanfilippo*, Annamaria Mazzoni*, 
          Dunia Bachour*, Daniel Perez-Astudillo*, Veronica Bermudez*, 
          Etienne Wey^, Mathilde Marchand-Lasserre^, Laurent Saboret^
          
        * Qatar Environment and Energy Research Institute (QEERI), 
        Hamad Bin Khalifa University (HBKU), Qatar Foundation, 
        P.O. Box 34110, Doha, Qatar
        
        ^ Transvalor S.A, Biot, France
        
        Corresponding authors: asanfilippo@hbku.edu.qa
        
@project-title: Data-driven Estimation and Forecasting of COVID-19 Rates (DEFCOV)

@Title of the study: Does climate help modeling COVID-19 risk and to what extent?
    
@file description: list and declaration of all the constants used in the study
            
   
"""

RANDOM_INT = 123 

dependent_var = 'daily_cases'


epidemiological_var = [
    'days_from_start',
    'daily_cases_3',
    'daily_cases_7']

climatic_var = [
    'Temperature_7',
    'Absolute Humidity_7',
    'Pressure_7',
    'Wind speed_7',
    'Rainfall_7',
    'Short-wave irradiation_7',
    # 'GHI_7',
    # 'UV-B_7',
    # 'UV-A_7',
    'PM2P5_7',
    'PM10_7',
    'UV_7',
    # 'UVindex_7'
    ]

policy_var = [
    'C1_School closing_7',
    'C2_Workplace closing_7',
    'C3_Cancel public events_7',
    'C4_Restrictions on gatherings_7',
    'C5_Close public transport_7',
    'C6_Stay at home requirements_7',
    'C7_Restrictions on internal movement_7',
    'C8_International travel controls_7',
    'H1_Public information campaigns_7',
    'H2_Testing policy_7',
    'H3_Contact tracing_7',
    'H6_Facial Coverings_7',
    # 'H7_Vaccination policy_7',
    'StringencyIndex_7',
    'ContainmentHealthIndex_7'
    ]

control_var = [ #socio-economic control factors
    'GHS_score',
    'prevent_score',
    'detect_score',
    'GDPP',
    'CO2_emission',
    'tot_greenhouse_ktCO2',
    'tot_methane_ktCO2',
    'tot_NOX_ktCO2',
    'PM2.5_year_exposure_mcg/m3',
    'pop_density_WB',
    'mobile_sub_100pp',
    'internet_servers_1Mpp',
    'diabetes_%pp',
    'pop_age_65more',
    'female_pop',
    'urban_pop'
    ]

explanatory_var = epidemiological_var +  policy_var + climatic_var + control_var