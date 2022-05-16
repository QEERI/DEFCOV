# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:56:06 2022

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
    
@file description: script for performing the statistical analysis (spearmanr and kendalltau)
            
            1. Load the data
            2. Descriptive statistics
            3. Run the statistical analysis
            4. Plot the results
            

    Input data is loaded from ./input_data/
    Output results as csv are stored in ./results_csv/ and as figures in ./figures/
    
"""

# import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from scipy.stats import kendalltau

from constants import dependent_var, climatic_var

def main():
    
    ## 1. Load the data
    
    print('Load the data')
    
    # Access data store
    data_store = pd.HDFStore('./input_data/DEFCOV_dataset_merra.h5')  #DEFCOV_dataset
    
    # Retrieve data using the key
    data = data_store['covid_key']
    data_store.close()
    
    ## 2. descriptive statistics
        
    descriptive_statistics = data.describe(include='all')
    
    descriptive_statistics.T.to_csv('results_csv/descriptive_statistics.csv')
    descriptive_statistics.T.to_excel('results_csv/descriptive_statistics.xlsx')
    
    ## 3. statistical analysis:
        
    print('performing the statistical analysis')
        
    # retrieve the list of all the single locations under study 
    list_of_locations = data['name'].unique()
    
    # dataframe where to store the results
    results_stat = pd.DataFrame(columns =['location','variable','test_size',
                                          'coef_sp','p_sp','coef_ke', 'p_ke'])
    
    i = 0
    
    for location in list_of_locations:
        
        if (i % 50 == 0):
            print('\t processing location: '+str(i)+'/'+str(len(list_of_locations)))
                        
        data_loc = data.loc[data['name'] == location]
        
        for var in climatic_var:
            
            # remove the rows with missing data
            filtered_data_loc = data_loc[[dependent_var, var]].dropna()
    
            # calculate the spearman and kendal_tau coefficient if there are enough 
            # observations (>90 - 3-months worth of data) and good diversity in 
            # the values of the two variables (more than 2 diffent unique values per variable)
            if ((filtered_data_loc.shape[0] > 90) & (                    
                    filtered_data_loc[var].unique().shape[0] > 2) & (
                    filtered_data_loc[dependent_var].unique().shape[0] > 2)):
            
                coef_sp, p_sp = spearmanr(filtered_data_loc[dependent_var], filtered_data_loc[var])
                
                coef_ke, p_ke = kendalltau(filtered_data_loc[dependent_var], filtered_data_loc[var])
                
            else : # otherwise NaN
                
                coef_sp, p_sp, coef_ke, p_ke = np.nan, np.nan, np.nan, np.nan 
            
            # append to the results dataframe
            results_stat.loc[results_stat.shape[0]] = [location,
                                                       var, 
                                                       filtered_data_loc.shape[0],
                                                       coef_sp,p_sp,
                                                       coef_ke,p_ke]
                
        i = i + 1
           
    # save the results     
    results_stat.to_csv('./results_csv/stat_results.csv', index=False)
        
        
    ## 4. Plot the results
    
    print('plotting the results')
    
    # keep only statistically significant results
    
    for coeff in ['coef_sp', 'coef_ke']:
    
        if coeff == 'coef_sp':
            p_val = 'p_sp'
            title = "Spearman's correlation"
            fig_name = 'Figure 4.tif'
        else :
            p_val = 'p_ke'
            title = "Kendall's correlation"
            fig_name = 'Figure 5.tif'
            
        temp_res = results_stat.loc[results_stat[p_val] <= 0.01]
            
        # ascending order of the median value of the coefficients    
        c_order = temp_res[[coeff,'variable']].groupby(by='variable', 
                      as_index=False).median().sort_values(by=coeff, ascending=False)
        c_var_order = c_order['variable']
        c_value_order = c_order[coeff]
        
        # set the color palette      
        custom_palette = []
        for c in c_value_order:
            
            if -0.1 < c < 0.1:
                custom_palette.append((0.95,0.95,0.95))
            elif 0.1 <= c < 0.3:
                custom_palette.append((0.95,0.95,0.95)) #((0.89443865, 0.89721298, 0.9202854)) 
            elif 0.3 <= c < 0.5:
                custom_palette.append((0.66563334, 0.72242871, 0.81414642)) 
            elif c > 0.5:
                custom_palette.append((0.4305964, 0.56276546, 0.74956387)) 
                
            elif -0.3 < c <= -0.1:
                custom_palette.append((0.95,0.95,0.95)) #((0.94742246, 0.87278899, 0.86691076))
            elif -0.5 < c <= -0.3:
                custom_palette.append((0.85164413, 0.65142189, 0.64145983))
            elif c < -0.5:
                custom_palette.append((0.76133542, 0.43410655, 0.42592523))
        
        # set outliers markers 
        flierprops = dict(markersize=5,
              markeredgecolor='none', marker='o')
        
        # plot
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(y=temp_res["variable"], x=temp_res[coeff], order=c_var_order, 
                    palette=custom_palette,flierprops=flierprops, ax=ax)
        
        ax.set_xlabel(title)
        ax.set_ylabel("Explanatory variable")
        
        ax.axvline(x=0, linestyle='--', alpha=0.5) 
        
        fig.tight_layout()
        
        fig.savefig('./figures/'+fig_name, dpi=300)
    
    
if __name__ == "__main__":
    
    main()
    