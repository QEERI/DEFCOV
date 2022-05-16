# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:39:11 2022

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
    
@file description: script for testing the ML model
            
            1. Load the data
            2. Create and train the model, and plot the results for each test location
            

    Input data is loaded from ./input_data/
    Output results as figures in ./figures/
    
"""

# import libraries
import pandas as pd
import numpy as np

from constants import dependent_var, explanatory_var

from sklearn.metrics import r2_score

import xgboost as xgb

import matplotlib.pyplot as plt


# Custom Objective and Evaluation Metric for training the XGBoost model
def xgb_mape(preds, dtrain):
    
   labels = dtrain.get_label()
   
   return('smape', np.mean(np.abs(preds - labels)/((np.abs(labels)+np.abs(preds))/2)))



def main():
             
    ## 1. Load the data
        
    print('Load the data')
    
    # Access data store
    data_store = pd.HDFStore('./input_data/DEFCOV_dataset_merra.h5')
    
    # Retrieve data using the key
    data = data_store['covid_key']
    data_store.close()
    
    ## 2. Create and train the model, and plot the results for each test location
    
    i = 0 
    j = 0
    
    fig, axs = plt.subplots(4,2, figsize=(12,9.5), sharex=True)
    
    for location in [
            'Italy, ',
            'United Kingdom, ',
            'Iran, ',
            'Bangladesh, ',
            'Indonesia, ',
            'South Africa, ',
            'US, New York',
            'US, California',
          ]:
        
        print(location)
        
        if j == 2:
            j = 0
            i = i + 1
        
        # the data of the location is used as test set
        train_data = data.loc[data.name != location]
        test_data = data.loc[data.name == location]
        
        X_train, y_train = train_data[explanatory_var], train_data[dependent_var]
        X_test, y_test = test_data[explanatory_var], test_data[dependent_var]
            
        # design the model (using the tuned hyperparameters)
        model = xgb.XGBRegressor(
            learning_rate = 0.1,      # 0.01 - 0.3
            max_depth = 10,             # 3 - 10
            min_child_weight = 7,     # 1 - 10
            gamma = 0.2,                # 0 - 0.4
            subsample = 1,            # 0.5 - 1
            colsample_bytree= 1,      # 0.3 - 1  
            n_jobs = 30,
            n_estimators  = 30,
            objective = 'reg:squarederror')
        
        # train the model
        model.fit(X_train, y_train, eval_metric=xgb_mape)
        
        # predict
        predicted = model.predict(X_test)
        
        actual = y_test.values
        
        # evaluate the model performance
        sape = np.abs(predicted - actual)/((np.abs(actual)+np.abs(predicted))/2)
        SMAPE = np.mean(sape)*100
        R_2 = r2_score(actual, predicted)
    
        # plot the results
        
        axs[i, j].plot(test_data.date, test_data.daily_cases, label='actual', color='lightblue')
        axs[i, j].plot(test_data.date, model.predict(X_test), label='modeled', color='darkslategray')
        
        if location[-2:] == ', ':
            loc_name = location[:-2]
        else :
            loc_name = location
            
        axs[i, j].set_title(loc_name)
        
        axs[i, j].set_ylabel('Daily cases')
        
        axs[i, j].legend(title='R$^2$: '+'{:.2f}'.format(R_2)+', SMAPE: '+'{:.1f}'.format(SMAPE)+'%')
        
        j = j + 1
        
    fig.tight_layout()
    
    fig.savefig('./figures/Fig 3.tif', dpi=300)



if __name__ == "__main__":
    
    main()