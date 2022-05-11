# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:52:13 2022

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
    
@file description: script for performing the SHAP analysis
            
            1. Load the data
            2. Create and train the model
            3. Run SHAP analysis - feature importance
            4. plot the results

    Input data is loaded from ./input_data/
    Output results as figures in ./figures/
    
"""


# import libraries
import pandas as pd
import numpy as np

from constants import RANDOM_INT, dependent_var, explanatory_var

import xgboost as xgb

from sklearn.linear_model import Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

import shap


# Custom Objective and Evaluation Metric for training the XGBoost model
def xgb_mape(preds, dtrain):
    
   labels = dtrain.get_label()
   
   return('smape', np.mean(np.abs(preds - labels)/((np.abs(labels)+np.abs(preds))/2)))


def main():
    
    ## 1. Load the data
        
    print('Load the data')
    
    # Access data store
    data_store = pd.HDFStore('./input_data/DEFCOV_dataset_merra.h5') #DEFCOV_dataset
    
    # Retrieve data using the key
    data = data_store['covid_key']
    data_store.close()
    
    # to select only the northern hemisphere 
    data_north = data.loc[data['lat'].astype(float) > 0 ].reset_index(drop=True)
    
    # to select only the southern hemisphere 
    data_south = data.loc[data['lat'].astype(float) < 0 ].reset_index(drop=True)
    
    
    # data without missing values (for the lasso, elasticNet, and random forest models)
    data_noNA = data[explanatory_var+[dependent_var]].dropna().reset_index(drop=True)
    
    ## 2. Create and train the models
    
    # design the model (using the tuned hyperparameters)
    xgb_model = xgb.XGBRegressor(
        learning_rate = 0.1,      # 0.01 - 0.3
        max_depth = 10,             # 3 - 10
        min_child_weight = 7,     # 1 - 10
        gamma = 0.2,                # 0 - 0.4
        subsample = 1,            # 0.5 - 1
        colsample_bytree= 1,      # 0.3 - 1  
        n_jobs = 30,
        n_estimators  = 30,
        objective = 'reg:squarederror')
    
    xgb_model_north = xgb.XGBRegressor(
        learning_rate = 0.1,      # 0.01 - 0.3
        max_depth = 10,             # 3 - 10
        min_child_weight = 7,     # 1 - 10
        gamma = 0.2,                # 0 - 0.4
        subsample = 1,            # 0.5 - 1
        colsample_bytree= 1,      # 0.3 - 1  
        n_jobs = 30,
        n_estimators  = 30,
        objective = 'reg:squarederror')
    
    xgb_model_south = xgb.XGBRegressor(
        learning_rate = 0.1,      # 0.01 - 0.3
        max_depth = 10,             # 3 - 10
        min_child_weight = 7,     # 1 - 10
        gamma = 0.2,                # 0 - 0.4
        subsample = 1,            # 0.5 - 1
        colsample_bytree= 1,      # 0.3 - 1  
        n_jobs = 30,
        n_estimators  = 30,
        objective = 'reg:squarederror')
    
    # other models
    
    lasso_model = Lasso(
        alpha=0.075,
        copy_X = True,
        random_state = RANDOM_INT,
        max_iter = 1000,
        normalize=True
        )
    
    elNet_model = ElasticNet(
        alpha=0.001,
        l1_ratio=0.999,
        copy_X = True,
        random_state = RANDOM_INT,
        max_iter = 1000,
        normalize=True
        )
    
    randTree_model = RandomForestRegressor(
            n_estimators=250,
            max_depth=40,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state = RANDOM_INT,
            verbose = 0,
            ) 
    
    
    # fit the models
    
    xgb_model_all = xgb_model.fit(data[explanatory_var], data[dependent_var], eval_metric=xgb_mape)
    
    xgb_model_north = xgb_model_north.fit(data_north[explanatory_var], data_north[dependent_var], eval_metric=xgb_mape)
    
    xgb_model_south = xgb_model_south.fit(data_south[explanatory_var], data_south[dependent_var], eval_metric=xgb_mape)
    
    lasso_model.fit(data_noNA[explanatory_var], data_noNA[dependent_var])
    
    elNet_model.fit(data_noNA[explanatory_var], data_noNA[dependent_var])
    
    randTree_model.fit(data_noNA[explanatory_var], data_noNA[dependent_var])
    
    
    ## 3. Run SHAP Analysis - feature importance
    
    shap_values = shap.TreeExplainer(xgb_model_all).shap_values(data[explanatory_var], check_additivity=False)
    
    shap_values_north = shap.TreeExplainer(xgb_model_north).shap_values(data_north[explanatory_var], check_additivity=False)
    
    shap_values_south = shap.TreeExplainer(xgb_model_south).shap_values(data_south[explanatory_var], check_additivity=False)
    
    ## 4. plot the results
    
    # bar plot
    
    fig1, ax1 = plt.subplots()
    
    shap.summary_plot(
        shap_values = shap_values[:,3:], 
        features = data[explanatory_var[3:]], 
        plot_type = "bar", 
        color = 'slategrey',
        use_log_scale = True)
    
    ax1.set_xlabel('mean(|SHAP value|) - log-scale') #average impact on model output magnitude (log-scale)
    fig1.tight_layout()
    
    fig1.savefig('./figures/merra_Figure 6.tif', dpi=300)
    
    # scatter plot
    
    fig2, ax2 = plt.subplots()
    
    # do not include lagged values
    
    shap.summary_plot(
        shap_values = shap_values[:,3:], 
        features = data[explanatory_var[3:]], 
        plot_type = "dot", 
        use_log_scale = False)
    
    ax2.set_xlabel('SHAP value (impact on model output)')
    
    ax2.set_xlim(-175, 175)
    
    fig2.tight_layout()
    
    fig2.savefig('./figures/merra_Figure 7.tif', dpi=300)
    
    
    # other models
    
    # sort by absolute value
    lasso_sort_idx = np.abs(lasso_model.coef_[3:]).argsort()
    
    elNet_sort_idx = np.abs(elNet_model.coef_[3:]).argsort()
    
    randTree_sort_idx = randTree_model.feature_importances_.argsort()
    
    
    fig3, ax3 = plt.subplots(1,3, figsize=(14,8))
    
    # do not include lagged values
    
    ax3[0].barh(np.array(explanatory_var[3:])[lasso_sort_idx][-20:], lasso_model.coef_[3:][lasso_sort_idx][-20:],
               color=(pd.Series(lasso_model.coef_[3:][lasso_sort_idx][-20:]) > 0).apply(lambda x: '#1e88e5' if x else '#f52757')
               )
    #ax[0].set_xscale('log')
    ax3[0].set_xlabel('Regression coefficient')
    ax3[0].set_title('Lasso')
    
    ax3[1].barh(np.array(explanatory_var[3:])[elNet_sort_idx][-20:], elNet_model.coef_[3:][elNet_sort_idx][-20:],
               color=(pd.Series(elNet_model.coef_[3:][elNet_sort_idx][-20:]) > 0).apply(lambda x: '#1e88e5' if x else '#f52757')
               )
    #ax[1].set_xscale('log')
    ax3[1].set_xlabel('Regression coefficient')
    ax3[1].set_title('ElasticNet')
    
    ax3[2].barh(np.array(explanatory_var)[randTree_sort_idx][-20:-2], 
               randTree_model.feature_importances_[randTree_sort_idx][-20:-2],
               color = 'slategrey')
    ax3[2].set_xscale('log')
    ax3[2].set_xlabel('Gini importance score')
    ax3[2].set_title('Random Forest Tree')
    
    fig3.tight_layout()
    
    fig3.savefig('./figures/merra_Figure S2.tif', dpi=300)
    
    # n_estimators = 3 -> 0.163
    # n_estimators = 10 -> 0.374
    # n_estimators = 100 -> 3.62
    # n_estimators = 250 -> 8.92
    
    ## north vs south hemisphere
    
    # Northern hemisphere countries
        
    fig4 = plt.figure(figsize=(10,8))
    
    # do not include lagged values
    ax4_0 = fig4.add_subplot(121)
    
    idx = np.mean(np.abs(shap_values_north[:,3:]), axis=0).argsort()
    
    ax4_0.barh(np.array(explanatory_var[3:])[idx][-20:], 
             np.mean(np.abs(shap_values_north[:,3:]), axis=0)[idx][-20:],
             color = 'slategrey')
    ax4_0.set_xscale('log')
    ax4_0.set_title('Northern hemisphere countries')
    
    ax4_0.set_xlabel('mean(|SHAP value|) - log-scale')
    
    ax4_0.set_xlim(0, 100)
    
    ax4_1 = fig4.add_subplot(122)
    
    idx = np.mean(np.abs(shap_values_south[:,3:]), axis=0).argsort()
    
    ax4_1.barh(np.array(explanatory_var[3:])[idx][-20:], 
             np.mean(np.abs(shap_values_south[:,3:]), axis=0)[idx][-20:],
             color = 'slategrey')
    ax4_1.set_xscale('log')
    ax4_1.set_title('Southern hemisphere countries')
    
    ax4_1.set_xlabel('mean(|SHAP value|) - log-scale')
    
    ax4_1.set_xlim(0, 100)
    
    fig4.tight_layout()
    
    fig4.savefig('./figures/merra_Figure S3.tif', dpi=300)
    
    plt.show()
    
    # southern hemisphere countries
    
    fig5 = plt.figure(figsize=(12,8))
    
    # do not include lagged values
    ax5_0 = fig5.add_subplot(121)
    
    shap.summary_plot(
        shap_values = shap_values_north[:,3:], 
        features = data_north[explanatory_var[3:]], 
        plot_type = "dot", 
        use_log_scale = False,
        show=False,
        plot_size=None)
    
    ax5_0.set_xlabel('SHAP value (impact on model output)')
    ax5_0.set_title('Northern hemisphere countries')
    ax5_0.set_xlim(-100,100)
    
    ax5_1 = fig5.add_subplot(122)
    
    shap.summary_plot(
        shap_values = shap_values_south[:,3:], 
        features = data_south[explanatory_var[3:]], 
        plot_type = "dot", 
        use_log_scale = False,
        show=False,
        plot_size=None)
    
    ax5_1.set_xlabel('SHAP value (impact on model output)')
    ax5_1.set_title('Southern hemisphere countries')
    ax5_1.set_xlim(-100, 100)
    
    fig5.tight_layout()
    
    fig5.savefig('./figures/merra_Figure S4.tif', dpi=300)
    
    plt.show()

    
if __name__ == "__main__":
    
    main()
    
    


