# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:45:53 2022

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
    
@file description: script for performing the model hyperparameter tuning

    # when calling main():
    # set run_CV to True to run the CV parameter optimiziation
    # set run_comparison_per_loc to True to run the performance evaluation of
        the optimizaed modelfor each location
    
    # if set to False the script will use the saved results from previous runs
            
            1. Load the data
            2. Run the model hyperparameter tuning process
              a. CV max_depth and min_child_weight
              b. subsample and colsample_bytree
              c. learning rate
              d. gamma
            3. Compare the base and optimized models
            4. Evaluate the optimal model for each single location
            5. Plot the results comparison
            

    Input data is loaded from ./input_data/
    Output results as csv are stored in ./results_csv/ and as figures in ./figures/
    
"""

# import libraries
import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.linear_model import Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor

from constants import RANDOM_INT, dependent_var, explanatory_var

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns


# Custom Objective and Evaluation Metric for training the XGBoost model
def xgb_mape(preds, dtrain):
    
   labels = dtrain.get_label()
   
   return('smape', np.mean(np.abs(preds - labels)/((np.abs(labels)+np.abs(preds))/2)))




def evaluate_score(actual, predicted):
    
    SMAPE = np.mean(np.abs(predicted - actual)/((np.abs(actual)+np.abs(predicted))/2))
    
    std = np.std(np.abs(predicted - actual)/((np.abs(actual)+np.abs(predicted))/2))    
    
    return SMAPE, std



def eval_perf(actual, predicted):
    
    rmse = np.sqrt(mean_squared_error(predicted, actual))
    
    if len(actual) > 2:
        R_2 = r2_score(actual, predicted)

    else :
        R_2 = np.nan
        
    cumulative_error_perc = (np.sum(predicted) - np.sum(actual))/np.sum(actual)
    
    return [np.median(actual),  # median_actual
            np.mean(actual),    # mean_actual
            rmse,               # RMSE
            rmse/np.median(actual), # relative RMSE to the median value
            rmse/np.mean(actual),   # relative RMSE to the mean value
            np.mean(np.abs(predicted - actual)/actual),     # MAPE
            np.mean(np.abs(predicted - actual)/((np.abs(actual)+np.abs(predicted))/2)), # SMAPE
            R_2,                    #  R²
            cumulative_error_perc   # cumulative error [%]
            ]



def CV_lasso_train(K, remaining_countries_names, cv_data, alpha):
    
    score_iteration = []
    
    # cross-validation: repeat 5 times changing list of countries used for the validation each time
    for iteration_fold in range(K):
        
        # select 22 countries for validation set, the remaining for training 
        rand_val_list = remaining_countries_names['name'].sample(22, 
                            random_state= RANDOM_INT + iteration_fold)
            
        train_data = cv_data.loc[~cv_data['name'].isin(rand_val_list)][explanatory_var + [dependent_var]].dropna()
        val_data = cv_data.loc[cv_data['name'].isin(rand_val_list)][explanatory_var + [dependent_var]].dropna()
        
        X_train, y_train = train_data[explanatory_var], train_data[dependent_var]
        
        X_val, y_val = val_data[explanatory_var], val_data[dependent_var]
        
        # model design
        model = Lasso(
            alpha=alpha,
            copy_X = True,
            random_state = RANDOM_INT,
            max_iter = 1000,
            normalize=True
            )
        
        # Fit model with coordinate descent
        model.fit(X_train, y_train)
        
        # score the model
        y_pred = model.predict(X_val)
        SMAPE, std = evaluate_score(y_val, y_pred)
        
        score_iteration.append(SMAPE)
    
    return score_iteration


def CV_elastic_net(K, remaining_countries_names, cv_data, alpha, l1_ratio):
    
    score_iteration = []
    
    # cross-validation: repeat 5 times changing list of countries used for the validation each time
    for iteration_fold in range(K):
        
        # select 22 countries for validation set, the remaining for training 
        rand_val_list = remaining_countries_names['name'].sample(22, 
                            random_state= RANDOM_INT + iteration_fold)
            
        train_data = cv_data.loc[~cv_data['name'].isin(rand_val_list)][explanatory_var + [dependent_var]].dropna()
        val_data = cv_data.loc[cv_data['name'].isin(rand_val_list)][explanatory_var + [dependent_var]].dropna()
        
        X_train, y_train = train_data[explanatory_var], train_data[dependent_var]
        
        X_val, y_val = val_data[explanatory_var], val_data[dependent_var]
        
        # model design
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            copy_X = True,
            random_state = RANDOM_INT,
            max_iter = 1000,
            normalize=True
            )
        
        # Fit model with coordinate descent
        model.fit(X_train, y_train)
        
        # score the model
        y_pred = model.predict(X_val)
        SMAPE, std = evaluate_score(y_val, y_pred)
        
        score_iteration.append(SMAPE)
    
    return score_iteration



def CV_randForr_train(K, remaining_countries_names, cv_data,
                      n_estimators=100,
                      min_samples_split=2,
                      min_samples_leaf=1,
                      max_depth=None):
    
    score_iteration = []
    
    # cross-validation: repeat 5 times changing list of countries used for the validation each time
    for iteration_fold in range(K):
        
        # select 22 countries for validation set, the remaining for training 
        rand_val_list = remaining_countries_names['name'].sample(22, 
                            random_state= RANDOM_INT + iteration_fold)
            
        train_data = cv_data.loc[~cv_data['name'].isin(rand_val_list)][explanatory_var + [dependent_var]].dropna()
        val_data = cv_data.loc[cv_data['name'].isin(rand_val_list)][explanatory_var + [dependent_var]].dropna()
        
        X_train, y_train = train_data[explanatory_var], train_data[dependent_var]
        
        X_val, y_val = val_data[explanatory_var], val_data[dependent_var]
        
        # model design
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            #criterion = 'squared_error',
            n_jobs=-1,
            random_state = RANDOM_INT,
            verbose = 0,
            )
        
        # Fit model with coordinate descent
        model.fit(X_train, y_train)
        
        # score the model
        y_pred = model.predict(X_val)
        SMAPE, std = evaluate_score(y_val, y_pred)
        
        score_iteration.append(SMAPE)
    
    return score_iteration

    
    

def CV_xgb_train(K, remaining_countries_names, cv_data, params, boost_round):
    
    score_iteration, round_iteration = [], []
    
    # cross-validation: repeat 5 times changing list of countries used for the validation each time
    for iteration_fold in range(K):
        
        # select 22 countries for validation set, the remaining for training 
        rand_val_list = remaining_countries_names['name'].sample(22, 
                            random_state= RANDOM_INT + iteration_fold)
            
        train_data = cv_data.loc[~cv_data['name'].isin(rand_val_list)]
        val_data = cv_data.loc[cv_data['name'].isin(rand_val_list)]
        
        X_train, y_train = train_data[explanatory_var], train_data[dependent_var]
        
        X_val, y_val = val_data[explanatory_var], val_data[dependent_var]
        
        # DMatrix is an internal data structure that is used by XGBoost
        dtrain = xgb.DMatrix(X_train, y_train)
        dval = xgb.DMatrix(X_val, y_val)
        
        # watchlist for early stopping
        watchlist = [(dtrain,'train'), (dval,'eval')]
        
        # train the model
        model = xgb.train(params, 
                          dtrain, 
                          boost_round, 
                          watchlist, 
                          feval = xgb_mape, 
                          maximize=False, 
                          verbose_eval=False, 
                          early_stopping_rounds=10)
        
        score_iteration.append(model.best_score)
        round_iteration.append(model.best_iteration)
    
    return score_iteration, round_iteration


def main(run_CV=False, run_comparison_per_loc=False):
    
              
    ## 1. Load the data
        
    print('Load the data')
    
    # Access data store
    data_store = pd.HDFStore('./input_data/DEFCOV_dataset_merra.h5') #DEFCOV_dataset
    
    # Retrieve data using the key
    data = data_store['covid_key']
    data_store.close()
    
    
    # preparing the data:
    # retrieve the list of all the single locations under study 
    list_of_locations = data['name'].unique()
    
    # list all the locations with at least 90 days of observations
    data_size_loc = data[['name','daily_cases']].groupby(by='name', 
            as_index=False).count().rename(columns={"daily_cases": "count_daily_cases"})
    
    data_size_loc = data_size_loc.loc[data_size_loc['count_daily_cases'] > 90]
    
    data_temp = data.loc[data['name'].isin(data_size_loc['name'])].reset_index(drop=True)
    
    # select 24 random locations to be used as test set
    # Each location in the filtered dataset has on average 273 data points, thus 
    # resulting in an overall test set size of ~6,500 observations 
    # (about 10% of the starting dataset)
    rand_test_list = data_size_loc['name'].sample(n=24, random_state= RANDOM_INT)
    
    test_data = data_temp.loc[data_temp['name'].isin(rand_test_list)]
    
    # the remaining data is used for the cross-validation process
    cv_data = data_temp.loc[~data_temp['name'].isin(rand_test_list)]
    
    remaining_countries_names = pd.DataFrame(cv_data['name'].unique(), columns=['name'])
    
    # starting XGBoost hyperparameters  # classical ranges
    params={
            'learning_rate' : 0.3,      # 0.01 - 0.3
            "max_depth": 6,             # 3 - 10
            'min_child_weight' : 1,     # 1 - 10
            'gamma' : 0,                # 0 - 0.4
            'subsample' : 1,            # 0.5 - 1
            'colsample_bytree': 1,      # 0.3 - 1  
            'lambda' : 1,
            'alpha' : 0,
            'n_jobs' : 30, # depending on the CPU cores available
        }
       
    boost_round = 100
    

    if run_CV is True:
        
        score_col = ['score_'+str(i) for i in range(5)]
        iter_col = ['iter_'+str(i) for i in range(5)]
        
        '''
    
        ## 2. hyperparameter tuning process - 
        
        # LASSO Regression Model - Gridsearch CV
        
        print('lasso regression - performing the model hyperparameter tuning process')
        
        ###############################################################################
        ### a. Parameter alpha - L1 regularization
        
        # result:
        # best: alpha = 0.075 - main parameter
        
        alpha_cv = pd.DataFrame(
            columns=['alpha', 'mean_smape'] + score_col)
        
        for alpha in [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075] + list(np.arange(0.1, 1.1, 0.1)):
            
            print("  CV with alpha={:1.4f}".format(alpha))
            
            score_iteration = CV_lasso_train(
                5, 
                remaining_countries_names, 
                cv_data, 
                alpha)
                        
            alpha_cv.loc[alpha_cv.shape[0]] = [
                np.round(alpha, 4),
                np.mean(score_iteration)] + score_iteration
            
       
        # ElasticNet Regression Model - Gridsearch CV
        
        print('lasso regression - performing the model hyperparameter tuning process')
        
        ###############################################################################
        ### a. Parameters alpha and l1_ratio - L1 and L2 regularization
        
        # result:
        # best: alpha = 0.001 - main parameter
        # best: l1_ratio = 0.999 - main parameter
        
        alpha_L1Ratio_cv = pd.DataFrame(
            columns=['alpha', 'l1_ratio', 'mean_smape'] + 
            score_col)
                            
        gridsearch_params = [
            (alpha, l1_ratio)
            for alpha in list(np.arange(0.001, 0.003, 0.001))
            for l1_ratio in list(np.arange(0.995, 0.999, 0.001))
            ]
        
        
        for alpha, l1_ratio in gridsearch_params:
            
            print("CV with alpha={}, l1_ratio={}".format(
                          alpha,
                          l1_ratio))
            
            score_iteration = CV_elastic_net(
                5, 
                remaining_countries_names, 
                cv_data, 
                alpha,
                l1_ratio)
            
            alpha_L1Ratio_cv.loc[alpha_L1Ratio_cv.shape[0]] = [
                np.round(alpha, 4), np.round(l1_ratio, 4),
                np.mean(score_iteration)] + score_iteration
            
        
        
        # Randomforrest Regression Model - Gridsearch CV
        
        print('Randomforrest regression - performing the model hyperparameter tuning process')
            

        ###############################################################################
        ### a. Parameters Number of Trees
        
        # result:
        # best: n_estimators = 250 - main parameter
        
        n_estimators_cv = pd.DataFrame(
            columns=['max_features', 'mean_smape'] + score_col)
        
        for n_estimators in [10, 50, 100, 250, 500, 1000]:
            
            print("  CV with n_estimators={}".format(n_estimators))
            
            score_iteration = CV_randForr_train(
                5, 
                remaining_countries_names, 
                cv_data, 
                n_estimators=n_estimators)
                        
            n_estimators_cv.loc[n_estimators_cv.shape[0]] = [
                np.round(n_estimators, 2),
                np.mean(score_iteration)] + score_iteration
            
          
          
        ###############################################################################
        ### b. Parameters Tree Depth
        
        # result:
        # best: max_depth = 40 - main parameter
        
        max_depth_cv = pd.DataFrame(
            columns=['max_depth', 'mean_smape'] + score_col)
        
        for max_depth in [5, 10, 20, 30, 40, 50, None]:
            
            print("  CV with max_depth={}".format(max_depth))
            
            score_iteration = CV_randForr_train(
                5, 
                remaining_countries_names, 
                cv_data, 
                n_estimators=250,
                max_depth=max_depth)
                        
            max_depth_cv.loc[max_depth_cv.shape[0]] = [
                max_depth,
                np.mean(score_iteration)] + score_iteration
            
        
           
        ###############################################################################
        ### c. Parameters min_samples_split 
        
        # result:
        # best: min_samples_split  = 2 
        
        min_samples_split_cv = pd.DataFrame(
            columns=['min_samples_split', 'mean_smape'] + score_col)
        
        for min_samples_split in [2, 10, 50, 100, 1000, 2000, 5000]:
            
            print("  CV with min_samples_split={}".format(min_samples_split))
            
            score_iteration = CV_randForr_train(
                5, 
                remaining_countries_names, 
                cv_data, 
                n_estimators=250,
                max_depth=40,
                min_samples_split=min_samples_split)
                        
            min_samples_split_cv.loc[min_samples_split_cv.shape[0]] = [
                min_samples_split,
                np.mean(score_iteration)] + score_iteration
        
            
        ###############################################################################
        ### f. Parameters min_samples_leaf 
        
        # result:
        # best: min_samples_leaf  = 1
        
        min_samples_leaf_cv = pd.DataFrame(
            columns=['min_samples_leaf', 'mean_smape'] + score_col)
        
        for min_samples_leaf in [1, 5, 10, 50, 100, 1000]:
            
            print("  CV with min_samples_leaf={}".format(min_samples_leaf))
            
            score_iteration = CV_randForr_train(
                5, 
                remaining_countries_names, 
                cv_data, 
                n_estimators=250,
                max_depth=40,
                min_samples_split=50,
                min_samples_leaf=min_samples_leaf,
                max_samples=0.1,
                max_features=1)
                        
            min_samples_leaf_cv.loc[min_samples_leaf_cv.shape[0]] = [
                min_samples_leaf,
                np.mean(score_iteration)] + score_iteration                
            
        
        # XGBOOST - Gridsearch CV

                
        print('xgboost - performing the model hyperparameter tuning process')
        
        ###############################################################################
        ### a. Parameters max_depth and min_child_weight
        
        # result:
        # best: max_depth = 10-12 - main parameter
        # min_child_weight = 7-8 - not relevant
                
        max_depth_min_child_cv = pd.DataFrame(
            columns=['max_depth', 'min_child_weight', 'mean_smape', 'mean_rounds'] + 
            score_col + iter_col)
                            
        gridsearch_params = [
            (max_depth, min_child_weight)
            for max_depth in range(6, 12+1)
            for min_child_weight in range(6, 8+1)
            ]
        
        for max_depth, min_child_weight in gridsearch_params:
            
            print("CV with max_depth={}, min_child_weight={}".format(
                          max_depth,
                          min_child_weight))
            
            # Update parameters
            params['max_depth'] = max_depth
            params['min_child_weight'] = min_child_weight
            
            score_iteration, round_iteration = CV_xgb_train(
                    5,
                    remaining_countries_names, 
                    cv_data, 
                    params, 
                    boost_round)
            
            # save the result
            max_depth_min_child_cv.loc[max_depth_min_child_cv.shape[0]] = [
                    max_depth, 
                    min_child_weight,
                    np.mean(score_iteration),
                    np.mean(round_iteration)] + score_iteration + round_iteration
        
        ###############################################################################
        ### b. Parameters subsample and colsample_bytree
        
        # result: 
        # best: subsample = 1 irrelevant
        # best: colsample_bytree = 1
        
        params['max_depth'] = 10                 
        params['min_child_weight'] = 7          
        
        subColsample_cv = pd.DataFrame(
            columns=['subsample', 'colsample', 'mean_smape', 'mean_rounds'] + 
                score_col + iter_col)
                            
        gridsearch_params = [
            (subsample, colsample)
            for subsample in [i/10. for i in range(5,10+1)]
            for colsample in [i/10. for i in range(7,10+1)]
            ]
        
        for subsample, colsample in gridsearch_params:
            
            print("CV with subsample={}, colsample={}".format(
                          subsample,
                          colsample))
            
            # Update parameters
            params['subsample'] = subsample
            params['colsample_bytree'] = colsample
            
            score_iteration, round_iteration = CV_xgb_train(
                    5,
                    remaining_countries_names, 
                    cv_data, 
                    params, 
                    boost_round)
            
            # save the result   
            subColsample_cv.loc[subColsample_cv.shape[0]] = [
                subsample, 
                colsample, 
                np.mean(score_iteration), 
                np.mean(round_iteration)] + score_iteration + round_iteration
        
        
        ###############################################################################
        ### c. Parameters learning rate
        
        # result:
        # best: learning_rate = 0.1 - 0.2
        
        params['max_depth'] = 10                 
        params['min_child_weight'] = 7    
        params['subsample'] = 1                 
        params['colsample_bytree'] = 1 
        
        eta_cv = pd.DataFrame(
            columns=['learning_rate', 'mean_smape', 'mean_rounds'] + 
                score_col + iter_col)
        
        for eta in [.3, .25, .2, .15, .1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001]:
            
            print("CV with learning_rate={}".format(eta))
            
            # Update parameters
            params['learning_rate'] = eta
            
            score_iteration, round_iteration = CV_xgb_train(
                    5,
                    remaining_countries_names, 
                    cv_data, 
                    params, 
                    boost_round)
                
            eta_cv.loc[eta_cv.shape[0]] = [
                eta,
                np.mean(score_iteration),
                np.mean(round_iteration)] + score_iteration + round_iteration
        
        
        ###############################################################################
        ### d. Parameters gamma
        
        # result:
        # best: gamma = 0.2 consistently among best, but gamma is not a relevant parameter
        
        params['learning_rate'] = 0.1 
        params['max_depth'] = 10                 
        params['min_child_weight'] = 7    
        params['subsample'] = 1                 
        params['colsample_bytree'] = 1 
        
        gamma_cv = pd.DataFrame(
            columns=['gamma','mean_smape', 'mean_rounds'] + score_col + iter_col)
        
        for gamma in np.array(range(0,11))/10: #[0, 0.1, 0.2, 0.3, 0.4]:
            
            print("CV with gamma={}".format(
                          gamma))
            
            # Update parameters
            params['gamma'] = gamma
            
            score_iteration, round_iteration = CV_xgb_train(
                    5,
                    remaining_countries_names, 
                    cv_data, 
                    params, 
                    boost_round)
                
            gamma_cv.loc[gamma_cv.shape[0]] = [
                gamma,
                np.mean(score_iteration),
                np.mean()(round_iteration)] + score_iteration + round_iteration
        
        ###############################################################################
    
        '''
        
    
        
    ## 3. compare the base and optimized models
      
    # setting optimal parameters for xgboost
    params['learning_rate'] = 0.1 
    params['max_depth'] = 10                 
    params['min_child_weight'] = 7    
    params['subsample'] = 1                 
    params['colsample_bytree'] = 1
    params['gamma'] = 0.2
    

    
    # select 22 countries for validation set, the remaining for training 
    rand_val_list = remaining_countries_names['name'].sample(22,
                            random_state = RANDOM_INT)
        
    # prepare the training, validation and test data
    train_data = cv_data.loc[~cv_data['name'].isin(rand_val_list)][explanatory_var + [dependent_var]].dropna()
    val_data = cv_data.loc[cv_data['name'].isin(rand_val_list)]
    
    test_data = test_data[explanatory_var + [dependent_var]].dropna()
    
    X_train, y_train = train_data[explanatory_var], train_data[dependent_var]
    
    X_val, y_val = val_data[explanatory_var], val_data[dependent_var]
    
    X_test, y_test = test_data[explanatory_var], test_data[dependent_var]
    
    # DMatrix is an internal data structure that is used by XGBoost
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    dtest = xgb.DMatrix(X_test, y_test)
    
    # watchlist for early stopping
    watchlist = [(dtrain,'train'), (dval,'eval')]
    
    ## lasso model
    
    # model design
    lasso_model = Lasso(
        alpha=0.075,
        copy_X = True,
        random_state = RANDOM_INT,
        max_iter = 1000,
        normalize=True
        )
    
    # Fit model with coordinate descent
    lasso_model.fit(X_train, y_train)  

    ## elasticNet model
    
    # model design
    elNet_model = ElasticNet(
        alpha=0.001,
        l1_ratio=0.999,
        copy_X = True,
        random_state = RANDOM_INT,
        max_iter = 1000,
        normalize=True
        )
    
    # Fit model 
    elNet_model.fit(X_train, y_train)     
    
    ## RandomForrest regression model
    
    # model design
    randForr_model = RandomForestRegressor(
            n_estimators=250,
            max_depth=40,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state = RANDOM_INT,
            verbose = 0,
            )   
    
    ## xgboost model
    randForr_model.fit(X_train, y_train) 
    
    # train the base xgboost model (default parameters)
    xgb_base_model = xgb.XGBRegressor(n_jobs = 30, 
                                  objective = 'reg:squarederror')
        
    xgb_base_model.fit(X_train, 
                    y_train, 
                    eval_metric=xgb_mape,
                    verbose=False)
        
    # train the optimized xgb model
    xgb_opt_model = xgb.train(params, 
                          dtrain, 
                          boost_round, 
                          watchlist, 
                          feval = xgb_mape, 
                          maximize=False, 
                          verbose_eval=False, 
                          early_stopping_rounds=10)   
    
    # evaluate the model on the train and test sets
    
    # train set
    
    actual = y_train.values                     # actual values
    
    lasso_pred = lasso_model.predict(X_train)            # prediction lasso model
    
    elNet_pred = elNet_model.predict(X_train)            # prediction elasticNet model
    
    randForr_pred = randForr_model.predict(X_train)      # prediction random Forrest model
    
    xgb_pred_base = xgb_base_model.predict(X_train)     # prediction base model
    
    xgb_pred_opt = xgb_opt_model.predict(dtrain)        # prediction optimized model
    
    
    
    lasso_SMAPE_train_base, lasso_std = evaluate_score(actual, lasso_pred)
    
    elNet_SMAPE_train_base, elNet_std = evaluate_score(actual, elNet_pred)
    
    randForr_SMAPE_train_base, randForr_std = evaluate_score(actual, randForr_pred)
    
    xgb_SMAPE_train_base, xgb_std_base = evaluate_score(actual, xgb_pred_base)
        
    xgb_SMAPE_train_opt, xgb_std_opt = evaluate_score(actual, xgb_pred_opt)
    
    print('Lasso model train score: '+str(round(lasso_SMAPE_train_base,3)) +'  '+str(round(lasso_std,3)) )
    print('ElasticNet model train score: '+str(round(elNet_SMAPE_train_base,3)) +'  '+str(round(elNet_std,3)) )
    print('RandomForrest model train score: '+str(round(randForr_SMAPE_train_base,3)) +'  '+str(round(randForr_std,3)) )
    print('Xgboost Base model train score: '+str(round(xgb_SMAPE_train_base,3)) +'  '+str(round(xgb_std_base,3)) )
    print('Xgboost OPT model train score: '+str(round(xgb_SMAPE_train_opt,3)) +'  '+str(round(xgb_std_opt,3)) )
    
    print('')
    
    # test set
    
    actual = y_test.values                      # actual values
    
    lasso_pred = lasso_model.predict(X_test)             # prediction lasso model
    
    elNet_pred = elNet_model.predict(X_test)             # prediction lasso model
    
    randForr_pred = randForr_model.predict(X_test)      # prediction random Forrest model
    
    xgb_pred_base = xgb_base_model.predict(X_test)      # prediction base model
    
    xgb_pred_opt = xgb_opt_model.predict(dtest)         # prediction optimized model
    
    
    
    lasso_SMAPE_test_base, lasso_std = evaluate_score(actual, lasso_pred)
    
    elNet_SMAPE_test_base, elNet_std = evaluate_score(actual, elNet_pred)
    
    randForr_SMAPE_test_base, randForr_std = evaluate_score(actual, randForr_pred)
    
    xgb_SMAPE_test_base, xgb_std_base = evaluate_score(actual, xgb_pred_base)
        
    xgb_SMAPE_test_opt, xgb_std_opt = evaluate_score(actual, xgb_pred_opt)
    
    print('Lasso model test score: '+str(round(lasso_SMAPE_test_base,3)) +'  '+str(round(lasso_std,3)) )
    print('ElasticNet model test score: '+str(round(elNet_SMAPE_test_base,3)) +'  '+str(round(elNet_std,3)) )
    print('RandomForrest model test score: '+str(round(randForr_SMAPE_test_base,3)) +'  '+str(round(randForr_std,3)) )
    print('Xgboost Base model test score: '+str(round(xgb_SMAPE_test_base,3)) +'  '+str(round(xgb_std_base,3)) )
    print('Xgboost OPT model test score: '+str(round(xgb_SMAPE_test_opt,3)) +'  '+str(round(xgb_std_opt,3)) )
    
    
    # Load the data
    # Lasso model train score: 0.457  0.422
    # ElasticNet model train score: 0.602  0.591
    # RandomForrest model train score: 0.119  0.146
    # Xgboost Base model train score: 0.335  0.339
    # Xgboost OPT model train score: 0.284  0.265
    
    # Lasso model test score: 0.401  0.372
    # ElasticNet model test score: 0.533  0.504
    # RandomForrest model test score: 0.314  0.283
    # Xgboost Base model test score: 0.365  0.355
    # Xgboost OPT model test score: 0.288  0.254
    
    
    '''
    
    ## 4. evaluate the optimal model for each single location
    
    if run_comparison_per_loc is True:
    
        print('\nCompare the performance of base and optimal models for each location: ')
        
        results_all = []
        
        results_base = []
        
        results_opt = []
        
        num = 1
        
        for location in list_of_locations:
            
            if (num % 50 == 0):
                print('\t'+str(num)+'/'+str(len(list_of_locations)))
                
            train_data = data.loc[data.name != location]
            test_data = data.loc[data.name == location]
            
            X_train, y_train = train_data[explanatory_var], train_data[dependent_var]
            X_test, y_test = test_data[explanatory_var], test_data[dependent_var]
            
            if X_test.shape[0] > 90:
            
                # model design
                
                base_model = xgb.XGBRegressor(
                    n_jobs = 30, 
                    objective = 'reg:squarederror')
                
                opt_model = xgb.XGBRegressor(
                    learning_rate = 0.1,      # 0.01 - 0.3
                    max_depth = 10,           # 3 - 10
                    min_child_weight = 7,     # 1 - 10
                    gamma = 0.2,              # 0 - 0.4
                    subsample = 1,            # 0.5 - 1
                    colsample_bytree= 1,      # 0.3 - 1  
                    n_jobs = 30,
                    n_estimators  = 30,
                    objective = 'reg:squarederror')
                
                # model training
                
                base_model.fit(X_train, y_train, eval_metric=xgb_mape)
                
                opt_model.fit(X_train, y_train, eval_metric=xgb_mape)
                
                actual = data['daily_cases'].iloc[X_test.index].values
                
                # prediction
                
                pred_base = base_model.predict(X_test)
                
                pred_opt = opt_model.predict(X_test)
                
                # evaluate prediction performance and save the result in the overall dataframe
                
                res_perf_base = eval_perf(actual, pred_base)
                
                results_base.append( [location, X_test.shape[0]] + res_perf_base )
            
        
                res_perf_opt = eval_perf(actual, pred_opt)
                
                results_opt.append( [location, X_test.shape[0]] + res_perf_opt )        
                    
                # temporary store the predictions for the location 
                
                temp_result = pd.DataFrame()
                temp_result['actual'] = actual
                temp_result['predicted'] = pred_opt
                temp_result['SMAPE'] = np.abs(pred_opt - actual)/((np.abs(actual)+np.abs(pred_opt))/2)
                temp_result['loc_name'] = location
                temp_result['test_size'] = X_test.shape[0]
                temp_result['median_actual'] = np.median(actual)
               
                # add the result to the full dataframe
                if num == 1:
                    
                    results_all = temp_result.copy(deep=True)
                    
                else:
                    results_all = results_all.append(temp_result, ignore_index=True)
                
            num = num + 1
                        
        results_base = pd.DataFrame(
            results_base,
            columns=['loc_name','test_size', 'median_actual','mean_actual','rmse',
                'r_rmse_med', 'r_rmse_mean','MAPE','SMAPE','R_2', 'cum_error_perc']) 
        
        results_opt = pd.DataFrame(
            results_opt,
            columns=['loc_name','test_size', 'median_actual','mean_actual','rmse',
                'r_rmse_med', 'r_rmse_mean','MAPE','SMAPE','R_2', 'cum_error_perc'])    
            
            
        print('\tCompleted')
        
        # save the results
        
        results_base.to_csv('./results_csv/results_base_2.csv', index=False)
        
        results_opt.to_csv('./results_csv/results_opt_2.csv', index=False)
        
        results_all.to_csv('./results_csv/results_all_predictions.csv', index=False)
    
        
    else :
        
        results_base = pd.read_csv('./results_csv/results_base.csv') 
        
        results_opt = pd.read_csv('./results_csv/results_opt.csv') 
        
        results_all = pd.read_csv('./results_csv/results_all_predictions.csv') 
            
    
    ## 5. plot the results comparison
    
    # filter locations with less than 3 months worth of observations
    results_base = results_base.loc[results_base['test_size'] > 90]
    results_opt = results_opt.loc[results_opt['test_size'] > 90]
    results_all = results_all.loc[results_all['test_size'] > 90]
    
    #results_base = results_base.loc[results_base.case_study == 4]
    
    # combine the two dataframes
    results_base['model'] = ' Base model'
    results_opt['model'] = ' Optimized model'
    results = results_base.append(results_opt, ignore_index=True)
    
    # Figure 1: Accuracy of the based model (left, blue) compared to the 
    # optimized model (right, orange) on the test set. 
    
    # palette
    pal = "colorblind"
    
    results['SMAPE'] = results['SMAPE'] * 100    
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(x='model', y='SMAPE', data=results, palette='pastel', fliersize=0, ax=ax)
    sns.stripplot(x='model', y='SMAPE', data=results, ax=ax,
                  jitter=True, split=True, linewidth=0.5, palette=pal, alpha=0.4)
    # plt.legend(loc='upper left')
    ax.set_ylim([0,100])
    ax.set_ylabel('SMAPE [%]')
    
    fig.tight_layout()
    
    fig.savefig('./figures/merra_Figure 1.tif', dpi=300)
    
    
    
    # Figure 2: Boxplot of the SMAPE distribution as a function of intervals 
    # of number of COVID-19 daily cases
    
    results_all['SMAPE'] = results_all['SMAPE'] * 100 
    
    # group the results in bins of confirmed daily cases
    
    results_all.loc[(0 <= results_all['actual'])&( results_all['actual'] < 100), 'Confirmed daily cases'] = '0-100'
    results_all.loc[(100 <= results_all['actual'])&( results_all['actual'] < 250), 'Confirmed daily cases'] = '101-250'
    results_all.loc[(250 <= results_all['actual'])&( results_all['actual'] < 500), 'Confirmed daily cases'] = '251-500'
    results_all.loc[(500 <= results_all['actual'])&( results_all['actual'] < 1000), 'Confirmed daily cases'] = '501-1000'
    results_all.loc[(1000 <= results_all['actual'])&( results_all['actual'] < 2500), 'Confirmed daily cases'] = '1001-2500'
    results_all.loc[(2500 <= results_all['actual'])&( results_all['actual'] < 5000), 'Confirmed daily cases'] = '2501-5000'
    results_all.loc[results_all['actual'] > 5000, 'Confirmed daily cases'] = '>5000'
    
    label_order = [
        '0-100',
        '101-250',
        '251-500',
        '501-1000',
        '1001-2500',
        '2501-5000',
        '>5000'
        ]
    
    # create a new category type of data based on 'label_order'
    cat = pd.api.types.CategoricalDtype(ordered= True, categories=label_order)
    
    #Apply this to the specific column in DataFrame
    results_all['Confirmed daily cases'] = results_all['Confirmed daily cases'].astype(cat)
    
    # groupby the results
    dfg = results_all.groupby('Confirmed daily cases')
    
    counts = [len(v) for k, v in dfg]
    total = float(sum(counts))
    
    widths = np.array([c/total for c in counts])
    
    c='black'
    
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    cax = results_all.boxplot('SMAPE', by='Confirmed daily cases', 
                    widths=widths*2, grid=False, ax=ax,
                    showfliers=False, patch_artist=True,
                    boxprops=dict(color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color=c))
       
    i = 0
    # colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink', 'blue' ,'red']
    colors = sns.color_palette('crest')[0:7]
    for label, color in zip(label_order, colors):
        cax.findobj(patches.Patch)[i].set_facecolor(color)
        i = i + 1
    
    ax.set_ylim([-10,130])
    
    ax.set_xticklabels(['%s\n$n$=%d'%(k, len(v)) for k, v in dfg])
    
    ax.set_title('')
    fig.suptitle('')
    ax.set_ylabel('SMAPE [%]')
    fig.tight_layout()
       
    fig.savefig('./figures/merra_Figure 2.tif', dpi=300)
    
    '''
          

if __name__ == "__main__":
    
    # set run_CV to True to run the CV parameter optimiziation
    # set run_comparison_per_loc to True to run the performance evaluation of
    #   the optimizaed modelfor each location
    
    # if set to False the script will use the saved results from previous runs
    
    a = main(run_CV=True, run_comparison_per_loc=False)
 
    
    














