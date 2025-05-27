"""
Created on Tue Jan 4 2022
@author: Sajjad
"""

# %% Importing modules 

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from copy import deepcopy

# %% Global constants

DATA_FILE = 'New_Dataset.csv'
COL_NAMES = ['design','PT','PP','descomp_auto','descomp_man','sim_disc','sim_perc','sim_ssim','sim_manual']
X_COLS = ['PT','PP','descomp_auto']
Y_COL = 'sim_disc'

# models that will be used
LINEARREG = {'model':LinearRegression, 'modelname':LinearRegression.__name__, 'HPO':None}
BAYINITS = [1.0, 1e-3]
BAYRIDGE = {'model':BayesianRidge, 'modelname':BayesianRidge.__name__, 
            'inits':{'alpha_init':BAYINITS[0], 'lambda_init':BAYINITS[1]}, 'HPO':None}
RANDFORST = {'model':RandomForestRegressor, 'modelname':RandomForestRegressor.__name__, 
             'inits':{'criterion':'absolute_error'}, 
             'HPO':{'criterion': ['squared_error', 'absolute_error', 'poisson'],
                    'max_features' : ['auto', 'sqrt', 'log2']}}
DECTREE = {'model':DecisionTreeRegressor, 'modelname':DecisionTreeRegressor.__name__, 
           'inits':{'criterion':'poisson', 'max_features':'auto'},
           'HPO':{'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                  'max_features' : ['auto', 'sqrt', 'log2']}}
SVRMDL = {'model':SVR, 'modelname':SVR.__name__, 'inits':{'C':10, 'degree':2, 'kernel':'sigmoid'},
       'HPO':{'C': [0.1,1, 10, 100],'degree': [2,3,4,5,6],'kernel': ['rbf', 'poly', 'sigmoid']}}
MODELS = [LINEARREG, BAYRIDGE, RANDFORST, DECTREE, SVRMDL]



# %% Necessary functions

def regperf(model, xtrain, xtest, ytrain, ytest, verbose=True):
    """
    regperf = Regression Performance. It evaluates the regression performance of a trained model.

    Parameters
    ----------
    model : machine learning regression model
        The trained model, whatever it is.
    xtrain/xtest : array-like
        Training/Testing set inputs.
    ytrain/ytest : array-like
        Training/Testing set target output.
    verbose : bool, optional
        Whether results should also be printed. The default is True.

    Returns
    -------
    Dictionary with the following keys: 'ypredtrain', 'ypredtest', 'rmse_train', 'rmse_test', 'r2_train', 'r2_test'

    """
    # train fitting
    Y_pred_train = model.predict(xtrain)
    rmse_train = np.sqrt(mean_squared_error(ytrain, Y_pred_train))
    r2_train = r2_score(ytrain, Y_pred_train)
    
    # test fitting
    Y_pred_test = model.predict(xtest)
    rmse_test = np.sqrt(mean_squared_error(ytest, Y_pred_test))
    r2_test = r2_score(ytest, Y_pred_test)
    
    if verbose:
        print("Training performance:")
        print('Root mean squared error (RMSE): %.2f' % rmse_train)
        print('Coefficient of determination (R^2): %.2f' % r2_train)
        print(" ")
        print("Testing performance:")
        print('Root mean squared error (RMSE): %.2f' % rmse_test)
        print('Coefficient of determination (R^2): %.2f' % r2_test)
    
    return {'ypredtrain':Y_pred_train, 'ypredtest':Y_pred_test, 'rmse_train':rmse_train, 'rmse_test':rmse_test,
            'r2_train':r2_train, 'r2_test':r2_test}



def fitmodels(model_list, xtrain, xtest, ytrain, ytest, verbose=True):
    """
    Fit machine learning models with hyperparemters

    Parameters
    ----------
    model_list : list of dicts
        List of dictionaries explaining the models to be trained.
        Every item in this list must be dictionary with the following keys:
            'model': Function handle of the model to be used
            'inits' [mandatory]: dictionary of initial parameters used in the model. Defaults to nothing.
            'HPO' [mandatory]: dictionary of hyperparameters passed to the GridSearchCV hyperparameter optimization 
            function. None means no HPO. Default is None.
    xtrain/xtest : array
        Training/Testing inputs.
    ytrain/ytest : array
        Training/Testing targets.
    verbose : bool, optional
        Verbosity of the output. The default is True.

    Returns
    -------
    model_list_out : list of dicts
        A new list of dictionaries similar to the input, 
        with additinal 'initial_performance' and 'HPO_grid' keys available..

    """
    model_list_out = deepcopy(model_list)
    for idx,modeldict in enumerate(model_list):
        print("\n===================================================================================")
        modelfun = modeldict.get('model')
        if not modelfun:
            print("WARNING in fitmodels(): " + 
                  "Item at index %d of model_list argument: There is no 'model' key present. Item ignored." % idx)
            continue
        print("Training %s ..." % modelfun.__name__)
        inits = modeldict.get('inits')
        model = modelfun() if not inits else modelfun(**inits)
        try:
            model.fit(xtrain, ytrain)
        except Exception as e:
            print("Training failed with the following message:")
            print(e)
            print("Skipping model and continuing ...")
            continue
        perfinit = regperf(model, xtrain, xtest, ytrain, ytest, verbose)
        model_list_out[idx]['initial_performance'] = perfinit
        if modeldict.get('HPO'):
            if verbose: print("\nPerforming hyperparamter optimization for %s ..." % modelfun.__name__)
            grid = GridSearchCV(modelfun(), modeldict.get('HPO'), refit=True, verbose=1 if verbose else 0)
            grid.fit(xtrain, ytrain)
            if verbose: print("Best Estimator:\n", grid.best_estimator_)
            model_list_out[idx]['HPO_grid'] = grid
    return model_list_out
        
        
        

# %% Loading and preprocessing data

dataset = pd.read_csv(DATA_FILE, header=0, names=COL_NAMES)
X = dataset[X_COLS]
y = dataset[Y_COL]
X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)
sc = StandardScaler()
X_train = sc.fit_transform(X_train_raw)
X_test = sc.transform(X_test_raw)

data = X_train, X_test, Y_train, Y_test 

# %% Training models

results = fitmodels(MODELS, *data, verbose=True)

