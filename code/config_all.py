#%% general configs
mystate = 123
import random
import os
import time
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#dirs
datadir = "../data/"
outs_dir = os.path.join(datadir,'models_outputs')
modelsdir = "../models/"
dataraw_dir = os.path.join(datadir, 'raw')
dataprocessed_dir = os.path.join(datadir, 'preprocessed')

#%% varie 
continuous_preds = ['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8','cont9', 'cont10', 'cont11', 'cont12', 'cont13']
categorical_preds = ['cat1', 'cat2', 'cat3','cat4', 'cat5', 'cat6', 'cat7', 'cat8']


#%% function to compunte gini
def Gini(y_true, y_pred):
    """compute the gini score

    Arguments:
        y_true {np.array} -- true values
        y_pred {np.array} -- predicted values

    Returns:
        Float -- Normalized Gini
    """    

    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred/G_true

#%%
def lorenz_curve(y_true, y_pred, exposure):
    y_pred_std = y_pred/exposure
    y_true, y_pred_std = np.asarray(y_true), np.asarray(y_pred_std)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred_std)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount

from sklearn.metrics import auc

def NewGini(y_true, y_pred, exposure):
    ordered_samples, cum_claims = lorenz_curve(y_true, y_pred, exposure)
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    return gini

#%% function to log analysis of results
def analyze_results(y_true, y_pred,exposure, verbose=True):
    """Function to analyze results

    Arguments:
        y_true {np.array} -- true outcomes
        y_pred {np.array} -- predicted outcomes
        exposure {np.array} -- exposure

    Keyword Arguments:
        verbose {bool} -- print synthesis of results (default: {True})

    Returns:
        tuple -- tuple oc a/p ratio and gini
    """    
    actual_predicted_ratio = np.sum(y_true) / np.sum(y_pred)
    gini = NewGini(y_true=y_true, y_pred=y_pred, exposure=exposure)
    if verbose:
        print(f"Actual/predicted ratio is {actual_predicted_ratio:.2f} and Gini is {gini:.2f}")
    return actual_predicted_ratio, gini
# %%
