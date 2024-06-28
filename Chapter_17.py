# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 07:46:08 2024

@author: charlesr
"""

import numpy as np
import pandas as pd


def lag_DF(df, lags):
    
    # Initialize data frame of lags
    df_lags = pd.DataFrame()
    
    # If lag is an integer...
    if isinstance(lags, int): 
        
        # ... make it every lag from 0 up to lags
        lags = range(lags + 1)
    
    # Otherwise...
    elif isinstance(lags, list):
        
        # ... make sure elements in list are itegers
        lags = [int(lag) for lag in lags]
        
    else:
        
        raise Exception(f'Type {type(lags)} is not supported for the variable lag.')
     
    # Loop over lags...
    for lag in lags:
        
        # ... shift data frame back
        df_lag = df.shift(lag).copy()
        
        # ... change columns
        df_lag.columns = [str(col) + '_' + str(lag) for col in df_lag.columns]
        
        # ... join to df_lags
        df_lags = df_lags.join(df_lag, how = 'outer')
        
    return df_lags


def get_beta_parts(x, y):

    # Calculate parts of OLS formula
    xy = x.T @ y
    xx = x.T @ x
    xx_inv = np.linalg.inv(xx)

    return xx, xx_inv, xy    
    

def get_betas(x, y):
    
    # Calculate parts of OLS formula
    xx, xx_inv, xy = get_beta_parts(x, y)
    
    # Calculate beta
    beta = xx_inv @ xy
    
    # Calculate residuals
    err = y - x @ beta
    
    # Calculate variance of beta
    beta_var = err.T @ err/(x.shape[0] - x.shape[1]) * xx_inv
    
    # Calculate std of beta
    beat_std = np.sqrt(np.diag(beta_var))
    
    return beta, beat_std


def get_xy(series, constant, lags):
    
    # Take the difference of the log prices
    series_diff = series.diff().dropna()
    
    # Add in lags
    x = lag_DF(series_diff, lags).dropna()
    
    # Lagged level
    x = pd.concat([series.loc[x.index, series.columns[0]], x], axis = 1)
    
    # Make x and y have same index
    y = series_diff.loc[x.index].values
    
    # Convert x to numpy array
    x = x.values
    
    if constant != 'nc':
        
        x = np.append(x, np.ones((x.shape[0], 1)), axis = 1)
        
    if constant[:2] == 'ct':
        
        trend = np.arange(x.shape[0]).reshape(-1, 1)
        x = np.append(x, trend, axis = 1)
        
    if constant == 'ctt':
        
        x = np.append(x, trend**2, axis = 1)
        
    return x, y
    
    
def get_bsadf(log_price, min_sample, constant, lags):
    
    # Make sure pandas data frame
    if isinstance(log_price, pd.Series):
        
        log_price = pd.DataFrame(log_price.values, columns = ['log_price'], 
                                 index = log_price.index)
    
    # Get x- and y-values
    x, y = get_xy(log_price, constant = constant, lags = lags)
    
    # Adjust minimum sample so always invertible
    min_sample = max([x.shape[1], min_sample])
    
    # Make iterable of starting points
    start_points = range(0, y.shape[0] - min_sample + 1)
    
    # Initialize list
    ADF = []
    
    # Loop over starting points
    for start in start_points:
        
        x_sub, y_sub = x[start:], y[start:] 
        
        beta, beta_error = get_betas(x_sub, y_sub)
        beta, beta_error = beta[0, 0], beta_error[0]
        ADF.append(beta/beta_error)
    
    return np.nanmax(ADF)


def get_brown_durban_evans(x, y, k):
    
    # Make sure x is a numpy array
    x = np.asarray(x)
    
    # Make sure y is a 1D numpy array
    y = np.asarray(y).flatten()
    
    omegas = np.zeros(len(y) - k - 1)
    
    for t in range(k, len(y) - 1):
        
        # Drop last observations    
        x_prev, y_prev = x[:t, :], y[:t] 
        
        x_now, y_now = x[:(t + 1), :], y[:(t + 1)]
        
        # Calculate parts of OLS formula
        beta_prev = get_betas(x_prev, y_prev)[0]
        
        # Use the beta calculation to start omega calculation
        omega = y[t] - x[t, :] @ beta_prev
    
        # Calculate parts of OLS formula this time using last observation
        xx, xx_inv, xy = get_beta_parts(x_now, y_now)
        
        # Calculate beta
        beta = xx_inv @ xy
        
        # Use beta to calculate f
        f = 1 + x[t, :].reshape(1, -1) @ xx_inv @ x[t, :].reshape(-1, 1)
        f *= np.var(y_now - x_now @ beta, ddof = x.shape[1])
        
        # Complete calculation of omega
        omega = omega/np.sqrt(f)
        
        # Add to list of omegas
        omegas[t - k] = omega[0, 0]
        
    # Take the sum
    S = np.sum(omegas)/np.std(omegas)
    
    # Scale so S follows standard normal distribution    
    S *= 1/np.sqrt(x.shape[0] - k - 1)
       
    return S

# =============================================================================
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# 
# x = np.random.normal(size = 1000)
# y = np.append(10 * x[:int(len(x)/2)], 0 * x[int(len(x)/2):], axis = 0) 
# y += np.random.normal(scale = 1e-3, size = len(y))
# 
# x = x.reshape(-1, 1)
# 
# k = 10
# 
# interval = 300
# 
# results = pd.DataFrame(index = range(interval, len(x)), columns = ['test_stat', 'p_val'])
# 
# for t in results.index:
#     
#     S = get_brown_durban_evans(x[(t - interval):t], y[(t - interval):t], k)
#     
#     p_val = 2 * norm.sf(np.abs(S))
#     
#     results.loc[t, :] = S, p_val
#     
# plt.plot(results.index, results['test_stat'], label = 'Test Statistic')
# plt.plot(results.index, results['p_val'], label = 'p-value')
# 
# t = interval
# 
# while t < np.max(results.index):
#     
#     if results.loc[t, 'p_val'] < 0.05:
#         
#         plt.axvline(x = t, color = 'red', linestyle = 'dashed')
#         
#         print(f'Regime shift at t = {t}')
#         
#         t += 20
#         
#     else:
#         
#         t += 1
# 
# plt.axvline(x = len(x)/2, color = 'blue', label = 'Regime Shift')
# 
# plt.legend()        
#         
# plt.show()
# =============================================================================




        
        


        
        
            
        
        
        