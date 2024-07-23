# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 07:46:08 2024

@author: charlesr
"""

import numpy as np
from statsmodels.tsa.stattools import adfuller


# =============================================================================
# def lag_DF(df, lags):
#     
#     # Initialize data frame of lags
#     df_lags = pd.DataFrame()
#     
#     # If lag is an integer...
#     if isinstance(lags, int): 
#         
#         # ... make it every lag from 0 up to lags
#         lags = range(lags + 1)
#     
#     # Otherwise...
#     elif isinstance(lags, list):
#         
#         # ... make sure elements in list are itegers
#         lags = [int(lag) for lag in lags]
#         
#     else:
#         
#         raise Exception(f'Type {type(lags)} is not supported for the variable lag.')
#      
#     # Loop over lags...
#     for lag in lags:
#         
#         # ... shift data frame back
#         df_lag = df.shift(lag).copy()
#         
#         # ... change columns
#         df_lag.columns = [str(col) + '_' + str(lag) for col in df_lag.columns]
#         
#         # ... join to df_lags
#         df_lags = df_lags.join(df_lag, how = 'outer')
#         
#     return df_lags
# =============================================================================

# =============================================================================
# def get_xy(series, regression, lags):
#     
#     # Take the difference of the log prices
#     series_diff = series.diff().dropna()
#     
#     # Add in lags
#     x = lag_DF(series_diff, lags).dropna()
#     
#     # Lagged level
#     x = pd.concat([series.loc[x.index, series.columns[0]], x], axis = 1)
#     
#     # Make x and y have same index
#     y = series_diff.loc[x.index].values
#     
#     # Convert x to numpy array
#     x = x.values
#     
#     if regression != 'n':
#         
#         x = np.append(x, np.ones((x.shape[0], 1)), axis = 1)
#         
#     if regression[:2] == 'ct':
#         
#         trend = np.arange(x.shape[0]).reshape(-1, 1)
#         x = np.append(x, trend, axis = 1)
#         
#     if regression == 'ctt':
#         
#         x = np.append(x, trend**2, axis = 1)
#         
#     return x, y
# 
# =============================================================================


def get_beta_parts(X, y):

    # Calculate parts of OLS formula
    xy = X.T @ y
    xx = X.T @ X

    # Use Moore-Penrose inverse	
    xx_inv = np.linalg.pinv(xx)

    return xx, xx_inv, xy    
    

def get_betas(X, y):
    
    # Calculate parts of OLS formula
    xx, xx_inv, xy = get_beta_parts(X, y)
    
    # Calculate beta
    beta = xx_inv @ xy
    
    # Calculate residuals
    err = y - X @ beta
    
    # Calculate variance of beta
    beta_var = err.T @ err/(X.shape[0] - X.shape[1]) * xx_inv
    
    # Calculate std of beta
    beat_std = np.sqrt(np.diag(beta_var))
    
    return beta, beat_std


def lag_x(x, lags):
    
    # If lag is an integer...
    if isinstance(lags, int): 
        
        # ... make it every lag from 0 up to lags
        lags = range(lags + 1)
    
    # Otherwise...
    elif isinstance(lags, list):
        
        # ... make sure elements in list are itegers
        lags = [int(lag) for lag in lags]
        
        # ... add 0 if it's not there
        lags = [0] + lags if 0 not in lags else lags 
        
        # ... sort results
        lags.sort()
        
    else:
        
        raise Exception(f'Type {type(lags)} is not supported for the variable lag.')
    
    # Concatenate lagged results
    x_lag = np.concatenate([x[(np.max(lags) - lag):(len(x) - lag)].reshape(-1, 1) 
                            for lag in lags], axis = 1)
             
    return x_lag


def np_get_xy(log_price, regression, lags):
    
    # Take the difference of the log prices
    log_rtn = log_price[1:] - log_price[:-1]
    
    # Add in lags
    X = lag_x(log_rtn, lags)
    
    # Get number of observations to skip
    n_skip = log_price.shape[0] - X.shape[0]
    
    # y is the zero lagged difference
    y = log_rtn[n_skip - 1:]
    
    # Replace the first column of X with log_price
    X[:, 0] = log_price[n_skip - 1:-1]
            
    if regression != 'n':
        
        ones = np.ones((X.shape[0], 1))
        X = np.append(X, ones, axis = 1)
        
    if regression[:2] == 'ct':
        
        trend = np.arange(X.shape[0]).reshape(-1, 1)
        X = np.append(X, trend, axis = 1)
        
    if regression == 'ctt':
        
        X = np.append(X, trend**2, axis = 1)
        
    return X, y
    
    
def get_bsadf(log_price, min_sample, regression = 'c', lags = 3):
    
    # Make sure log_price is a numpy array
    log_price = np.asarray(log_price)
    
    # Get x- and y-values
    X, y = np_get_xy(log_price, regression = regression, lags = lags)
    
    # Following warning in statsmodels.tsa.stattools.adfuller
    min_sample = max([int(2 * np.max(maxlag) + 9), min_sample])
    
    # Make iterable of starting points
    start_points = range(0, y.shape[0] - min_sample + 1)
    
    # Initialize list
    adf_vals = np.zeros(len(start_points))
    
    # Loop over starting points
    for start in start_points:
        
        # Subset X and y
        X_sub, y_sub = X[start:], y[start:] 
        
        # Calculate beta and its standard error
        beta, beta_error = get_betas(X_sub, y_sub)
        
        # Calculate ADF statistic
        adf_vals[start] = beta[0]/beta_error[0]
    
    # Supposed to be max, but seems numerically unstable
    return np.nanquantile(adf_vals, 0.99)


# Very slow!
def get_bsadf_statsmodels(log_price, min_sample, maxlag = 3, **kwargs):
    
    # Make sure log_price is a numpy array
    log_price = np.asarray(log_price)
    
    # Following warning in statsmodels.tsa.stattools.adfuller
    min_sample = max([int(2 * np.max(maxlag) + 9), min_sample])
    
    # Make iterable of starting points
    start_points = range(0, log_price.shape[0] - min_sample + 1)
    
    # Initialize list
    adf_vals = np.zeros(len(start_points))
    
    # Loop over starting points
    for start in start_points:
        
        adf_vals[start] = adfuller(x = log_price[start:], 
                                   maxlag = maxlag,
                                   **kwargs)[0]
    
    return np.nanmax(adf_vals)


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
# =============================================================================
# import matplotlib.pyplot as plt
# 
# log_price = np.random.normal(size = 1000)
# 
# for i in range(1, len(log_price)):
#     
#     if i < 500:
#         
#         rho = 0.95 
#         
#     elif i < 600: 
#         
#         rho = 1.03
#         
#     else:
#         
#         rho = 0.8
# 
#     log_price[i] += rho * log_price[i - 1]
# 
# # Add trend
# log_price += 0.01 * np.arange(len(log_price)) + 0.5 * np.random.normal(size = log_price.shape[0])
# 
# maxlag = 3
# min_sample = 10
# interval = 126
# regression = 'ct'
# 
# sadf0 = np.zeros(len(log_price) - interval)
# sadf1 = np.zeros(len(log_price) - interval)
# 
# for t in range(interval, len(log_price)):
#     
#     sadf0[t - interval] = get_bsadf_statsmodels(log_price[(t - interval):t], 
#                                                 min_sample = min_sample, 
#                                                 regression = regression, 
#                                                 maxlag = maxlag)
# 
#     sadf1[t - interval] = get_bsadf(log_price[(t - interval):t], 
#                                     min_sample = min_sample, 
#                                     regression = regression, 
#                                     lags = maxlag)    
# 
# #plt.plot(np.arange(interval, len(log_price)), log_price[interval:], label = 'Log Price')
# plt.plot(np.arange(interval, len(log_price)), sadf0, label = 'SADF0')
# plt.plot(np.arange(interval, len(log_price)), sadf1, label = 'SADF1')
# plt.legend()
# 
# plt.show()
# =============================================================================
