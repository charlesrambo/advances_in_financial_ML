# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:30:19 2024

@author: charlesr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from yfinance import download
from scipy.special import gamma


def get_weights_frac_diff(d, size = None, threshold = None):
    """
    Function to replace getWeights and getWeights_FFD. The gamma function in 
    SciPy computes values much faster than how values are calculated in 
    "Advances in Financial Machine Learning", though computation is not a 
    material issue in either calculation.

    Parameters
    ----------
    d : float
        Amount of differencing. The value of d can be any positive real number. 
        It is not necessarily bounded 0 and 1.
    size : int, optional
        Number of terms to consider. The default is 80 when d < 1, which is 
        materially more than what is required in most contexts. When d >= 1,
        the number of terms is floor(d + 2) which is one more than need when
        d is a positive integer.
    threshold : float, optional
        Once weights fall below the threshold they are dropped. The default is
        None.

    Returns
    -------
    numpy array
        Binomial coefficents for binomial expansion.
    """
    
    if size is None:
        
        size = int(d + 2) if d >= 1 else 80
    
    # Get k-values in binomial expansion
    k_vals = np.arange(0, size)
    
    # Calculate weights
    w = (-1)**k_vals * gamma(d + 1)/(gamma(k_vals + 1) * gamma(d - k_vals + 1))
    
    # Handle bad values
    w = np.nan_to_num(w)
    
    # Change order
    w = w[::-1]
    
    if threshold is None:
        
        return w.reshape(-1, 1)
    
    else:
        
        return w[np.abs(w) > threshold].reshape(-1, 1)
 
    
        
def plot_weights(d_range, num_plots, size, significant_figures = 3):
    
    
    # Get d_values; round results so it looks nice in figure
    d_vals = np.round(np.linspace(d_range[0], d_range[1], num_plots), 
                      significant_figures)
    
    # Construct data frame
    weight_df = pd.concat(
        [pd.DataFrame(get_weights_frac_diff(d, size = size)[::-1], columns = [d])
                   for d in d_vals], axis = 1)
    
    # Plot data frame    
    ax = weight_df.plot()
    
    # Place legend in the top right corner
    ax.legend(loc = 'upper right')
    
    # Show plot
    plt.show()

    
def calculate_frac_diff(df, d, threshold = None):
    """
    Lopez de Prado's fractional differencing. Within this formulation nan 
    values are converted to zero via numpy's np.nan_to_num function. For 
    threshold = 1, all meaningful results are reported.

    Parameters
    ----------
    df : pandas data frame
        Presumably the data frame contains prices that we would like to 
        difference, but it can be any series of values.
    d : float
        Amount of differencing. The value of d can be any positive real number. 
        It is not necessarily bounded 0 and 1.
    threshold : float, optional
        After the sum of the coefficients fall below this level we stop 
        adding additional terms. 

    Returns
    -------
    Pandas data frame of diffenced series
    
    """
    
    # Compute the weights
    w = get_weights_frac_diff(d, size = df.shape[0])

    
    if threshold is not None:
        
        # Determine the fraction of weight explained by the last terms
        frac_wt = np.cumsum(np.abs(w)) 
        frac_wt /= frac_wt[-1]
        
        # Calculate the number of terms to skip
        skip = frac_wt[frac_wt > threshold].shape[0]
        
    else:
        
        skip = 0
    
    # Create dictionary to hold fractionally differenced series
    frac_diff_dict = {}
    
    # Loop over columns of df
    for name in df:
        
        # Get the particular price series
        series = df[[name]].ffill().dropna()
        
        # Initialize series of values
        diff_series = pd.Series(index = series.index[skip:])
        
        # Create function to calculate terms
        calc_term = lambda i: (w[-(i + 1):, :].T @ series.loc[:series.index[i]].values)[0, 0]
        
        # Vectorize function
        calc_term = np.vectorize(calc_term)
        
        # Compute terms
        diff_series[diff_series.index] = calc_term(np.array(range(skip, series.shape[0])))
        
        # Replace non-finite values with np.nan
        diff_series[~np.isfinite(diff_series)] = np.nan
        
        # Add series to dictionary
        frac_diff_dict[name] = diff_series.copy()
    
    # Concatinate the results
    result = pd.concat(frac_diff_dict, axis = 1)
    
    return result


def calculate_frac_diff_fixed(df, d, threshold = 1e-5):
    """
    Lopez de Prado's fractional differencing with a fixed width. 

    Parameters
    ----------
    df : pandas data frame
        Presumably the data frame contains prices that we would like to 
        difference, but it can be any series of values.
    d : float
        Amount of differencing. The value of d can be any positive real number. 
        It is not necessarily bounded 0 and 1.
    threshold : float, optional
        Once terms of w drop below this value, they are no longer calculated. 
        The default is 1e-5.

    Returns
    -------
    Pandas data frame of diffenced series
    
    """
    
    # Compute the weights
    w = get_weights_frac_diff(d, threshold = threshold)
    
    # Get value of width minus 1
    width = w.shape[0] - 1
    
    # Create dictionary to hold fractionally differenced series
    frac_diff_dict = {}
    
    # Loop over columns of df
    for name in df:
        
        # Get the particular price series
        series = df[[name]].ffill().dropna()
        
        # Initialize series of values
        diff_series = pd.Series(index = series.index[width:])
        
        # Create function to calculate terms
        calc_term = lambda i: (w.T @ series.loc[series.index[i - width]:series.index[i]].values)[0, 0]
        
        # Vectorize function
        calc_term = np.vectorize(calc_term)
        
        # Compute terms
        diff_series[diff_series.index] = calc_term(np.array(range(width, series.shape[0])))
        
        # Replace non-finite values with np.nan
        diff_series[~np.isfinite(diff_series)] = np.nan
        
        # Add series to dictionary
        frac_diff_dict[name] = diff_series.copy()
    
    # Concatinate the results
    result = pd.concat(frac_diff_dict, axis = 1)
    
    return result



def plot_yf_ADF(ticker, start, end, frequency = '1D', filename = None): 
    """
    Modification of Lopez de Prado's plotMinFFD function. This function pulls
    a series of adjusted close prices from Yahoo Finance. Displays an image of
    the correlation and augmented Dicky-Fuller statistic.

    Parameters
    ----------
    ticker : string
        The ticker of the time series we would like to compute the fractional
        difference of. For example, '^SPX' will show the fractional differenced
        results for the S&P 500.
    start : string
        The start date of the series as a string.
    end : string
        The end date of the series as a string.
    frequency : string, optional
        The frequency of the price series. The default is daily.
    filename : string, optional
        If the filename is not one, the image will be saved to the specified 
        location.

    Returns
    -------
    The pandas data frame results. This data frame contains results for the 
    given values of d.

    """
    
    # Initialize results data frame
    results = pd.DataFrame(columns = ['adfStat', 'p-val', 'lags', 'n', 
                                      '95% conf', 'corr'])
    
    # Get results from Yahoo Finance
    df0 = pd.DataFrame(download(ticker, start = start, end = end, 
                                progress = False)['Adj Close'])
    
    
    for d in np.linspace(0.0, 1.0, 11):
        
        # Downcast to observations with diven frequency
        df1 = np.log(df0[['Adj Close']]).resample(frequency).last()
        
        # Calculate fractional differenced series
        df2 = calculate_frac_diff_fixed(df1, d, threshold = 0.01)
        
        # Calculate correlation
        corr = df1.loc[df2.index, 'Adj Close'].corr(df2['Adj Close'])
        
        # Perform augmented Dicky-Fullter test
        adf_stats = adfuller(df2['Adj Close'], maxlag = 1, regression = 'c', 
                       autolag = None)
        
        # Add results to row of results data frame
        results.loc[d] = list(adf_stats[:4]) + [adf_stats[4]['5%']] + [corr]
        
    # Plot results
    results[['adfStat', 'corr']].plot(secondary_y = 'adfStat')
    
    # Add horizontal line showing 95% confidence level
    plt.axhline(results['95% conf'].mean(), linewidth = 1, color = 'r', 
                linestyle = 'dotted', label = '95% Conf')
    
    # Add ticker to the top
    plt.suptitle(ticker)
    
    # Add subtitle
    plt.title(f'{df2.index.min().strftime("%d %b %Y")}--{df2.index.max().strftime("%d %b %Y")}')
    
    # If filename is not None...
    if filename is not None:
        
        # ... save everything
        plt.savefig(filename, dpi = 300)

    # Display plot
    plt.show()
    
    return results


# =============================================================================
# plot_weights(d_range = [0, 1], num_plots = 11, size = 6)
# 
# plot_weights(d_range = [1, 2], num_plots = 11, size = 6)
#             
# _ = plot_yf_ADF('MSFT', '2022-01-01', '2023-12-31', frequency = '1D')    
# =============================================================================
            
    
    
    
    
    
    