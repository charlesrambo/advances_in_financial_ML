# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 06:46:25 2024

@author: charlesr
"""

import numpy as np
import pandas as pd


def get_beta(series, sample_length):
    
    h1 = series[['High', 'Low']].values
    h1 = np.log(h1[:,0]/h1[:, 1])**2
    h1 = pd.Series(h1, index = series.index)
    beta = h1.rolling(window = 2).sum()
    beta = beta.rolling(window = sample_length).mean()
    
    return beta.dropna()


def get_gamma(series):
    
    h2 = series['High'].rolling(window = 2).max()
    l2 = series['Low'].rolling(window = 2).min()
    gamma = np.log(h2.values/l2.values)**2
    gamma = pd.Series(gamma, index = h2.index)
    
    return gamma.dropna()


def get_alpha(beta, gamma):
    
    denominator = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2) - 1) * np.sqrt(beta)/denominator
    alpha -= np.sqrt(gamma/denominator)
    
    # Set negative alphas to 0 (see page 727 of paper)
    alpha[alpha < 0] = 0
    
    return alpha.dropna()


def corwin_schultz_spread(series, sample_length = 1):
    
    # Note: S < 0 if and only if alpha < 0
    beta = get_beta(series, sample_length)
    gamma = get_gamma(series)
    alpha = get_alpha(beta, gamma)
    
    spread = 2 * (np.exp(alpha) - 1)/(1 + np.exp(alpha))
    
    return spread

def becker_parkinson_vol(series, sample_length = 1):

    beta = get_beta(series, sample_length)
    gamma = get_gamma(series)
    
    k2 = np.sqrt(8/np.pi)
    denominator = 3 - 2 * np.sqrt(2)
    sigma = (np.sqrt(2) - 1) * np.sqrt(beta)/(k2 * denominator)
    sigma += np.sqrt(gamma/(k2**2 * denominator))
    sigma[sigma < 0] = 0
    
    return sigma


# =============================================================================
# from yfinance import download
# 
# spx = download('^SPX', start = '2020-01-01', end = '2024-05-31', progress = False)
# =============================================================================

