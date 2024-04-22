# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:44:30 2024

@author: charlesr
"""

import numpy as np
import pandas as pd


def get_indicator_matrix(index, interval_df):
    
    # Get indicator matrix
    indicator_matrix = pd.DataFrame(0, index = index, 
                                   columns = range(interval_df.shape[0]))
    
    # Construct indicator matrix with for-loop
    for i, (t0, t1) in enumerate(interval_df.items()):
        
        indicator_matrix.loc[t0:t1, i] = 1
        
    return indicator_matrix


def get_average_uniqueness(indicator_matrix):
    
    # Average uniqueness from indicator matrix
    
    # Concurrency
    c = indicator_matrix.sum(axis = 1)
    
    # Uniqueness
    u = indicator_matrix.div(c, axis = 0)
    
    # Average uniqueness
    average_uniqueness = u[u > 0].mean()
    
    return average_uniqueness


def sequential_bootstrap(indicator_matrix, sample_length = None):
    
    # Generate a smaple via sequentail bootstrap
    if sample_length is None: sample_length = indicator_matrix.shape[1]
    
    # Initialize sample
    sample = []
    
    for _ in range(sample_length):
        
        average_U = pd.Series(index = indicator_matrix.columns)
        
        for col in indicator_matrix:
            
            matrix = indicator_matrix[sample + [col]]
            
            average_U.loc[col] = get_average_uniqueness(matrix).iloc[-1]
        
        # Calculate probability distribution
        prob = average_U/average_U.sum()
        
        # Add random selection to sample
        sample += [np.random.choice(indicator_matrix.columns, p = prob)]
        
    return sample


# Simplified version Lopez de Prado functions      
def num_concurrent_events(N, window):

    # Create matrix of zeros
    C = np.zeros(shape = (N, N))
    
    # Loop cover columns
    for i in range(N):
        
        if i + window <= N:
            
            C[i:(i + window), i] = 1
     
    # Take the sum across each row
    c = C.sum(axis = 1)
    
    return c


# =============================================================================        
# import Chapter_3 as three
#          
# rtn = pd.DataFrame(np.random.normal(0, 0.30/np.sqrt(52), size = 100), columns = ['rtn'])
# 
# window = 6
# 
# upper = 0.05
# lower = -0.05 + window/2 * 0.30**2/(2 * 52) 
#   
# rtn['sig'] = rtn['rtn'].rolling(window = window).apply(three.get_triple_barrier_label, 
#                                                             raw = True, 
#                                                             args = (upper, lower)).shift(-window + 1)
#         
# print(rtn['sig'].value_counts(normalize = True))
# 
# print(f'upper = {upper:.3f}, lower = {lower:.3f}')
# 
# rtn['c'] = num_concurrent_events(rtn.shape[0], window) 
# 
# rtn['wt'] = rtn['rtn']/rtn['c']
# 
# rtn['wt'] = rtn['wt'].rolling(window = window).apply(lambda x: np.abs(x).sum()).shift(-window + 1)
# 
# 
#               
# 
# interval_df = pd.Series([2, 3, 5], index = [0, 2, 4])
# 
# # Index bars
# index = range(interval_df.max() + 1)
# 
# # Get indicator matrix
# indicator_matrix = get_indicator_matrix(index, interval_df)
# =============================================================================
