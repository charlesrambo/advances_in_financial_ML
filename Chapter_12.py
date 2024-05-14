# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:11:58 2024

@author: charlesr
"""

import numpy as np
import pandas as pd
from scipy.special import comb

from sklearn.model_selection._split import  _BaseKFold
from itertools import combinations

# Better version at https://github.com/sam31415/timeseriescv/blob/master/timeseriescv/cross_validation.py

class CombPurgedKFoldCV(_BaseKFold):
    """
    Lopez de Prado's combinatorially purged k-folds cross-validation. This 
    implementation performs k-fold cross-validation while purging data points 
    based on predefined holding periods, specified in the holding_dates data
    frame, as well as purge and embargo periods.
  
    Parameters:
        
        n_splits : int
            Number of folds for k-fold CV.
        
        n_test_splits : int
            Number of test splits per fold (must be between 1 and n_splits-1).
        
        holding_dates : pd.Series or pd.DataFrame
            Pandas object with timestamps representing holding periods.
        
        purge : pd.Timedelta
            Time delta for purging data before the holding period.
        
        embargo : pd.Timedelta
            Time delta for purging data after the holding period.
        
        warm_up_end : pd.Timestamp, optional
            End date for the warm-up period (if any).
        
        fixed_width : pd.Timedelta, optional
            Fixed width for the holding periods if dates not provided in 
            holding_dates.
    """
    
    def __init__(self, n_splits = 5, n_test_splits = 2, holding_dates = None, 
                 purge = pd.Timedelta(days = 0), embargo = pd.Timedelta(days = 0), 
                 warm_up_end = None, fixed_width = None):
        
        if not isinstance(holding_dates, pd.Series)|isinstance(holding_dates, pd.DataFrame):
            
            raise ValueError('Holding dates must be a pandas series or data frame.')
            
        elif isinstance(holding_dates, pd.Series):
            
            holding_dates = pd.DataFrame(holding_dates)
        
        # Save holding_dates as a class object
        self.holding_dates = holding_dates.copy()
        
        if n_test_splits <= 0 or n_test_splits >= n_splits - 1:
            
            raise ValueError(f'K-fold cross-validation requires at least one train/test split.'
                             f'This requires n_test_splits to be between 1 and n_splits - 1, inclusive. '
                             f'Got n_test_splits = {n_test_splits}.')
        
        if fixed_width is not None:
            
            self.holding_dates['t1'] = self.holding_dates['t0'] + fixed_width
            
        else:
            
            if 't1' not in self.holding_dates:
                
                raise ValueError("The pandas object holding_dates must include a column 't1' or you must specify fixed_width.")
                     
        # Save splits as a class object
        self.n_splits = int(n_splits)
        
        # Save test splits as a class object
        self.n_test_splits = int(n_test_splits)
        
        # Save the purge time delta as a class object
        self.purge = purge
        
        # Save the embargo time delta as a class object
        self.embargo = embargo
        
        # Save the warm_up_end as a class object
        self.warm_up_end = pd.Timestamp(warm_up_end)
        
        # Calculate number of paths
        self.path_count = (n_test_splits/n_splits) * comb(n_splits, n_test_splits)


    # Create method to clean up train
    def prep_train(self, train_splits, test_splits):
            
        # Start time of test set minus purge
        start_times = [np.min(a) - self.purge for a in test_splits] 
        
        # End time of test plus embargo; add one hour to fix endpoint problem
        end_times = [np.max(a) + self.embargo + pd.Timedelta(hours = 1) for a in test_splits]
        
        # Initialize is_bad
        is_bad = pd.Series(False, index = self.holding_dates.index)
        
        # Remove outer nesting
        train_dates = [date for a in train_splits for date in a]
        
        for i in range(len(test_splits)):
             
            # Train envelopes test
            envelopes = (self.holding_dates['t0'] <= start_times[i]) & (
                    self.holding_dates['t1'] >= end_times[i])
            
            # Starts in
            starts_in = self.holding_dates['t0'].between(
                start_times[i], end_times[i], inclusive = 'left')
            
            # Ends in
            ends_in = self.holding_dates['t1'].between(
                start_times[i], end_times[i], inclusive = 'right')
                     
            # Three cases:
            
            # (1) the train envelopes test
            is_bad = is_bad|envelopes 
            
            # (2) train starts inside test 
            is_bad = is_bad|starts_in
            
            # (3) train ends inside test
            is_bad = is_bad|ends_in
            
        # What do we want to keep?
        to_keep = holding_dates['t0'].isin(train_dates) & ~is_bad
         
        # Train index values
        train_idx = list(self.holding_dates.loc[to_keep, :].index)
        
        return train_idx
    
    
    # Create method to clean up test   
    def prep_test(self, test_splits):
        
        # Convert test_splits to index values; prep_train already did it for train_splits
        test_idx = np.concatenate([self.holding_dates.loc[self.holding_dates['t0'].isin(a), 
                                                      't0'].index for a in test_splits])
        
        return test_idx.tolist()
        
        
    def split(self, X, y = None, groups = None):
        
        # Check if index lines up
        if np.any(X.index != self.holding_dates.index):
            
            raise ValueError('X and holding dates must have the same index')
          
        # Is it a warm up date?
        is_warm_up = self.holding_dates['t0'] <= self.warm_up_end
        
        # Get the non-warm up dates
        splits = np.array_split(self.holding_dates.loc[~is_warm_up, 't0'].unique(), 
                                self.n_splits) 
        
        # Save missing stuff 
        warm_up_dates = list(self.holding_dates.loc[is_warm_up, 't0'].unique())
            
        # Convert to list so no trouble
        splits = [list(a) for a in splits]
        
        # Loop over splits of index
        for test_splits in combinations(splits, r = self.n_test_splits):
            
            # Get train date lists; add back warm up dates which may be empty
            train_splits = [warm_up_dates] + [a for a in splits if a not in test_splits]
            
            # Fix train; remove layer of brackets, bad observations, convert to index values
            train_idx = self.prep_train(train_splits, test_splits)
            
            # Remove layer of brackets and convert to index values
            test_idx = self.prep_test(test_splits)
               
            # Not helpful if either list is empty
            if len(train_idx) == 0 or len(test_idx) == 0:
                
                continue
            
            else:
                
                yield train_idx, test_idx
                

# =============================================================================
# import matplotlib.pyplot as plt
# 
# holding_dates = pd.DataFrame(index = range(50))
# holding_dates['t0'] = pd.date_range("2018-01-01", periods = holding_dates.shape[0], 
#                                     freq = "W-Fri")
# 
# holding_dates['t1'] =  holding_dates['t0'] + pd.Timedelta(days = 7)
#                
# cv = CombPurgedKFoldCV(n_splits = 5, n_test_splits = 2, 
#                        holding_dates = holding_dates, 
#                        purge = pd.Timedelta(weeks = 2), 
#                        embargo = pd.Timedelta(weeks = 6)) 
# 
# for i, (train_idx, test_idx) in enumerate(cv.split(X = holding_dates, y = None, groups = None)):
#     
#     print(f'i = {i}: ')
#     print(f'train: {train_idx}')
#     print(f'test: {test_idx}')
#     print('\n')
#     
#     plt.title(f'i = {i}')
#     plt.scatter(holding_dates.index, holding_dates.shape[0] * [0], 
#                 color = 'gray', label = 'full')
#     plt.scatter(train_idx, len(train_idx) * [0], color = 'blue', label = 'train')
#     plt.scatter(test_idx, len(test_idx) * [0], color = 'red', label = 'test')   
#     plt.legend()
#     plt.show()
# =============================================================================
                    