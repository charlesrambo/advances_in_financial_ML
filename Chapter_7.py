# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:52:20 2024

@author: charlesr
"""

import numpy as np
import pandas as pd
from sklearn.model_selection._split import  _BaseKFold
from sklearn.metrics import log_loss, accuracy_score

# =============================================================================
# I'm not planning to run this exact code so there has been very limited 
# debugging. Some code may not have even been run.
# =============================================================================


def get_train_times(t1, test_times):
    """
    Lopez de Prado's getTrainTimes. Given test_times, find the times of the 
    train observations.
    
    Parameters
    ----------
    t1 : pandas series
        - t1.index: Time when the observation started.
        - t1.value: Time when the observation ended.
    test_times : pandas series
        Times of testing observations; same format as t1

    Returns
    -------
    Pandas series
    """   
    
    # Make a deep copy of t1 so we don't alter it
    train = t1.copy()
   
    # Loop over index-value pairs
    for t_i, t_j in test_times.items():
        
        # If train starts within test, then drop it
        idx_drop_0 = train[(t_i <= train.index) & (train.index <= t_j)].index 
               
        # If train ends within test, then drop it
        idx_drop_1 = train[(t_i <= train) & (train <= t_j)].index 
        
        # If train envelopes test, then drop it
        idx_drop_2 = train[(train.index <= t_i) & (t_j <= train)].index 
        
        # Drop bad stuff
        train = train.drop(idx_drop_0.union(idx_drop_1).union(idx_drop_2))
        
    return train


def get_embargo_times(times, percent_embargo):
    
    # Calculate step size
    step = int(times.shape[0] * percent_embargo)
    
    if step == 0:
        
        # If step size is 0 embargo of length 0
        embargo = pd.Series(times, index = times)
        
    else:
        
        # Embargo from t_i to t_{i + step}, i.e. from index to value
        embargo = pd.Series(times[step:], index = times[:-step])
        embargo = embargo.append(pd.Series(times[-1], index = times[-step]))
        
    return embargo


class PurgedKFold(_BaseKFold):
    """
    Extend KFold to work with labels that span intervals. The train is purged
    of observations overlapping test-label intervals. Test set is assumed 
    contiguous (suffle = False), without training examples in between.
    """
    
    def __init__(self, n_splits = 3, t1 = None, percent_embargo = 0.0):
        
        if not isinstance(t1, pd.Series):
            
            raise ValueError('Label Through Dates must be a pandas series')
            
        super(PurgedKFold, self).__init__(n_splits, suffle = False, 
                                          random_state = None)
        
        # t1 is a pandas series of start and end times
        self.t1 = t1
        
        # Percent embargo is the fraction of observations skipped
        self.percent_embargo = percent_embargo
 
        
    def split(self, X, y = None, groups = None):
        
        # Check if index lines up
        if (X.index == self.t1.index).sum() != len(self.t1):
            
            raise ValueError('X and ThruDateValues must have the same index')
            
        # Get the number of rows in X
        nrows = X.shape[0]
        
        # Intialize indices
        indices = np.arange(nrows)
        
        # Calculate the embargo step
        embargo_step = int(nrows * self.percent_embargo)
        
        # Get the index splits
        test_starts = [(array[0], array[-1] + 1) for array in np.array_split(indices, self.n_splits)]
        
        # Loop over splits of index
        for i, j in test_starts:
            
            # Start time of test set
            t_0 = self.t1.index[i] 
            
            # Get the test indices
            test_indices = indices[i:j]
            
            # Get the maximum t-value in the test set
            max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            
            # Begin constructing the train indices...
            
            # ... it's the stuff before t_0
            train_indices = self.t1.index.searchsorted(
                self.t1[self.t1 <= t_0 ].index)
            
            # ... and the stuff after max_t1 + embargo_step
            train_indices = np.concatenate([train_indices, 
                                            indices[(max_t1_idx + embargo_step):]])
            
            yield train_indices, test_indices
            

# Open https://github.com/scikit-learn/scikit-learn/issues/6231
# Closed (duplicate of 6231) https://github.com/scikit-learn/scikit-learn/issues/9144

def cvScore(clf, X, y, sample_weight, scoring = 'neg_log_loss', t1 = None, 
            cv = None, cv_gen = None, percent_embargo = None):
    
    # Only supposed to be used for neg_log_loss and accuracy
    if scoring not in ['neg_log_loss', 'accuracy']:
        
        raise Exception('Wrong scoring method')
        
    if cv_gen is None:
        
        # Purged CV as shown in class definition above
        cv_gen = PurgedKFold(n_splits = cv, t1 = t1, 
                             percent_embargo= percent_embargo)
    
    # Initialize list to hold scores
    score = []
    
    # Loop over cross validation generator
    for train, test in cv_gen.split(X = X):
        
        # Fit classifier using training results
        clf_fit = clf.fit(X = X.iloc[train, :], y = y.iloc[train], 
                      sample_weight = sample_weight.iloc[train].values)
        
        # If scoring is negative log loss...
        if scoring == 'neg_log_loss':
            
            # ... calculate predicted probabilities
            prob = clf_fit.predict_proba(X.iloc[test, :])
            
            # ... then calcualte score
            score_ = -log_loss(y.iloc[test], prob,
                sample_weight = sample_weight.iloc[test].values, labels = clf.classes_)
        
        # If scoring is accuracy...
        elif scoring == 'accuracy':
            
            # ... predict y-values
            y_pred = clf_fit.predict(X.iloc[test, :])
            
            # ... calculate the accuracy using the predicted values
            score_ = accuracy_score(y.iloc[test], y_pred , 
                                    sample_weights = sample_weight.iloc[test].values)
        
        # Append score to list of scores
        score.append(score_)
        
    return np.array(score)
        