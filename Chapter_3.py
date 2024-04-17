# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:34:11 2024

@author: charlesr
"""

import numpy as np
import pandas as pd

import Chapter_20 as twenty

# =============================================================================
# There are likely to be bugs and typos in this code. I fould the functions in
# this section to be more complicated than what I need, so I wrote one function
# at the end which is the only code based on this chapter that I plan to use. 
# As a result, the original functions were not run very many times (if at all).
# =============================================================================


def getDailyVol(close, span = 126, percent = False):
    
    # Calculate return
    rtn = close.pct_change()
    
    # If percent is true...
    if percent:
        
        # .. convert to percent
        rtn *= 100
    
    # Use return to calculate vol
    vol = rtn.ewm(span = span).std()
    
    return vol


def applyPtS1OnT1(close, events, ptS1, molecule):
    """
    
    Parameters
    ----------
    close : pandas series
        Pandas series of closing prices
    events : pandas data frame
        Its columns are the following:
            - 't1': The timestamp of vertical barrier. When the value is np.nan,
            there will not be a vertical barrier.
            - 'target': The unit width of the horizontal barriers.
            - 'side': The side of the position. (?)
    ptS1 : list
        A list of two non-negative float values:
            - ptS1[0]: The factor that multiplies 'target' to set the width of 
            the upper barrier. If 0, there will not be an upper barrier.
            - ptS1[1]: The factor that multiplies 'target' to set the width of 
            the lower barrier. If 0, there will not be a lower barrier.
    molecule : list
        A list with the subset of event indices that will be processed by a 
        single thread. 

    Returns
    -------
    out : pandas data frame
        The output is a pandas data frame containing the timestamps (if any) at
        which each barrier was touched.

    """
       
    # Apply stop loss/profit taking, if it takes place before t1 (end of events)
    events_ = events.loc[molecule]
    
    # Make a deep copy of the data frame with column only 't1'
    out = events_[['t1']].copy()
    
    if ptS1[0] > 0:
        
        pt = ptS1[0] * events['target']
        
    else:
        
        pt = pd.Series(index = events_.index)
        
    if ptS1[1] > 0:
        
        s1 = -ptS1[1] * events_['target']
    
    else:
        
        s1 = pd.Series(index = events_.index)
        
    for t0, t1 in events_['t1'].fillna(close.index[-1]).items():
          
        # Get the price path
        path = (close[t0:t1]/close[t0] - 1) * events_.at[t0, 'side']

        # Earliest profit taking      
        out.loc[t0, 'pt'] = path[path > pt[t0]].index.min()
        
        # Earliest stop loss
        out.loc[t0, 's1'] = path[path < s1[t0]].index.min() 
     
    return out


def getEvents(close, tEvents, ptS1, trgt, minRet, numThreads, numDays = None, 
              verticalBarrier = False, side = None):
    
    # Get target
    trgt = trgt.loc[tEvents]
    
    # Drop observations below minRet
    trgt = trgt[trgt > minRet]
    
    # Get the max holding period t1
    if verticalBarrier:
        
        t1 = close.index.searchsorted(tEvents + pd.Timedelta(days = numDays))
        t1 = t1[t1 < close.shape[0]]
        
        # NaNs at end
        t1 = pd.Series(close.index[t1], index = tEvents[:t1.shape[0]])
        
    else:
        
        t1 = pd.Series(pd.NaT, index = tEvents)
    
    # Form events object, apply stop loss on t1
    if side is None:
        
        side_ = pd.Series(1.0, index = trgt.index)
        ptS1_ = ptS1[ptS1[0], ptS1[0]]
        
    else:
        
        side_ = side.loc[trgt.index]
        ptS1_ = ptS1[:2]
      
    events = pd.concat({'t1':t1, 'target':trgt, 'side':side_}, 
                       axis = 1).dropna(subset = ['target'])
       
    # Use run_queued_multiprocessing to run applyPtS1OnT1 using multithreading
    events = twenty.run_queued_multiprocessing(func = applyPtS1OnT1, 
                                         index = events.index, 
                                         num_threads = numThreads, 
                                         prep_func = False,
                                         close = close, events = events,
                                         ptS1 = ptS1_)
    
    # Drop missing values
    events['t1'] = events.dropna(how = 'all').min(axis = 1)
    
    # If we didn't specify side...
    if side is None:
        
        # ... drop column from events
        events = events.drop('side', axis = 1)
    
    return events


def getBins(events, close):
    """
    Compute event's outcome includig side information if provided.

    Parameters
    ----------
    events : pandas data frame
        - events.index is event's start time
        - events['t1'] is event's end time
        - events['target'] is event's target
        - events['side'] (optional) gives the algorithm's position side
        Case 1: 'side' is not in events and bin in [-1, 1]
        Case 2: 'side' in events and bin in [0, 1]
    close : pandas series
        Closing price information. The index is the date of the closing price.

    Returns
    -------
    out : data frame
        Labels for each observation of events where all necessary information 
        is defined.

    """
    
    # Make it so prices aligned with events
    events_ = events.dropna(subset = ['t1'])
    
    # Extend the events index
    idx = events_.index.union(events_['t1'].values).drop_duplicates()
    
    # Back fill (?) missing prices
    px = close.reindex(idx, method = 'bfill')
    
    # Create out object with index the same as events_
    out = pd.DataFrame(index = events_.index)
    
    # Calculate return
    out['ret'] = px.loc[events_['t1'].values].values/px.loc[events_.index] - 1

    # If side is in events_...
    if 'side' in events_:
        
        # ... adjust return accordingly
        out['ret'] *= events_['side']
    
    # Define bin based on sign
    out['bin'] = np.sign(out['ret'])
    
    if 'side' in events_:
        
        out.loc[out['ret'] <= 0, 'bin'] = 0
    
    return out


def dropLabels(events, minPct = 0.05):
    
    # Apply weights, drop labels with insufficient examples
    while True:
        
        # Get normalized count of bins
        bin_df = events['bin'].value_counts(normalize = True)
        
        # Break if all bin types > 5% or only two bin types left
        if bin_df.min() > minPct or bin_df.shape[0] < 3:
            
            break
        
        # Drop smallest bin type
        events = events.loc[events['bin'] != bin_df.argmin(), :]
        
        # Print a message so we know what's going on
        print(f'Dropped label {bin_df.argmin()}. It was {100 * bin_df.min():.2f}% of observations.')
        
    return events

# === Own version ===

def get_triple_barrier_label(x, upper = np.inf, lower = -np.inf, log_rtn = True, 
                             zero_label = False):
    
    # Convert to numpy array
    x = np.asarray(x)
    
    # If log returns...
    if log_rtn:
        
        # ... the path is of this form
        path = np.exp(np.cumsum(x)) - 1
    
    # If price returns...
    else:
        
        # ... the path is of this form
        path = x/x[0] - 1
    
    # Get indices where the upper barrier is crossed
    upper_args = np.where(path > upper)[0]
    
    # Get indices where the lower barrier is crossed
    lower_args = np.where(path < lower)[0]
    
    # If both lists of arguments are empty... 
    if len(upper_args) == 0 and len(lower_args) == 0:
        
        # ... return either 0 or the sign of the final return
        return 0 if zero_label else np.sign(path[-1])
    
    # If only upper_args is empty...
    elif len(upper_args) == 0:
        
        # ... then lower_args isn't empty so return -1
        return -1
    
    # If only lower_args is empty...
    elif len(lower_args) == 0:
        
        # ... then upper_args isn't empy so return 1 
        return 1
    
    # If neither list is empty... 
    else:
        
        # ... get first element of lists
        upper_arg, lower_arg = upper_args[0], lower_args[0]
        
        # Assign label based on what happened first
        return 1 if upper_arg <= lower_arg else -1
    
# =============================================================================
# import matplotlib.pyplot as plt
# 
# upper, lower = 0.10, -0.07
# 
# fig, ax = plt.subplots(nrows = 4, ncols = 2)
# 
# plt.suptitle(r'Triple Barrier Label')
# 
# 
# for i in range(8):
#     
#     row, col = i//2, i % 2
#    
#     t = np.arange(1, 61, 1)
#     x = np.random.normal(0, 0.30/np.sqrt(250), size = 60)
#     p = 50 * np.exp(np.cumsum(x))
#     
#     label = get_triple_barrier_label(x, upper = upper, lower = lower, 
#                                      log_rtn = True, zero_label = True)
#     
#     ax[row, col].set_title(f'The label is {label}.')
#     
#     ax[row, col].hlines((1 + upper) * 50, xmin = np.min(t), xmax = np.max(t), 
#                         color = 'red')
#     ax[row, col].hlines((1 + lower) * 50, xmin = np.min(t), xmax = np.max(t), 
#                         color = 'red')
#     
#     ax[row, col].vlines(np.max(t), ymin = (1 + lower) * 50, 
#                         ymax = (1 + upper) * 50, 
#                         color = 'red')
#     
#     ax[row, col].plot(t, p)
#     
#     ax[row, col].hlines(50, xmin = np.min(t), xmax = np.max(t), color = 'gray', 
#                linestyles = 'dashed')
# 
# plt.subplots_adjust(hspace = 1)
# 
# plt.show()
# 
# =============================================================================
    
        
    
    
    
    
        
        
    
