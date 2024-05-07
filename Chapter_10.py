# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:14:49 2024

@author: charlesr
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

import Chapter_20 as twenty

# =============================================================================
# I'm not planning to run this exact code so there has been very limited 
# debugging. Some code may not have even been run.
# =============================================================================

def avg_base(signals, molecule):
    """
    This function computes the average baseline for a set of molecules in a vectorized manner.

    Parameters
    ----------
    signals : Pandas data frame
        Contains signals. The index is the time when the signal starts, and t1
        is the time when the signal ends.

    molecule : array-like object
        A list of time points for which to compute the average baseline.

    Returns
    -------
    Pandas series
    Contains the average baseline for each time point in molecule.

    """
  
    # Converte molecule to a numpy array
    molecule = np.asarray(molecule)

    # Use boolean indexing to efficiently select the relevant rows
    mask = (signals.index.values <= molecule[:, np.newaxis]) & (
      molecule < signals['t1'].values[:, np.newaxis] | pd.isnull(signals['t1']).values[:, np.newaxis])

    # Efficiently compute the mean along the rows using vectorized operations
    out = signals.loc[mask, 'signal'].mean(axis = 0)

    # Fill missing values with 0
    out = out.fillna(0)

    # Create a Series with the desired index
    return pd.Series(out, index = molecule)

    

def average_active_signals(signals, num_threads, **kwargs):
    
    # Compute the average signal amoung those active
    
    # Time points where signals change (either one starts or one ends)
    time_points = set(signals['t1'].dropna().values)
    time_points = time_points.union(signals.index.values)
    
    # Convert time points to list
    time_points = list(time_points)
    
    # Sort list
    time_points.sort()
    
    # Run queued multiprocessing
    out = twenty.run_queued_multiprocessing(avg_base, time_points,  
                                   num_threads = 6, mp_batches = 12, 
                                   prep_func = False, verbose = False, 
                                   signals = signals)
    
    return out


def discretize_signal(signal, step_size):
    
    # Discrete signal
    signal = (signal/step_size).round() * step_size
    
    # Cap at 1
    signal[signal > 1] = 1.0
    
    # Floor at -1
    signal[signal < -1] = -1.0
    
    return signal


def get_signal(events, step_size, prob, pred, num_classes = 2, num_threads = 6, **kwargs):
    """
    Signals from multinomial classification one-vs-rest

    Parameters
    ----------
    events : Data frame
        Index is start times, t1 is a column containing end times. Possibly a side column
    step_size : float
        How large of a step size for discretization of signal at end
    prob : Pandas series
        Probablilities calculated as one-rest
    pred : Pandas series
        Label found using first ML algorithm. Second algorithm gives probability
    num_classes : int, optional
        Number of classes. The default is 2
    num_threads : int, optional
        Number of threads in multithreading. The default is 6.
    **kwargs : TYPE
        Additional arguments to be passed to average_active_signals

    Returns
    -------
    discrete_signal : Pandas series

    """
    
    # Get singlas from predictions
    if prob.shape[0] == 0: return pd.Series()
    
    # t-value of one-vs-rest probability
    signal = (prob - 1/num_classes)/np.sqrt(prob * (1 - prob))
    
    # signal = side * size
    signal = pred * (2 * norm.cdf(signal) - 1)
    
    # Meta-labeling
    if 'side' in events: signal *= events.loc[signal.index, 'side']
    
    # Convert to data frame and join t1 column of events
    avg_sig = signal.to_frame('signal').join(events[['t1']], how = 'left')
    
    # Compute average signal amoung those concurrently open
    avg_sig = average_active_signals(avg_sig, num_threads, **kwargs)
    
    # Calculate discrete signal
    discrete_signal = discretize_signal(signal = avg_sig, step_size = step_size)
    
    return discrete_signal 




def bet_size(w, x, sigmoid = True):
    
    if sigmoid:
        return x * np.sqrt(1/(w + x**2))
    else:
        return np.sign(x) * np.abs(x)**w


def get_target_position(w, f, p, max_position, sigmoid = True):
    return int(bet_size(w, f - p, sigmoid = sigmoid) * max_position)


def price_from_bet_size(f, w, m, sigmoid = True):
    
    if sigmoid:
        return f - m * np.sqrt(w/(1 - m**2))
    else:
        return f - np.sign(m) * np.abs(m)**(1/w)


def limit_price(target_position, position, f, w, max_position, sigmoid = True):
    
    # Calculate whether we are above or below target
    sign = np.sign(target_position - position)
    
    # Calculate the limit price
    lim_price = np.sum([price_from_bet_size(f, w, j/max_position, sigmoid = sigmoid) 
                        for j in range(np.abs(position + sign), np.abs(target_position + 1))])
    
    # Divide by target_position minus position    
    lim_price /= target_position - position
    
    return lim_price

def get_omega(x, m, sigmoid = True):
    
    if sigmoid:
    
        return x**2 * (m**-2 - 1)
    
    else:
        
        return np.log(np.abs(m))/np.log(np.abs(x))
    


if __name__ == '__main__':
        
    pos, max_pos, price, f = 0, 100, 100, 115
    
    w_params = {'divergence':10, 'm': 0.95} 
    
    # Calibrate w
    w = get_omega(w_params['divergence'], w_params['m'])
    
    # Get target position
    target_pos = get_target_position(w, f, price, max_pos)
    
    # Get the limit price
    lim_price = limit_price(target_pos, pos, f, w, max_pos)
    
        
    # Calculate x-values for plot
    x_vals = np.linspace(-1, 1, 50)
    
    # Give plot a title
    plt.title('Bet Size Functions')
    
    # Plot the sigmoid function
    plt.plot(x_vals, [bet_size(1, x) for x in x_vals], label = 'Sigmoid', 
             linestyle = 'dashed')
    
    # Plot the power function
    plt.plot(x_vals, [bet_size(2, x, False) for x in x_vals], label = 'Power')
    
    plt.legend()
    
    plt.show()
