# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:25:33 2024

@author: charlesr
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.special import comb

# Set random seed
np.random.seed(0)

# Generate data for test

# Define alphas
alphas = norm.rvs(loc = 5e-2, scale = 1e-5, size = 75)

# Randomly generate M
M = norm.rvs(loc = alphas/12, scale = 0.50/np.sqrt(12), size = (36, alphas.shape[0])) 

# Convert to pandas data Frame
M = pd.DataFrame(M)

# Create performance statistic; bigger must be better
def sharpe_ratio(x):
    
    return np.sqrt(12) * np.mean(x) /np.std(x)


# Create Baily PBO function
def run_baily_pbo_sims(M, fun, S = 16, shuffle = False, replace = False, 
                       simulations = 50000):
    
    # Get dimensions of M
    T, N = M.shape
    
    # Get index
    index = np.asarray(M.index)
    
    # If shuffle is true
    if shuffle:
        
        # ... shuffle elements
        np.random.shuffle(index)
    
    # Break up the results
    subarrays_list = np.array_split(index, S)
    
    # Calculate number of choices; if replace is true use stars-and-bars
    choices = comb(S + int(S/2) - 1, int(S/2)) if replace else comb(S, int(S/2)) 
    
    # Double because np.random.choice doesn't go through list
    choices *= 2
    
    # Too many combinations to do all for S large
    simulations = int(min(simulations, choices))
      
    # Create pandas data frame to hold results
    results = pd.DataFrame(index = range(simulations), 
                           columns = ['OOS', 'IS', 'Logit'])
    
    # Loop over combinations
    for c in range(simulations):
        
        # Select S/2 element of the S bins
        J_idx = np.random.choice(range(S), size = int(S/2), replace = replace)
        
        # Get the index values
        J_idx = [i for s in J_idx for i in subarrays_list[s]]
            
        # Get the remaining columns
        J_c_idx = [i for i in index if i not in J_idx]
        
        # Get J and J_c using the index
        J = M.loc[J_idx, :].values
        J_c = M.loc[J_c_idx, :].values
        
        # Calculate performance metrics for each strategy IS
        R = np.apply_along_axis(fun, axis = 0, arr = J)
        
        # Get n_star; the argument of max performance metric
        n_star = np.argmax(R)
        
        # Save the best result
        results.loc[c, 'IS'] = R[n_star]
    
        # Calculate performance metrics for each strategy OOS    
        R_c = np.apply_along_axis(fun, axis = 0, arr = J_c)
    
        # Save the best result
        results.loc[c, 'OOS'] = R_c[n_star]
        
        # Calculate omega_c; draws go to value, but float accuracy may affect results (?)
        omega_c = (np.searchsorted(R_c, R_c[n_star]) - 0.5)/N
        
        # Clip results
        omega_c = np.clip(omega_c, a_min = 0.1/N, a_max = 1 - 0.1/N)
        
        # Save logit
        results.loc[c, 'Logit'] = np.log(omega_c/(1 - omega_c))
        
    # Make sure float value
    results = results.astype(float)
    
    return results
       
# Get results
results = run_baily_pbo_sims(M, fun = sharpe_ratio, S = 16)
 
   
# == Generate graphs ==

# Create scatter plot
ax = results.plot.scatter(x = 'IS', y = 'OOS', title = 'OOS Performance Degradation')

# Perform linear regression
reg = LinearRegression().fit(X = results['IS'].values.reshape((-1, 1)), 
                             y = results['OOS'].values)

# Get x-values for prediction
x = np.linspace(results['IS'].min(), results['IS'].max(), 100).reshape((-1, 1))

# Get predicted y-values
y_pred = reg.predict(x)
       
# Plot linear regression
ax.plot(x, y_pred, color = 'red')

# Add annotation
ax.annotate(f"Prob[OOS < 0] = {(results['OOS'] < 0).mean():.2f}",
            xy = (0.65, 0.93), xycoords = 'axes fraction',
            bbox = dict(boxstyle = 'square', fc = 'white'))

# Set axis limits
ax.set(xlim = [results['IS'].quantile(0.01), results['IS'].quantile(0.99)],
       ylim = [results['OOS'].quantile(0.01), results['OOS'].quantile(0.99)])

# Show plot
plt.show()  


# Histograms are way off; don't know if it's because of generated data

# Plot histogram
ax = results['Logit'].hist(bins = 50, density = True)

# Fit normal distribution
loc, scale = norm.fit(results['Logit'].values)

# Get x-values for distribution
x = np.linspace(results['Logit'].min(), results['Logit'].max(), 100)

# Plot x- and y-values for normal density
ax.plot(x, norm(loc = loc, scale = scale).pdf(x), linestyle = 'dashed')

# Add title
ax.set_title('Histogram of Rank Logits')

# Add annotation
ax.annotate(f"Prob Overfit = {(results['Logit'] < 0).mean():.2f}",
            xy = (0.69, 0.93), xycoords='axes fraction',
            bbox=dict(boxstyle='square', fc = 'white'))

plt.show()
