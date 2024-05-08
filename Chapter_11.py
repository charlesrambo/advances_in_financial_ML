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

# Define dimensions
T, N = 36, 75

# Define S
S = 10

# Define alphas
alphas = norm.rvs(loc = 5e-2, scale = 1e-5, size = N)

# Randomly generate M
M = norm.rvs(loc = alphas/12, scale = 0.50/np.sqrt(12), size = (T, N)) 

# Create list subarrays

# First generate the elements
subarrays_list = np.arange(T)

# Shuffle elements
np.random.shuffle(subarrays_list)

# Break up the results
subarrays_list = [subarrays_list[int(i * T/S):int(i * T/S + T/S)] for i in range(S)]

# Too many permutations to do all for large S; just choose a large number
num_perms = 5000

# Create performance statistic
def sharpe_ratio(x):
    
    return np.sqrt(12) * np.mean(x) /np.std(x)

# Create pandas data frame to hold results
results = pd.DataFrame(index = range(num_perms), 
                       columns = ['OOS', 'IS', 'Logit'])

# Loop over permutations
for perm in range(num_perms):
    
    # Select S/2 of S; was combinations in book but doesn't make sense to me
    J_idx = np.random.choice(range(S), size = int(S/2), replace = False)
        
    # Get the remaining columns
    J_c_idx = [i for i in range(T) if i not in J_idx]
    
    # Get J and J_c using the index
    J = M[J_idx, :]
    J_c = M[J_c_idx, :]
    
    # Calculate performance metrics for each strategy IS
    R = np.apply_along_axis(sharpe_ratio, axis = 0, arr = J)
    
    # Get n_star; the argument of max performance metric
    n_star = np.argmax(R)
    
    # Save the best result
    results.loc[perm, 'IS'] = R[n_star]

    # Calculate performance metrics for each strategy OOS    
    R_c = np.apply_along_axis(sharpe_ratio, axis = 0, arr = J_c)

    # Save the best result
    results.loc[perm, 'OOS'] = R_c[n_star]
    
    # Calculate omega_c
    omega_c = (np.searchsorted(R_c, R_c[n_star]) - 0.5)/N
    
    # Clip results
    omega_c = np.clip(omega_c, a_min = 0.1/N, a_max = 1 - 0.1/N)
    
    # Save logit
    results.loc[perm, 'Logit'] = np.log(omega_c/(1 - omega_c))
    
# Make sure float value
results = results.astype(float)
    

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
ax = results['Logit'].hist(bins = 100, density = True)

# Fit normal distribution
loc, scale = norm.fit(results['Logit'].values)

# Get x-values for distribution
x = np.linspace(results['Logit'].min(), results['Logit'].max(), 100)

# Plot x- and y-values for normal density
ax.plot(x, norm(loc = loc, scale = scale).pdf(x), linestyle = 'dashed')

# Add title
ax.set_title('Histogram of Rank Logits')

# Add annotation
ax.annotate(f"Prob Overfit = {(results['Logit'] > 0).mean():.2f}",
            xy = (0.69, 0.93), xycoords='axes fraction',
            bbox=dict(boxstyle='square', fc = 'white'))

plt.show()
