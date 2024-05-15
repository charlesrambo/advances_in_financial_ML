# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:03:27 2024

@author: charlesr
"""
import numpy as np
import pandas as pd
import random 
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

import Chapter_20 as twenty


# Create function for one simulation; needed to vectorize
def run_price_sim(seed, forecast, phi, sigma, profit_taking_rule, 
                    stop_loss_rule, max_holding_period = 100, initial_price = 0):
    
    # Set random seed
    np.random.seed(seed)
    
    # Set the initial price
    price = initial_price
    
    # Set the holding period to 0
    holding_period = 0
    
    while True:
        
        # Simulate O-U process
        price = (1 - phi) * forecast + phi * price 
        
        # Add random noise
        price += sigma * random.gauss(0, 1)
        
        # Calculate divergence 
        divergence = price - initial_price
        
        # Add 1 to the holding period
        holding_period += 1
       
        # If divergence greater than profit-taking rule or less than stop-loss rule terminate process
        if (divergence > profit_taking_rule) or (
                divergence < -stop_loss_rule) or (
                    holding_period > max_holding_period):
            
            break
            
    return divergence
            


# Create run_simulations function for vectorization
def run_trading_rule_simulations(forecast, half_life, sigma, profit_taking_rule, 
                    stop_loss_rule, n_iter = 1e5, max_holding_period = 100, 
                    initial_price = 0):
    
    # Calculate phi; larger phi implied slower convergence to forecast
    phi = 2**(-1.0/half_life)
    
    # Vectorize run_price_sim
    run_price_sim_vec = np.vectorize(run_price_sim)
    
    # Run simulations
    output = run_price_sim_vec(seed = np.arange(int(n_iter)), 
                               forecast = forecast, 
                               phi = phi, 
                               sigma = sigma, 
                               profit_taking_rule = profit_taking_rule, 
                               stop_loss_rule = stop_loss_rule, 
                               max_holding_period = max_holding_period, 
                               initial_price = initial_price)
   
    # Calculate mean and std
    mean, std = np.mean(output), np.std(output)
    
    # Return mean, std, and Sharpe ratio
    return mean, std, mean/std



if __name__ == '__main__':
    
    # Create arrays of profit-taking and stop-loss rules
    profit_taking_rules = stop_loss_rules = np.linspace(0, 10, 11)

    # Create lists of forecasts and half-lives
    forecasts, half_lifes = [10, 5, 0, -5, -10], [5, 10, 25, 50, 100]

    # Create numpy array that has all possible products
    results = np.array(list(product(forecasts, half_lifes, profit_taking_rules, 
                                    stop_loss_rules)))

    # Convert from numpy array to data frame
    results = pd.DataFrame(results, columns = ['forecast', 'half_life', 
                                               'profit_taking_rule', 
                                               'stop_loss_rule'])
    
    # Use multiprocessing to get results
    results[['mean', 'std', 'Sharpe']] = twenty.run_queued_multiprocessing(
                                        run_trading_rule_simulations, 
                                        results.index, 
                                        params_dict = {
                                            'forecast':results['forecast'].values, 
                                            'half_life':results['half_life'].values, 
                                            'profit_taking_rule':results['profit_taking_rule'].values,
                                            'stop_loss_rule':results['stop_loss_rule'].values}, 
                                        num_threads = 6, 
                                        mp_batches = 20, 
                                        sigma = 1.0, 
                                        n_iter = 5e4, 
                                        max_holding_period = 100,
                                        initial_price = 0)
    
    # Loop gover forecasts and half-life values
    for forecast, half_life in zip(forecasts, half_lifes):
        
        # Define which rows to keep
        to_keep = (results['forecast'] == forecast) & (
            results['half_life'] == half_life)
        
        # Subset results
        head_map_vals = results.loc[to_keep, ['profit_taking_rule', 
                                              'stop_loss_rule',
                                              'Sharpe']]
        
        # Create values for heatmap
        heatmap_vals = head_map_vals.pivot(index = 'stop_loss_rule', 
                                           columns = 'profit_taking_rule',
                                           values = 'Sharpe')
        
        # Sort values for heatmap
        heatmap_vals = heatmap_vals.sort_index(ascending = False)
        
        # Define axes
        fig, ax = plt.subplots(figsize = (15, 7))

        # Generate heatmap
        sns.heatmap(heatmap_vals, cmap = 'Blues', ax = ax, annot = False)

        # Set title
        ax.set_title(f"Forecast = {forecast} | Half-Life = {half_life} | Sigma = 1")
        
        # Label for x-axis
        ax.set_xlabel('Profit-Taking')
        
        # Label for y-axis
        ax.set_ylabel('Stop-Loss')
        
        # Show result
        plt.show()
        
    del heatmap_vals