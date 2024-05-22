# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:02:38 2024

@author: charlesr
"""
import numpy as np
from scipy.stats import norm


# Function to generate returns assuming the distribution is a mixture of Gaussians
def gen_mix_model_returns(mu_vals, sigma_vals, probs, n_obs):
    
    # Get the number of gaussians in mixture
    n_gaussians = len(mu_vals)
    
    if (len(sigma_vals) != n_gaussians) or (len(probs) != n_gaussians):
        
        raise ValueError(r"The input dimensions don't match!")
        
    # Covnert probabilities to array
    probs = np.asarray(probs)
     
    # Rescale so they sum to 1
    probs = probs / np.sum(probs)
    
    # Sample component indices based on weights
    component_indices = np.random.choice(n_gaussians, size = n_obs, p = probs)
    
    # Initialize returns
    ret = np.concatenate([np.random.normal(mu_vals[i], sigma_vals[i], 
                            size = int(np.sum(component_indices == i))) 
                    for i in range(n_gaussians)]).flatten()  
    
    # Shuffle results
    np.random.shuffle(ret)
 
    # Return values
    return ret


def compute_precision(stop_loss_rule, profit_taking_rule, freq, target_sharpe_ratio):
    """
    Given a trading rule characterized by the parameters 
    (stop_loss_rule, profit_taking_rule, freq), what is the minimum precision 
    required to achive the target Sharpe ratio?

    Parameters
    ----------
    stop_loss_rule : float
        Stop loss threshold
    profit_taking_rule : float
        Profit taking threshold
    freq : float
        Number of pets per year
    target_sharpe_ratio : float
        target annual Sharpe ratio

    Returns
    -------
        The minimum precision rate is required to achieve target_sharpe_ratio
    """
    
    a = (freq + target_sharpe_ratio**2) * (profit_taking_rule - stop_loss_rule)**2
    b = (2 * freq * stop_loss_rule - target_sharpe_ratio**2 * 
         (profit_taking_rule - stop_loss_rule)) * (profit_taking_rule - stop_loss_rule)
    c = freq * stop_loss_rule**2
    
    # Use quadratic formula to calculate precision
    precision = (-b + np.sqrt(b**2 - 4 * a * c))/(2 * a)
    
    return precision


def compute_sharpe_ratio(stop_loss_rule, profit_taking_rule, freq, precision):
    """
    Given a trading rules characterized by the parameters 
    (stop_loss_rule, profit_taking_rule, freq) for a strategy with given 
    percision, what is the Sharpe ratio?

    Parameters
    ----------
    stop_loss_rule : float
        Stop loss threshold
    profit_taking_rule : float
        Profit taking threshold
    freq : float
        Number of pets per year
    precision : float
        Precision rate of strategy

    Returns
    -------
        Sharpe ratio implied by above parameters.
    """   
    
    # Calculate expected return
    sharpe_ratio = ((profit_taking_rule - stop_loss_rule) * precision 
                    + stop_loss_rule)/(profit_taking_rule - stop_loss_rule)
    
    # Calculate variance and divide expected return by it
    sharpe_ratio /= np.sqrt(precision * (1 - precision)/freq)
    
    return sharpe_ratio


def compute_frequency(stop_loss_rule, profit_taking_rule, precision, target_sharpe_ratio):
    """
    Given a trading rule characterized by the parameters 
    (stop_loss_rule, profit_taking_rule, freq) what is the number of bets/year 
    needed to achieve the target Sharpe ratio?
    None: Equation with radicals, check for extraneou solution.

    Parameters
    ----------
    stop_loss_rule : float
        Stop loss threshold
    profit_taking_rule : float
        Profit taking threshold
    precision: float
        Percision rate requirted to achieve target Sharpe ratio.
    target_sharpe_ratio : float
        target annual Sharpe ratio

    Returns
    -------
    Number of bets per year needed

    """
    
    # Calculate frequency
    freq = (target_sharpe_ratio * (profit_taking_rule - stop_loss_rule))**2 
    freq *= precision * (1 - precision)
    freq /= ((profit_taking_rule - stop_loss_rule) * precision + stop_loss_rule)**2
    
    # Calculate Sharpe ratio
    sharpe = compute_sharpe_ratio(stop_loss_rule, profit_taking_rule, freq, 
                                  precision)
    
    if np.isclose(sharpe, target_sharpe_ratio, rtol = 0.001, atol = 0):
        
        return freq

        

def get_failure_prob(ret, freq, target_sharpe_ratio):
    
    # Get the mean positive return
    pos_ret = ret[ret > 0].mean() 
    
    # Get the mean negative return
    neg_ret = ret[ret <= 0].mean()
    
    # Get the probability of a positive return
    p = ret[ret > 0].shape[0]/ret.shape[0]
    
    # Comput the minimum precision to get a Sharpe ratio no lower than target Sharpe ratio
    precision_threshold = compute_precision(neg_ret, pos_ret, freq, 
                                            target_sharpe_ratio)
    
    # Derive probability that strategy may fail; approximate bootstrap
    risk = norm.cdf(precision_threshold, loc = p, scale = np.sqrt(p * (1 - p)))
    
    return risk


# Parameters
mu = [0.05, -0.10]
sigma = [0.5, 0.1]
probs = [0.75, 0.25]
n_obs = 10000

target_sharpe_ratio = 2
freq = 252

# Generate sample returns from mixture model
ret = gen_mix_model_returns(mu, sigma, probs, n_obs)

# Compute the probability of failure
fail_prob = get_failure_prob(ret, freq, target_sharpe_ratio)

print(f'The probability that the strategy will fail is {100 * fail_prob:0.0f}%.')