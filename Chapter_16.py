# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:41:39 2024

@author: charlesr
"""
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns

import CLA


def get_quasi_diag(link):
    
    # Convert to integer
    link = link.astype(int)
    
    # Get the adjacent clusters merged on last step of clustering algorithm
    adj_items = pd.Series([link[-1, 0], link[-1, 1]])
    
    # Number of original items
    n_features = link[-1, 3]

    # Repeat process while there are still clusters with more than one element
    while adj_items.max() >= n_features:
        
        # Make space; going to use odd index for adjacent clusters
        adj_items.index = range(0, 2 * adj_items.shape[0], 2) 
        
        # Find clusters that were formed by merging of other clusters
        merged_clusters = adj_items[adj_items >= n_features]
        
        # Get the index of these clusters
        i = merged_clusters.index
        
        # Get the cluster labels
        j = merged_clusters.values - n_features
        
        # Replace merged clusters with those lower on dendrogram
        adj_items[i] = link[j, 0]
        
        # Add clusters that were merged with cluster j on rows with odd index
        adj_items = pd.concat([adj_items, pd.Series(link[j, 1], index = i + 1)])
        
        # Re-sort
        adj_items = adj_items.sort_index()
        
    return adj_items.tolist()


def get_IVP(cov, **kwargs):
    
    # Compute the inverse-variance portfolio
    ivp = 1/np.diag(cov)
    
    ivp /= ivp.sum()
    
    return ivp


def get_HRP(cov, corr):
    
    # Construct a hierarchical portfolio
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    
    # Calculate distance
    dist = corr_dist(corr)
    
    # Use hierarchical clustering on distance of distance
    link = sch.linkage(dist, 'single')
    
    # Find adjacent clusters
    adj_clusters = get_quasi_diag(link)
    
    # Recover labels
    adj_clusters = corr.index[adj_clusters].tolist()
    
    # Perform HRP
    hrp = get_recursive_bisect(cov, adj_clusters)
    
    return hrp.sort_index()


def get_CLA(cov, **kwargs):
    
    # Compute CLA's minimum variance portfolio
    
    # Not used by minimum variance portfolio
    mean = np.arange(cov.shape[0]).reshape(-1, 1)
    
    # Define upper and lower bounds
    lower_bound = np.zeros(mean.shape)
    upper_bound = np.ones(mean.shape)
    
    # Initialize instance of CLA
    cla = CLA.CLA(mean, cov, lower_bound, upper_bound)
    
    # Solve
    cla.solve()
    
    # Provide weights; last one is MVE with lowest expected return
    return cla.w[-1].flatten()


def get_cluster_var(cov, cluster):
    
    # Compute variance per cluster
    
    # Subset to just rows and columns in cluster
    cluster_cov = cov.loc[cluster, cluster].values
    
    # Get IVP weight
    wt = get_IVP(cluster_cov).reshape(-1, 1)
    
    # Calculate variance
    cluster_var = (wt.T @ cluster_cov @ wt)[0, 0]
    
    return cluster_var


def bisect_item(item):
    # Function write just to make code more redable
    
    if len(item) > 1:
        
        return [item[0:int(len(item)/2)], item[int(len(item)/2):]]
    
    else:
        
        return []


def get_recursive_bisect(cov, adj_items):
    
    # Compute HRP allocation
    w = pd.Series(1.0, index = adj_items)
    
    # Initialize all items in one cluster
    cluster_items = [adj_items]
    
    while len(cluster_items) > 0:
        
        # Bisect list of lists of adjacent clusters
        cluster_items = [b for item in cluster_items for b in bisect_item(item)]
        
        # Parse in pairs
        for i in range(0, len(cluster_items), 2):
            
            # Cluster 1
            left_items = cluster_items[i]
            
            # Cluster 2
            right_items = cluster_items[i + 1]
            
            # Calculate variance of cluster
            var_left = get_cluster_var(cov, left_items)
            var_right = get_cluster_var(cov, right_items)
            
            # Calculate scalar to modify weight
            alpha = 1 - var_left/(var_left+ var_right)
            
            # Rescale weights
            w[left_items] *= alpha
            w[right_items] *= 1 - alpha
            
    return w


def corr_dist(corr):
    
    # A distance matrix based on correlation, where 0 <= d[i, j] <= 1
    
    # This is a proper distance metric
    dist = np.sqrt(0.5 * (1 - corr))
    
    return dist


def plot_corr_matrix(corr, path = None, labels = None, **kwargs):
    
    # Heatmap of the correlation matrix
    if labels is None: labels = []
    
    fig, ax = plt.subplots(figsize = (15, 7))
    
    sns.heatmap(corr, cmap = 'Blues', ax = ax, annot = True, fmt = '.2f')
    
    if path is not None:
        
        # ... save the figure
        plt.savefig(path)
        
    plt.show()
    
    plt.clf()
    
    # Reset pylab
    plt.close()
    
    
def shock_data(x, cols, n_obs, n_cols, n_redundant, s_length, shock_scale = 1):
    
    # Add common random shock
    point = np.random.randint(s_length, n_obs - 1, size = 2)
    x[np.ix_(point, [cols[0], n_cols - n_redundant])] = shock_scale * np.array([[-0.5, -0.5], [2, 2]])
    
    # Add specific random shhock
    point = np.random.randint(s_length, n_obs - 1, size = 2)
    x[point, cols[-1]] = shock_scale * np.array([-0.5, 2])
      
    return x
    

def generate_data(n_obs, n_cols, n_redundant, mu1, mu2, sigma1, sigma2, rho, 
                  seed = None, **kwargs):
    
    # Time series of correlated variables
    
    # Set random seed
    np.random.seed(seed = seed)
    
    # Each column is a variable
    x = np.random.normal(0, 1.0, size = (n_obs, n_cols))
    
    # Select columns from original set to be redundant variables
    cols = np.random.choice(n_cols - n_redundant, size = n_redundant).tolist()
    
    # Scale down the idiosyncratic component
    x[:, n_cols - n_redundant:] *= np.sqrt(1 - rho**2)
    
    # The redundant variables have correlation rho with the variables in cols
    x[:, n_cols - n_redundant:] += rho * x[:, cols]
    
    # Give stastical properties
    x[:, :n_cols - n_redundant] = mu1 + sigma1 * x[:, :n_cols - n_redundant]
    x[:, n_cols - n_redundant:] = mu2 + sigma2 * x[:, n_cols - n_redundant:]    
    
    # Shock x if parameters are given
    try:
        
        x = shock_data(x, cols, n_obs, n_cols, n_redundant, **kwargs)
        
        print('The data were shocked!')
        
    except:
        
        print('The data were not shocked!')
        
        pass
    
    # Convert to data frame; I think bad design but copied from book
    x = pd.DataFrame(x, columns = range(1, 1 + x.shape[1]))
    
    # Redundant dictionary
    redunt_dict = {n_cols - n_redundant + i:cols[i - 1] + 1 for i in range(1, n_redundant + 1)}
    
    return x, redunt_dict


def run_full_hrp_implementation(n_obs = 100000, n_cols = 5, n_redundant = 2, 
                                mu1 = 0, mu2 = 0, sigma1 = 1, sigma2 = 1.25, 
                                rho = 0.75):
    
    # Generate data
    x, redunt_dict = generate_data(n_obs, n_cols, n_redundant, mu1, mu2, sigma1, 
                                   sigma2, rho, seed = 1)
    
    # Print redundancy
    print('Redundant column and original')
    
    for key in redunt_dict:
        
        print(f'{key}:{redunt_dict[key]}')
    
    # Get covariance and correlation matracies
    cov, corr = x.cov(), x.corr()
    
    # Compute and plot correlation matrix
    plot_corr_matrix(corr, labels = corr.columns, cmap = 'Blues')
    
    # Use correlations to generate distances
    dist = corr_dist(corr)
    
    # Cluster based on distance of distances
    link = sch.linkage(dist, method = 'single', metric = 'euclidean')
    
    # Plot dendrogram
    _ = sch.dendrogram(link)
    
    # Order based on adjacency
    adj_clusters = get_quasi_diag(link)
    
    # Recover labels (starting columns at 1 makes it confusing)
    adj_clusters = corr.index[adj_clusters].tolist()
    
    # Reorder
    corr_sorted = corr.loc[adj_clusters, adj_clusters]
    
    # Plot correlation matrix after reordering
    plot_corr_matrix(corr_sorted, labels = corr_sorted.columns, cmap = 'Blues')
    
    # Use HRP to obtain capital allocation
    hrp = get_recursive_bisect(cov, adj_clusters)
    
    print(hrp)
    
    
def hrp_MC(n_iters = 1e4, n_obs = 520, n_cols = 10, n_redundant = 5, mu1 = 0, 
           mu2 = 0, sigma1 = 0.01, sigma2 = 0.015, rho = 0.95, s_length = 260, 
           rebal = 22, shock_scale = 0.1, seed = 0):
    # Low vol better for rho small; at rho = 0.75 low vol materially reduces var compared to HRP
    
    # Methods to try
    methods = [get_IVP, get_HRP, get_CLA]
    
    # Define shared index for stats
    index = range(int(n_iters))
    
    # Initialize dictionary to hold stats for each method
    stats = {m.__name__:pd.Series(index = index) for m in methods}
    
    # Helps us calculate in- and out-of-sample observations
    pointers = range(s_length, n_obs, rebal)
    
    for i in index:
              
        # Prepare data for one experiment
        x, _ = generate_data(n_obs, n_cols, n_redundant, mu1, mu2, sigma1, 
                             sigma2, rho, seed = i + seed, s_length = s_length, 
                             shock_scale = shock_scale)
        
        # Initialize dictionary to hold returns
        ret = {m.__name__:pd.Series() for m in methods}
        
        # Compute portfolios in-sample
        for pointer in pointers:
            
            # Asset returns in-sample
            x_in = x.iloc[pointer - s_length:pointer, :]
            
            # Calculate sample covariance and correlation
            S, C = x_in.cov(), x_in.corr()
            
            # Asset returns out-of-sampe
            x_out = x.iloc[pointer:pointer + rebal, :].values
            
            # Calculate weights for each method
            wt = [func(cov = S, corr = C) for func in methods]
            
            # Convert list to array
            wt = np.array(wt)  

            # Calculate array of returns
            r_m = x_out @ wt.T  

            # Store returns for each method
            for j, func in enumerate(methods):
                
                # Convert to series
                r = pd.Series(r_m[:, j])
                
                # pd.concat functionality for empty series is depreciated
                if ret[func.__name__].shape[0] > 0:
                    
                    ret[func.__name__] = pd.concat([ret[func.__name__], r])
                    
                else:
                    
                    ret[func.__name__] = r                    
                
        # Evaluate and store results
        for func in methods:
            
            ret_ = ret[func.__name__].reset_index(drop = True)
            
            # Get cumulatice return
            cum_ret = (1 + ret_).cumprod() - 1
            
            # Record results
            stats[func.__name__].loc[i] = cum_ret.iloc[-1]
    
    # Convert stats to a dictionary
    stats = pd.DataFrame.from_dict(stats, orient = 'columns')
    
    return stats     
        