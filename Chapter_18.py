# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 08:35:43 2024

@author: charlesr
"""

import numpy as np


def get_pmf(message, window):
    # Compute the prob mass function for a one-dim discrete random variable
    
    # Initialize library
    library = {}
    
    # If message is in the form of a list... 
    if not isinstance(message, str): 
        
        # ... join list elements together
        message = ''.join(map(str, message))
    
    for i in range(window, len(message)):
        
        # Get (i - window)-th data sequance
        x = message[i-window:i]
        
        # If data sequence not in library... 
        if x not in library:
            
            # ... add entry
            library[x] = [i - window]
         
        # Otherwise...
        else:
            
            # ... add element to list
            library[x] += [i - window]
    
    # len(message) - window sequences
    num_sequances = float(len(message) - window)
    
    # Calculate pmf
    pmf = {i:len(library[i])/num_sequances for i in library}
    
    return pmf
            
            
def plug_in(message, window):
    # Compute plug-in (ML) entropy rate

    # Get the pmf of message    
    pmf = get_pmf(message, window)
    
    # Compute Shannon entropy
    H = -np.sum([p * np.log2(p) for p in pmf.values()])/window
    
    return H, pmf


def lempel_ziv_library(message):
    
    # Initialize counter at 1
    i = 1
    
    # Start by adding first and most simple sequence
    library = [message[0]] 
    
    while i < len(message):
        
        for j in range(i, len(message)):
            
            # Get sequence
            x = message[i:j + 1]
            
            # If it isn't in the library...
            if x not in library:
                
                # ... add it
                library.append(x)
                
                break
            
        i = j + 1
        
    return library


def match_length(message, i, n):
    # Maximum matched length + 1, with overlap
    # i >= n & len(mesage) >= i + n
    
    # Initialize substring
    substring = ''
    
    for k in range(n):
        
        x = message[i:i + k + 1]
        
        for j in range(i - n, i):
            
            y = message[j:j + k + 1]
            
            if x == y:
                
                substring = x
                
                break # Search for higher k
    
    return len(substring) + 1, substring #matched length + 1


def konto(message, window = None):
    """
    * Kontoyiannis' LZ entropy estimate, 2013 version (centered window)
    * Inverse of the average length of the shortest non-redundant substring.
    * If non-redundant substrings are short, the text is highly entropic.
    * window == None for expaninding window, in which case len(message) is even.
    * If the end of message is more relevant, try konto[::-1]

    Parameters
    ----------
    msg : TYPE
        DESCRIPTION.
    window : TYPE, optional
        If window is None (default) then expaninding window, in which case 
        len(msg) % 2 == 0

    Returns
    -------
    Kontoyiannis' LZ entropy estimate, 2013 version (centered window)

    """
    out = {'num':0, 'sum':0, 'substring':[]}
    
    if window is None:
        
        if len(message) % 2 == 0:
            
            points = range(1, int(len(message)/2) + 1)            
        
        else:
            
            raise Exception('The message length must be even to use an expanding window')
            
    else:
        
        window = min([window, len(message)/2])
        points = range(window, len(message) - int(window) + 1)      
        
    for i in points:
        
        if window is None:
            
            length, x = match_length(message, i, i)
            
            # To avoid Doeblin condition
            out['sum'] += np.log2(i + 1)/length 
            
        else:
            
            length, x = match_length(message, i, window)

            # To avoid Doeblin condition            
            out['sum'] += np.log2(window + 1)/length 
            
        out['substring'].append(x)
        
        out['num'] += 1
     
    # Entropy
    out['H'] = out['sum']/out['num']
    
    # Redundacy, 0 <= r <= 1
    out['R'] = 1 - out['H']/np.log2(len(message)) 
    
    return out


def calc_portfolio_concentration(cov, wt):
    
    # Make sure numpy arrays
    cov = np.asarray(cov)
    wt = np.asarray(wt)
    
    # Get N
    N = cov.shape[0]
    
    # Calculate eigenvectors and eigenvalues
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Use eigenvectors to calculate loadings
    loadings = eigvecs.T @ wt.reshape((N, 1))
    
    # Flatten loadings
    loadings = loadings.flatten()
    
    # Calculate theta
    theta = loadings**2 * eigvals
    
    # Divide by sum
    theta /= np.sum(theta)
    
    # Drop zeros to make it easier for machine; theta * log(theta) = 0 assumed so fine
    theta = theta[theta > 0]
    
    # Calculate H
    H = 1 - 1/N * np.exp(-np.sum(theta * np.log(theta)))
    
    return H
            