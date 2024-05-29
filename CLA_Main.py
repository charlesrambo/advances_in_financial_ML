# -*- coding: utf-8 -*-
"""
Taken directly from zip file found at:
https://www.davidhbailey.com/dhbpapers/
"""
import matplotlib.pyplot as mpl
import numpy as np
import CLA

#---------------------------------------------------------------
def plot2D(x, y, xLabel ='', yLabel = '', title ='', pathChart = None):
    
    fig = mpl.figure()
    
    # One row, one column, first plot
    ax = fig.add_subplot(1,1,1) 
    
    ax.plot(x, y, color = 'blue')
    
    ax.set_xlabel(xLabel)
    
    ax.set_ylabel(yLabel, rotation = 90)
    
    mpl.xticks(rotation = 'vertical')
    
    mpl.title(title)
    
    if pathChart == None:
        
        mpl.show()
        
    else:
        
        mpl.savefig(pathChart)
     
    # Reset pylab
    mpl.clf() 

#---------------------------------------------------------------
def main(path):
    
    #1) Path
    #path = r'string'
    
    #2) Load data, set seed
    headers = open(path + r'\CLA_Data.csv', 'r').readline().split(',')[:-1]
    
    # Load as numpy array
    data = np.genfromtxt(path + r'\CLA_Data.csv', delimiter = ',', 
                         skip_header = 1) 
    
    mean = np.array(data[:1]).T
    lB = np.array(data[1:2]).T
    uB = np.array(data[2:3]).T
    covar = np.array(data[3:])
    
    #3) Invoke object
    cla = CLA.CLA(mean, covar, lB, uB)
    cla.solve()
    
    # Print all turning points
    print(cla.w)
    
    #4) Plot frontier
    mu, sigma, weights = cla.efFrontier(100)
    
    plot2D(sigma,mu,'Risk','Expected Excess Return',
           'CLA-derived Efficient Frontier')
    
    #5) Get Maximum Sharpe ratio portfolio
    sr, w_sr = cla.getMaxSR()
    
    print(np.sqrt(((w_sr.T@ cla.covar) @ w_sr)[0,0]), sr)
    
    print(w_sr)
    
    #6) Get Minimum Variance portfolio
    mv, w_mv = cla.getMinVar()
    
    print(mv)
    
    print(w_mv)
    
    x, y, z, w_ = [], [], [], []
    
    for i in range(len(cla.w) - 1):
        
        w0 = np.copy(cla.w[i])
        w1 = np.copy(cla.w[i + 1])
        
        for a in np.linspace(1, 0, 10000):
            
            w = a * w0 + (1 -a) * w1
            w_.append(w)
            x.append(np.sqrt(((w.T @ cla.covar)@ w)[0,0]))
            y.append((w.T @ cla.mean)[0,0])
            z.append(cla.evalSR(a,w0,w1))
            
    print(np.max(y),w_[np.argmax(z)])
          
    plot2D(x,y,'Risk','Expected Excess Return','CLA-derived Efficient Frontier', 
        path + r'\Figure1.png')
        
    plot2D(x,z,'Risk','Sharpe ratio','CLA-derived Sharpe Ratio Function',
        path + r'\Figure2.png')
#---------------------------------------------------------------
# Boilerplate
if __name__=='__main__':main(path)