# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:29:55 2024

@author: charlesr
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import rv_continuous, kstest
import matplotlib.pyplot as plt

# Use LaTeX
plt.rcParams['text.usetex'] = True

import Chapter_7 as seven

# Lopez de Prado's fix for annoying Pipeline syntax 
# https://stackoverflow.com/questions/36205850/sklearn-pipeline-applying-sample-weights-after-applying-a-polynomial-feature-t
class MyPipeline(Pipeline):
    
    def fit(self, X, y, sample_weights = None, **fit_params):
        
        if sample_weights is not None:
            
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weights
            
        return super().fit(X, y, **fit_params)


def clf_hyperparameter_fit(X, y, t1, pipe_clf, param_grid, scoring, cv = 3, 
                           bagging_dict = None, n_random_iter = 0, n_jobs = -1, 
                           percent_embargo = 0, **fit_params):
    
    # Construct iterable to use for grid search or random search
    inner_cv = seven.PurgedKFold(n_splits = cv, t1 = t1, 
                                 percent_embargo = percent_embargo) 
    
    # If n_random_iter is 0...
    if n_random_iter == 0:
        
        # ... just perform regular grid search
        gs = GridSearchCV(estimator = pipe_clf, param_grid = param_grid,
                          scoring = scoring, cv = inner_cv, n_jobs = n_jobs)
    
    # Otherwise...
    else:
        
        # ... use random search
        gs = RandomizedSearchCV(estimator = pipe_clf, 
                                param_distributions = param_grid,
                                scoring = scoring, cv = inner_cv, 
                                n_jobs = n_jobs, n_iter = n_random_iter)
     
    # Fit grid search and record the best estimator
    gs = gs.fit(X, y, **fit_params).best_estimator_ 
        
    # fit validated model on the entirety of the data
    if bagging_dict is not None:
        
        gs = BaggingClassifier(estimator = MyPipeline(gs.steps), n_jobs = n_jobs, 
                               **bagging_dict)
        
        gs = gs.fit(X, y, sample_weight = fit_params[gs.base_estimator.steps[-1][0] + '__stample_weight'])
        
        gs = Pipeline([('bag', 'gs')])
        
    return gs


class log_uniform(rv_continuous):
    
    # Rendom numbers log-uniformly distributioned between 1 and e
    def _cdf(self, x):
        return np.log(x/self.a)/np.log(self.b/self.a)
    
def gen_log_uniform(a = 1, b = np.exp(1)):
    return log_uniform(a = a, b = b, name = 'log uniform')



a, b, size = 1e-3, 1e3, 10000

vals = gen_log_uniform(a = a, b = b).rvs(size = size)

print(kstest(rvs = np.log(vals), cdf = 'uniform', args = (np.log(a), np.log(b/a)), N = size))

print(pd.Series(vals).describe())


fig = plt.figure(dpi = 300)

plt.suptitle(r'$X_i$ Distributed Log-Uniform')

ax0 = plt.subplot(121)

pd.Series(np.log(vals)).hist()

ax0.title.set_text(r'$\log(X_i)$')

ax1 = plt.subplot(122)

pd.Series(vals).hist()

ax1.title.set_text(r'$X_i$')

plt.show()

    
    