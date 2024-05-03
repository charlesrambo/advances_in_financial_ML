# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:56:27 2024

@author: charlesr
"""
import numpy as np
import pandas as pd
import re
import time
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from itertools import product
import matplotlib.pyplot as plt
import Chapter_7 as seven
from Chapter_20 import run_queued_multiprocessing

# Use LaTeX
plt.rcParams['text.usetex'] = True


def feature_importance_MDI(clf_fit, feat_names):
    
    # Feature importance based on in-sample mean impurity reduction
    df = {i:tree.feature_importances_ for i, tree in enumerate(clf_fit.estimators_)}
    
    # Convert from dictionary to data frame
    df = pd.DataFrame.from_dict(df, orient = 'index')
    
    # Name the columns
    df.columns = feat_names
    
    # If a feature is 0 it means it wasn't seen since in-sample everything helps
    df = df.replace(0, np.nan)
    
    # Calculate the mean and std of the samples
    imp = pd.concat({'mean':df.mean(), 'std':df.std()/np.sqrt(df.shape[0])}, 
                    axis = 1)
    
    # Rescale by dividing by mean
    imp /= imp['mean'].sum()
    
    return imp


def feature_importance_MDA(clf, X, y, cv, sample_weight, t1, percent_embargo, 
                           scoring = 'neg_log_loss'):
    
    # Feature importance based on OOS score reduction
    if scoring not in ['neg_log_loss', 'accuracy']:
        
        raise Exception('Wrong scoring method.')
    
    # Create instance of purged K-folds
    cv_gen = seven.PurgedKFold(n_splits = cv, t1 = t1, 
                              percent_embargo = percent_embargo)
    
    # Initialize pandas objects to hold raw and shuffled neg_log_loss scores
    score_raw, score_shuff = pd.Series(), pd.DataFrame(columns = X.columns)
    
    # Generate split
    for fold, (train_idx, test_idx) in enumerate(cv_gen.split(X = X)):
        
        # Get train arrays
        X_train, w_train = X.iloc[train_idx, :], sample_weight.iloc[train_idx] 
        y_train = y.iloc[train_idx]
        
        # Get test arrays
        X_test, w_test = X.iloc[test_idx, :], sample_weight.iloc[test_idx] 
        y_test = y.iloc[test_idx]
        
        # Fit the model using the training data
        clf_fit = clf.fit(X = X_train, y = y_train, 
                          sample_weight = w_train.values)
        
        if scoring == 'neg_log_loss':
            
            # Use testing data to predict probabilities
            prob = clf_fit.predict_proba(X_test)
            
            # Record negative log-loss
            score_raw.loc[fold] = -log_loss(y_test, prob, 
                                            sample_weight = w_test.values, 
                                            labels = clf.classes_)
            
        else:
            
            # Make class predictions
            pred = clf_fit.predict(X_test)
            
            # Record accuracy
            score_raw.loc[fold] = accuracy_score(y_test, pred, 
                                                 sample_weight = w_test.values)
            
        for col in X.columns:
            
            # Make a deep copy of X_test
            X_shuff = X_test.copy()
            
            # Shuffle the column
            np.random.shuffle(X_shuff[col].values)
            
            if scoring == 'neg_log_loss':
                
                # Predict the probabilities
                probs_shuff = clf_fit.predict_proba(X_shuff)
                
                # Calculate the score
                score_shuff.loc[fold, col] = -log_loss(y_test, probs_shuff, 
                                                       sample_weight = w_test, 
                                                       labels = clf.classes_)
                
            else:
                
                # Make class predictions
                pred_shuff = clf_fit.predict(X_shuff)
                
                # Calculate the score
                score_shuff.loc[fold, col] = accuracy_score(y_test, pred_shuff, 
                                                sample_weight = w_test)
                
    # Subtract the raw score from the score after the shuffle
    imp = score_shuff.sub(score_raw, axis = 0)
    
    if scoring == 'neg_log_loss': 
        
        imp = imp/score_shuff
        
    else:
        
        imp = imp/(score_shuff - 1.0)
    
    # Concatenate results
    imp = pd.concat({'mean':imp.mean(), 'std':imp.std()/np.sqrt(imp.shape[0])}, 
                    axis = 1)
    
    return imp, score_raw.mean()
        
    
def feature_importance_SFI(feat_names, clf, X, y, scoring, cv_gen):

    # Initialize data frame to hold results
    imp = pd.DataFrame(columns = ['mean', 'std'])

    # Loop over feature names
    for feat_name in feat_names:  
        
        # Get cross-validated score
        score = seven.cvScore(clf, X = X[[feat_name]], y = y['bin'],
                            sample_weight = y['w'],
                            scoring = scoring, cv_gen = cv_gen)   

        # Record mean
        imp.loc[feat_name, 'mean'] = score.mean()
            
        # Record standard deviation
        imp.loc[feat_name, 'std'] = score.std()/np.sqrt(score.shape[0])
        
    # Divide std by absolute value of sum
    imp['std'] /= np.abs(imp['mean'].sum())
    
    # Divide mean by sum of absolute values
    imp['mean'] /= imp['mean'].sum()    

    return imp           


def get_test_data(n_features = 40, n_informative = 10, n_redundant = 10, 
                n_samples = 10000, random_state = 0):
    
    # Calculate the number of noise features
    n_noise = n_features - n_informative - n_redundant
    
    # Generate a random dataset for a classification prolem
    np.random.seed(random_state)
    
    # Use make_classification to construct informative and noise features
    X, y = make_classification(n_samples = n_samples,
                               n_features = n_features,
                               n_informative = n_informative,
                               n_redundant = n_redundant,
                               shuffle = False,
                               random_state = random_state)
    
    index = pd.date_range(periods = n_samples, freq = pd.tseries.offsets.BDay(),
                             end = pd.to_datetime('now'))
    
    # Add names for the informative features
    cols = [f'$I_{{{i}}}$' for i in range(n_informative)]

    # Add names for the redundant features
    cols += [f'$R_{{{i}}}$' for i in range(n_redundant)]
    
    # Add names for the noise features
    cols += [f'$N_{{{i}}}$' for i in range(n_noise)]
    
    # Convert results to a pandas data frame
    X = pd.DataFrame(X, columns = cols, index = index)
    y = pd.Series(y, index = index).to_frame('bin')
    
    y['w'] = 1/y.shape[0]
    y['t1'] = index
        
    return X, y


def feat_importance(X, y, n_estimators = 10000, cv = 10, max_samples = 1, 
                   num_threads = 24, percent_embargo = 0, scoring = 'accuracy',
                   method = 'SFI', min_weight_fraction_leaf = 0.0, 
                   verbose = False, **kwargs):
    
    # Run 1 multithreading at a higher level when num_threads > 1
    n_jobs = -1 if num_threads > 1 else 1
    
    # Prepare classifier cv max_features = 1 to prevent masking
    clf = DecisionTreeClassifier(criterion = 'entropy', 
                                 max_features = 1, 
                                 class_weight = 'balanced', 
                                 min_weight_fraction_leaf = min_weight_fraction_leaf)
    
    # Bag classifier
    clf = BaggingClassifier(estimator = clf,
                            n_estimators = n_estimators,
                            max_features = 1.0,
                            max_samples = max_samples, 
                            oob_score = True, 
                            n_jobs = n_jobs)

    # Fit the classifier    
    clf_fit = clf.fit(X = X, y = y['bin'], sample_weight = y['w'].values)
    
    # Get the out-of-bag score
    oob = clf_fit.oob_score_
    
    if method == 'MDI':
        
        imp = feature_importance_MDI(clf_fit, feat_names = X.columns)
        
        oos = seven.cvScore(clf, X = X, y = y['bin'], cv = cv, 
                            sample_weight = y['w'], t1 = y['t1'],
                            percent_embargo = percent_embargo, 
                            scoring = scoring).mean()
        
    elif method == 'MDA':
        
        imp, oos = feature_importance_MDA(clf, X = X, y = y['bin'], cv = cv,
                              sample_weight = y['w'], t1 = y['t1'], 
                              scoring = scoring, 
                              percent_embargo = percent_embargo)
        
    elif method == 'SFI':
        
        cv_gen = seven.PurgedKFold(n_splits = cv, t1 = y['t1'], 
                                  percent_embargo = percent_embargo)
        
        oos = seven.cvScore(clf, X = X, y = y['bin'], sample_weight = y['w'], 
                            t1 = y['t1'], cv_gen = cv_gen, 
                            scoring = scoring).mean()
        
        
        # Paralellize feature_importance_SFI rather than clf
        clf.n_jobs = 1
        
        # Run queued multiprocessing
        imp = run_queued_multiprocessing(feature_importance_SFI, X.columns, 
                                         params_dict = {'feat_names':X.columns}, 
                                         num_threads = 6, mp_batches = 4, 
                                         linear_molecules = False, 
                                         prep_func = False, 
                                         clf = clf, X = X, y = y,
                                         scoring = scoring, cv_gen = cv_gen,
                                         verbose = verbose)
        
    return imp, oob, oos

 
def plot_feat_importance(imp, oob, oos, method, title, filename = None, **kwargs):
    
    # Plot mean imp bars with std
    plt.figure(figsize = (10, imp.shape[0]/5))
    
    # Sort features by mean
    imp = imp.sort_values('mean', ascending = True)
    
    # Plot histogram
    ax = imp['mean'].plot(kind = 'barh', color = 'b', alpha = 0.25,
                          xerr = imp['std'], error_kw = {'ecolor':'r'})
    
    # If MDI...
    if method == 'MDI':
        
        # ... set x-range which is no less than 0
        plt.xlim([0, imp.sum(axis = 1).max()])
        
        # ... draw verticle line to see average feature importance
        plt.axvline(1./imp.shape[0], linewidth = 1, color = 'r', 
                    linestyle = 'dotted')

    # Make the y-axis invisible
    ax.get_yaxis().set_visible(False)

    # Place feature name as center of each bar
    for bar, feature_name in zip(ax.patches, imp.index): 
        
        ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                feature_name, ha = 'center', va = 'center', color = 'black')
    
    # Give plot title
    plt.title(title)
    
    # If filename is defined...
    if filename is not None:
        
        # ... save the plot
        plt.savefig(filename, dpi = 100)
    
    # Show the plot
    plt.show()
    
    # Close figure
    plt.close()
    
    
def test_function(n_features = 30, n_informative = 10, n_redundant = 10, 
             n_estimators = 100, n_samples = 10000, cv = 5, scoring = 'accuracy',
             verbose = False, filename = None):
    """
    Test the performance of the feat importance functions on artificial data
    The number of noise features is n_features - n_informative - n_redundant.
    See https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    for more details on the classification function.

    Parameters
    ----------
    n_features : int, optional
        The number of features considered. The default is 30.
    n_informative : int, optional
        The number of informative features. The default is 10.
    n_redundant : int, optional
        The number of redundat features. The default is 10.
    n_estimators : int, optional
        The number of estimators to use for bagging. The default is 100.
    n_samples : int, optional
        The number of samples to generate. The default is 10000.
    cv : int, optional
        The number of partitions for the cross-validation. The default is 5.
    scoring: string, optional
        Whether to use accuracy or neg_log_loss. The default is accuracy.
    verbose : Boolean, option
        Whether to print progress during miltiprocessing. The default is False.
    filename : str, optional
        If specified, the file location to save the csv file containing the
        results. The default is None.

    Returns
    -------
    out : pandas data frame
        Pandas data frame of results

    """
    
    # Generate the test data
    X, y = get_test_data(n_features, n_informative, n_redundant, n_samples)
    
    # Create a dictionary of parameters
    params_dict = {'min_weight_fraction_leaf':[0.0], 
                   'scoring':[scoring], 
                   'method':['MDI', 'MDA', 'SFI'],
                   'max_samples':[1.0]}
    
    # Create list of jobs to do
    jobs = [dict(zip(params_dict, i)) for i in product(*params_dict.values())]
    
    # Initialize list to hold results
    out = []
    
    # Set key word arguments
    kwargs = {'n_estimators':n_estimators, 'tag':'testFunc', 'cv':cv,
              'verbose':verbose}
    
    if filename is not None:
        
        kwargs['filename'] = filename.repalce('.csv', '.png')
    
    # Loop over jobs
    for job in jobs:
        
        # Get time stamp
        time_stamp = time.strftime("%m-%d %H:%M:%S")
        
        # Print results
        print(time_stamp + ': ' + job['method'] + ' ' + job['scoring'] + '\n' )
        
        # Add job to kwargs
        kwargs.update(job)
        
        # Get feature importance
        imp, oob, oos = feat_importance(X = X, y = y, **kwargs)
        
        # Plot feature importance
        plot_feat_importance(imp = imp, oob = oob, oos = oos, 
                           title = job['method'], **kwargs)
        
        # Divide mean by sum of absolute values
        df_job = imp[['mean']]/imp['mean'].abs().sum()
        
        # Get whether the feature is informative, redundant, or noise
        df_job['Type'] = [re.search(r"['I', 'R', N']", feature_name).group(0) for feature_name in df_job.index]
        
        # Group by type and take the sum
        df_job = df_job.groupby('Type')['mean'].sum().to_dict()
        
        # Add out-of-bas and out-of-sample results
        df_job.update({'oob':oob, 'oos':oos})
        
        # Add parameter information
        df_job.update(job)
        
        # Append to out
        out.append(df_job)
    
    # Convert to data frame and sort by specified columns
    out = pd.DataFrame(out).sort_values(['method', 'scoring', 
                                         'min_weight_fraction_leaf', 
                                         'max_samples'])
    
    # Subset to just columns we care about
    out = out[['method', 'scoring', 'I', 'R', 'N', 'min_weight_fraction_leaf', 
               'max_samples', 'oob', 'oos']]
    
    if filename is not None:
        
        out.to_csv(filename)
        
    return out
    
# =============================================================================    
# if __name__ == '__main__':    
#     
#     out = test_function()
# =============================================================================
