# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:40:11 2024

@author: charlesr
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
import time


 
def linear_parts(atoms, threads):
    
    # Partition of atoms with a single loop
    parts = np.linspace(0, atoms, min(atoms, threads) + 1)
    parts = np.ceil(parts).astype(int)
    
    return parts

def nested_parts(atoms, threads, upper_triange = False):
    
    # partition of atoms with an inner loop
    parts, threads_ = [0], min(atoms, threads)
    
    for _ in range(threads_):
        
        part = 1 + 4 * (parts[-1]**2 + parts[-1] + atoms * (atoms + 1)/threads_)
        part = (-1 + np.sqrt(part))/2
        parts.append(part)
        
    parts = np.round(parts).astype(int)
    
    # If true make the first rows are the heaviest
    if upper_triange: 
    
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
        
    return parts

def expand_call(kargs):
    
    # Get the function arguement
    func = kargs['func']
    
    # Delete it from the fictionary
    del kargs['func']
    
    # Evaluate function with other arguments
    out = func(**kargs)
    
    return out

def process_jobs_single_core(jobs):
    
    print('We have begun processing!\n')
    
    # Run jobs sequentially for debugging
    out = [expand_call(job) for job in jobs]
        
    return out

def report_progress(job_num, num_jobs, start_time):
    
    # Report progress as asynch jobs are completed
    message_stats = [job_num/num_jobs, (time.perf_counter() - start_time)/60]
    message_stats.append(message_stats[1] * (1/message_stats[0] - 1))
    
    time_stamp = time.strftime("%m-%d %H:%M:%S")
    
    message = f'{time_stamp}:'
    message += f'{100 * message_stats[0]: .2f}% complete. '
    message += f'This job took {message_stats[1]:.2f} minutes. ' 
    message += f'About {message_stats[2]:.2f} minutes left.' 
    message += '\n'
    
    if job_num == num_jobs:
        
        message += 'Processing is complete!\n'
    
    print(message)
        

def process_jobs(jobs,  num_threads = 6):
    
    print('We have begun multiprocessing!\n')
    
    # Initialize pool and specify the number of threads
    pool = mp.Pool(processes = num_threads)
    
    # Run imap_unordered; we need index in function to keep track of order
    outputs = pool.imap_unordered(expand_call, jobs)
    
    # Initialize list to contain output
    out = []
    
    # Record time
    start_time = time.perf_counter()
    
    # Process asynchronous output, report progress
    for job_num, out_ in enumerate(outputs, 1):

        # Append results
        out.append(out_)
        
        # Report progress
        report_progress(job_num, len(jobs), start_time)
        
    pool.close()
    pool.join()
    
    return out

# Wrapper to vectorize; hard to pickle np.vectorize because it runs in C
# See https://stackoverflow.com/questions/78307097/multiprocessing-pool-imap-unordered-cant-pickle-function/78307726?noredirect=1#comment138058459_78307726
class vectorize_wrapper:
    
    def __init__(self, pyfunc):
        
        self.__name__ = 'wrapped_' + pyfunc.__name__ 
        
        self.func = np.vectorize(pyfunc)


    def __call__(self, index, *args, **kwargs):
        
        # Convert index to pandas data frame
        index_df = pd.DataFrame(index, columns = ['index'])
            
        # Convert function output to data frame
        out_df = pd.DataFrame(self.func(*args, **kwargs))
           
        if out_df.shape[0] == index_df.shape[0]:
            
            return pd.concat([index_df, out_df], axis = 1)
        
        elif out_df.shape[1] == index_df.shape[0]:

            return pd.concat([index_df, out_df.T], axis = 1)
        
        else:
            
            raise ValueError("The dimensions are inconsistent!")
            
    def __setstate__(self, state):
        
        self.func = np.vectorize(state)

    def __getstate__(self):
        
        return self.func.pyfunc
    
  

def run_queued_multiprocessing(func, index, params_dict, num_threads = 24,
                       mp_batches = 1, linear_molecules = False, 
                       prep_func = True):
    """
    Parallelize jobs, returns a data frame or series.

    Parameters
    ----------
    func : function
        Function to be parallelized. 
    index : list, numpy array, pandas index, or pandas series
        Used to keep track of returned observations
    params_dict: dictionary
        Contains a dictionary of the variables to input into func. The keys are
        the argument names and the values are pandas series of the corresponding
        values.
    num_threads : int, optional
        The number of threads that will be used in parallel (one processor per thread). 
        The default is 24.
    mp_batches : TYPE, optional
        Number of parallel batches (jobs per core). The default is 1.
    linear_molecules : boolean, optional
        Whether partitions will be linear or double-nested. The default is False.
    prep_func: boolean, optional
        Whether to vectorize function and make the first input the index. 
        Functions vectorized using np.vectorize are not pickleable so care must
        be taken to prep the functions if done manually.

    Returns
    -------
    Pandas data frame of sorted outputs

    """
    
    if prep_func:
        
        # Modify function
        new_func = vectorize_wrapper(func)
    
    # Add index to the parameters
    params_dict['index'] = index
    
    # Get observations
    num_obs = len(index)
    
    # Define how we're doing to break up the taks
    if linear_molecules: 
        
        parts = linear_parts(num_obs, num_threads * mp_batches)
        
    else:
        
        parts = nested_parts(num_obs, num_threads * mp_batches)
    
    # Initialize list to hold jobs
    jobs = []
    
    # Creaete jobs
    for i in range(1, len(parts)):
        
        job = {key:params_dict[key][parts[i - 1]:parts[i]] for key in params_dict}
        job.update({'func':new_func})
        jobs.append(job)
        
    # If number of threads is one...   
    if num_threads == 1: 
        
        # ... run simply using list comprehension
        out = process_jobs_single_core(jobs)
    
    # Otherwise...
    else:
        
        # ... use multiprocessing module
        out = process_jobs(jobs, num_threads = num_threads)
        
    
    # Concatinate results in list
    result_df = pd.concat(out, axis = 0)
    
    # Set index as the index and drop as column
    result_df = result_df.set_index('index', drop = True)

    # Sort by the index
    result_df = result_df.sort_index()   
    
    return result_df      

# =============================================================================
# def test_func(a, b):
#     
#     for _ in range(100):
#         
#         continue
#     
#     return a**1 + b**2
# 
# if __name__ == '__main__':
#     
#     test_df = pd.DataFrame(np.random.normal(size = (1000000, 2)), columns = ['a', 'b'])
#     
#     func_dict = {col:test_df[col] for col in test_df.columns}
#       
#     print('Multiprocessing\n')
#     
#     start_time = time.perf_counter()
#     
#     result_df = run_queued_multiprocessing(test_func, test_df.index, func_dict, 
#                                  num_threads = 6, mp_batches = 2)
#     
#     print(f'{time.perf_counter() - start_time:.5f} seconds.\n')
#     
#     del func_dict['index']
#     
#     print('Boaring way.\n')
#     
#     test_func_vec = np.vectorize(test_func)
#     
#     result_df = test_func(**func_dict)
#     
#     print(f'{time.perf_counter() - start_time:.5f} seconds.')    
#     
# =============================================================================
            

    
            