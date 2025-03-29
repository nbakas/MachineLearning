# python ScalabilityTuningRF.py

import numpy as np
import multiprocessing
from multiprocessing import Process, Array
import time
import os
from sklearn.ensemble import RandomForestRegressor

# define the function to parallelize the fitting of the model and the calculation of the pearson correlation
def parallel_sum(all_min_samples_leaf_1, all_min_samples_leaf_2, all_min_samples_leaf_3, all_min_samples_leaf_4, ind_proc, Xtrain, ytrain, Xtest, ytest):
    print(f"Process {ind_proc} started")
    # use this if-else to assign the min_samples_leaf to the process
    if ind_proc == 0:   
        min_samples_leaf = all_min_samples_leaf_1
    elif ind_proc == 1:
        min_samples_leaf = all_min_samples_leaf_2
    elif ind_proc == 2:
        min_samples_leaf = all_min_samples_leaf_3
    elif ind_proc == 3:
        min_samples_leaf = all_min_samples_leaf_4

    # use this for-loop to fit the model and calculate the pearson correlation
    for msf in min_samples_leaf:
        # Initialize the model
        model = RandomForestRegressor(n_jobs=1, min_samples_leaf=msf)
        # Fit the model
        model.fit(Xtrain, ytrain)
        # Predict the model
        pred = model.predict(Xtest)
        # Calculate the pearson correlation
        pearson = np.corrcoef(ytest, pred)[0,1]
        print(f"Process {ind_proc}, min_samples_leaf: {msf}, Pearson: {pearson}")

# define the function to initialize the processes and start the parallel execution
def multiprocessing_sum(num_processes, all_min_samples_leaf_1, all_min_samples_leaf_2, all_min_samples_leaf_3, all_min_samples_leaf_4, Xtrain, ytrain, Xtest, ytest):
    # initialize the processes
    jobs = []

    # start the parallel execution
    start_time = time.time()
    # create the processes
    for ind_proc in range(num_processes):
        # assign the min_samples_leaf to the process
        p = Process(target=parallel_sum, args=(all_min_samples_leaf_1, all_min_samples_leaf_2, all_min_samples_leaf_3, all_min_samples_leaf_4, ind_proc, Xtrain, ytrain, Xtest, ytest))
        # add the process to the list of processes
        jobs.append(p)
        # start the process
        p.start()

    # wait for the processes to finish
    for p in jobs:
        p.join()

    parallel_time = time.time() - start_time

    # print the parallel execution time
    print(f"Parallel Execution Time: {parallel_time*1000:.5f} milliseconds")


# Parallel execution
if __name__ == "__main__":

    # get the number of existing processes
    num_existing_processes = multiprocessing.cpu_count()
    # set the number of used processes
    num_processes = 4
    # print the number of existing and used processes
    print(f"Number of existing processes: {num_existing_processes}, Number of used processes: {num_processes}")

    # generate the data for the training and testing
    n_samples = 1000
    n_features = 10
    X = np.random.rand(n_samples, n_features)
    y = np.sum(X, axis=1)

    # split the data into training and testing
    Xtrain = X[:n_samples//2,:]
    ytrain = y[:n_samples//2]
    Xtest = X[n_samples//2:,:]
    ytest = y[n_samples//2:]

    # define the min_samples_leaf for the training and testing
    all_min_samples_leaf_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    all_min_samples_leaf_2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    all_min_samples_leaf_3 = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    all_min_samples_leaf_4 = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    # start the parallel execution
    multiprocessing_sum(num_processes, all_min_samples_leaf_1, all_min_samples_leaf_2, all_min_samples_leaf_3, all_min_samples_leaf_4, Xtrain, ytrain, Xtest, ytest)