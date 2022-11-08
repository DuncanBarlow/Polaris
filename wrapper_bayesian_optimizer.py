# code copied from https://github.com/fmfn/BayesianOptimization edited by Duncan Barlow
from bayes_opt import BayesianOptimization, UtilityFunction
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


def optimize_known_func(pbounds, init_points, n_iter):

    optimizer = BayesianOptimization(
      f=black_box_function, # Need to change to run this!
      pbounds=pbounds,
      random_state=1,
    )

    optimizer.maximize(
      init_points=init_points,
      n_iter=n_iter,
    )
    return optimizer


def initialize_unknown_func(input_data, target, pbounds, init_points, num_inputs):

    optimizer = BayesianOptimization(
      f=None,
      pbounds=pbounds,
      random_state=1,
    )

    utility = UtilityFunction(kind = "ucb", kappa = 2.5, xi = 0.0)

    # initial data points
    params = {}
    tic = time.perf_counter()
    for ieval in range(init_points):
        # put data in dict
        for ii in range(num_inputs):
            params["x"+str(ii)] = input_data[ii, ieval]
        # add to optimizer
        try:
            optimizer.register(params = params, target = target[ieval])
        except:
            print("Broken input!", input_data[:, ieval])
        if ieval%100 <= 0.0:
            print(str(ieval) + " initialization data points added")
    toc = time.perf_counter()
    print("{:0.4f} seconds".format(toc - tic))

    return optimizer, utility