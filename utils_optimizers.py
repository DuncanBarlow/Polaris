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



def gradient_stencil(X_new, learning_rate, pbounds, num_inputs, stencil_size):
    X_stencil = np.zeros((num_inputs, stencil_size))

    counter = 0
    X_stencil[:, counter] = X_new[:,0]
    counter += 1
    for ii in range(num_inputs):
        X_stencil[:, counter] = X_new[:,0]
        X_stencil[ii, counter] = X_new[ii,0] - learning_rate
        if (X_stencil[ii,counter] < pbounds[ii,0]):
            X_stencil[ii,counter] = pbounds[ii,0] # to avoid stencil leaving domain
        counter += 1
        X_stencil[:, counter] = X_new[:,0]
        X_stencil[ii, counter] = X_new[ii,0] + learning_rate
        if (X_stencil[ii,counter] > pbounds[ii,1]):
            X_stencil[ii,counter] = pbounds[ii,1] # to avoid stencil leaving domain
        counter += 1

    return X_stencil



def determine_gradient(X_stencil, target, learning_rate, pbounds, num_inputs):

    grad = np.zeros(num_inputs)
    counter = 0
    f_centre = target[counter]
    counter += 1
    for ii in range(num_inputs):

        centred_diff = True
        forward_diff = False
        backward_diff = False

        if (X_stencil[ii,counter] < pbounds[ii,0]):
            centred_diff = False
            forward_diff = True
        else:
            f_minus = target[counter]
        counter += 1

        if (X_stencil[ii,counter] > pbounds[ii,1]):
            centred_diff = False
            backward_diff = True
        else:
            f_plus = target[counter]
        counter += 1

        if centred_diff:
            grad[ii] = (f_plus - f_minus) / (2.0 * learning_rate)
        elif forward_diff:
            grad[ii] = (f_plus - f_centre) / learning_rate
        elif backward_diff:
            grad[ii] = (f_centre - f_minus) / learning_rate
        else:
            grad[ii] = 0.0
            print("Broken gradients!")

    return grad



def grad_descent(X_old, grad, step_size, pbounds, num_inputs, num_steps_per_iter):

    learning_rates = np.logspace(step_size[0], step_size[1], num_steps_per_iter)
    X_new = np.zeros((num_inputs, num_steps_per_iter))
    for ieval in range(num_steps_per_iter):
        X_new[:,ieval] = X_old[:,0]
        for ii in range(num_inputs):
            X_new[ii,ieval] = X_old[ii,0] - learning_rates[ieval] * grad[ii]
            if (X_new[ii,ieval] < pbounds[ii,0]):
                X_new[ii,ieval] = pbounds[ii,0]
            elif (X_new[ii,ieval] > pbounds[ii,1]):
                X_new[ii,ieval] = pbounds[ii,1]

    return X_new
