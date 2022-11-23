# code copied from https://github.com/fmfn/BayesianOptimization edited by Duncan Barlow
from bayes_opt import BayesianOptimization, UtilityFunction
import numpy as np
import sys
import time
import training_data_generation as tdg
import netcdf_read_write as nrw
import os
import shutil
import copy


def define_optimizer_dataset(X_all, Y_all, avg_powers_all):
    dataset_optimizer = {}
    dataset_optimizer["X_all"] = X_all
    dataset_optimizer["Y_all"] = Y_all
    dataset_optimizer["avg_powers_all"] = avg_powers_all
    return dataset_optimizer



def define_optimizer_parameters(run_dir, num_inputs, num_modes, num_init_examples, n_iter, num_parallel):
    optimizer_params = {}
    optimizer_params["run_dir"] = run_dir
    optimizer_params["num_inputs"] = num_inputs
    optimizer_params["num_modes"] = num_modes
    optimizer_params["num_init_examples"] = num_init_examples
    optimizer_params["n_iter"] = n_iter
    optimizer_params["num_parallel"] = num_parallel
    optimizer_params["iter_dir"] = "iter_"
    optimizer_params["trainingdata_filename"] = "flipped_training_data_and_labels.nc"
    optimizer_params["hemisphere_symmetric"] = True
    optimizer_params["run_clean"] = True
    optimizer_params["random_generator"] = np.random.default_rng(12345)
    return optimizer_params

#################################### Bayesian Optimization ################################################

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

######################################## Gradient Descent ############################################

def define_gradient_descent_params(num_steps_per_iter):
    grad_descent_params = {}
    grad_descent_params["learn_exp"] = -1.0
    grad_descent_params["num_steps_per_iter"] = num_steps_per_iter
    return grad_descent_params



def wrapper_gradient_descent(dataset_optimizer, grad_descent_params, optimizer_params):
    learning_rate = 10.0**grad_descent_params["learn_exp"]
    step_size = np.array([grad_descent_params["learn_exp"] - 1.0, grad_descent_params["learn_exp"] + 1.0])
    stencil_size = optimizer_params["num_inputs"] * 2 + 1

    X_old = np.zeros((optimizer_params["num_inputs"], 1))
    Y_old = np.zeros((optimizer_params["num_modes"], 1))
    avg_powers_old = np.array([0.0])

    mindex = np.argmin(np.sqrt(np.sum(dataset_optimizer["Y_all"]**2, axis=0)))
    X_old[:,0] = dataset_optimizer["X_all"][:, mindex]

    pbounds = np.zeros((optimizer_params["num_inputs"], 2))
    pbounds[:,1] = 1.0
    tic = time.perf_counter()
    for ieval in range(optimizer_params["n_iter"]):

        if (sum(abs(dataset_optimizer["X_all"][:,-1] - dataset_optimizer["X_all"][:,-2])) <= 0.0):
            grad_descent_params["learn_exp"] = grad_descent_params["learn_exp"]-0.5
            learning_rate = 10.0**(grad_descent_params["learn_exp"])
            step_size = step_size - 0.5
            print("Reducing step size to: " + str(learning_rate))
            if learning_rate < 1.0e-4:
                print(str(ieval+1) + " Bayesian data points added, saving to .nc")
                print("Early stopping due to repeated results")
                filename_trainingdata = optimizer_params["run_dir"] + '/' + optimizer_params["trainingdata_filename"]
                nrw.save_training_data(dataset_optimizer["X_all"], dataset_optimizer["Y_all"],
                                       dataset_optimizer["avg_powers_all"], filename_trainingdata)
                break

        X_stencil = gradient_stencil(X_old, learning_rate, pbounds,
                                     optimizer_params["num_inputs"], stencil_size)
        Y_stencil, avg_powers_stencil = tdg.run_ifriit_input(stencil_size, X_stencil,
                                                             optimizer_params["run_dir"],
                                                             optimizer_params["num_modes"],
                                                             optimizer_params["num_parallel"],
                                                             optimizer_params["hemisphere_symmetric"],
                                                             optimizer_params["run_clean"])
        target_stencil = np.sqrt(np.sum(Y_stencil**2, axis=0))
        mindex_stencil = np.argmin(target_stencil)
        print("The minimum in the stencil", np.min(target_stencil), mindex_stencil)
        print("The previous value was: ", target_stencil[0], 0)
        print(X_stencil[:,0])
        os.rename(optimizer_params["run_dir"]  + "/run_" + str(mindex_stencil),
                  optimizer_params["run_dir"] + "/" + optimizer_params["iter_dir"] + str(ieval+optimizer_params["num_init_examples"]))

        grad = determine_gradient(X_stencil, target_stencil, learning_rate, pbounds, optimizer_params["num_inputs"])
        X_new = grad_descent(X_old, grad, step_size, pbounds,
                                  optimizer_params["num_inputs"],
                                  grad_descent_params["num_steps_per_iter"])

        Y_new, avg_powers_new = tdg.run_ifriit_input(grad_descent_params["num_steps_per_iter"],
                                                     X_new, optimizer_params["run_dir"],
                                                     optimizer_params["num_modes"],
                                                     optimizer_params["num_parallel"],
                                                     optimizer_params["hemisphere_symmetric"],
                                                     optimizer_params["run_clean"])
        target_downhill = np.sqrt(np.sum(Y_new**2, axis=0))
        mindex_downhill = np.argmin(target_downhill)
        print("The minimum downhill", np.min(target_downhill), mindex_downhill)

        if target_downhill[mindex_downhill] < target_stencil[mindex_stencil]:
            shutil.rmtree(optimizer_params["run_dir"] + "/" + optimizer_params["iter_dir"] + str(ieval+optimizer_params["num_init_examples"]))
            os.rename(optimizer_params["run_dir"] + "/run_" + str(mindex_downhill),
                      optimizer_params["run_dir"] + "/" + optimizer_params["iter_dir"] + str(ieval+optimizer_params["num_init_examples"]))
            X_old[:,0] = X_new[:,mindex_downhill]
            Y_old[:,0] = Y_new[:,mindex_downhill]
            avg_powers_old = avg_powers_new[mindex_downhill]
        else:
            X_old[:,0] = X_stencil[:,mindex_stencil]
            Y_old[:,0] = Y_stencil[:,mindex_stencil]
            avg_powers_old = avg_powers_stencil[mindex_stencil]

        dataset_optimizer["X_all"] = np.hstack((dataset_optimizer["X_all"], X_old))
        dataset_optimizer["Y_all"] = np.hstack((dataset_optimizer["Y_all"], Y_old))
        dataset_optimizer["avg_powers_all"] = np.hstack((dataset_optimizer["avg_powers_all"], avg_powers_old))

        print("Iteration {} with learn rate {} value:{}".format(ieval, learning_rate, np.sqrt(np.sum(Y_old**2))))
        print(X_old[:,0])

        if (np.sqrt(np.sum(dataset_optimizer["Y_all"][:,-1]**2)) > np.sqrt(np.sum(dataset_optimizer["Y_all"][:,-2]**2))):
            print("Bug! Ascending slope!")
            print(np.sqrt(np.sum(dataset_optimizer["Y_all"][:,-1]**2)), np.sqrt(np.sum(dataset_optimizer["Y_all"][:,-2]**2)))
            break

        if (ieval+1)%10 <= 0.0:
            toc = time.perf_counter()
            print("{:0.4f} seconds".format(toc - tic))
            print(str(ieval+1) + " Bayesian data points added, saving to .nc")
            filename_trainingdata = optimizer_params["run_dir"] + '/' + optimizer_params["trainingdata_filename"]
            nrw.save_training_data(dataset_optimizer["X_all"], dataset_optimizer["Y_all"], dataset_optimizer["avg_powers_all"], filename_trainingdata)
            mindex = np.argmin(np.mean(dataset_optimizer["Y_all"], axis=0))
            print(mindex)
            print(np.sum(dataset_optimizer["Y_all"][:,mindex]))
            print(np.sqrt(np.sum(dataset_optimizer["Y_all"][:,mindex]**2)))
            mindex = np.argmin(np.sqrt(np.sum(dataset_optimizer["Y_all"]**2, axis=0)))
            print(mindex)
            print(np.sum(dataset_optimizer["Y_all"][:,mindex]))
            print(np.sqrt(np.sum(dataset_optimizer["Y_all"][:,mindex]**2)))
    for isten in range(stencil_size):
        try:
            shutil.rmtree(optimizer_params["run_dir"] + "/run_" + str(isten))
        except:
            print("File: " + optimizer_params["run_dir"] + "/run_" + str(isten) + ", already deleted.")
    return dataset_optimizer



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

###################################### Genetic Algorithm ##############################################
# Taken from https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/master/Tutorial%20Project/Example_GeneticAlgorithm.py
# https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    fitness = np.sum(pop*equation_inputs, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, rng, pbounds, num_mutations=1, mutation_amplitude=1.0):
    for idx in range(offspring_crossover.shape[0]):
        gene_mutation_ind_float = rng.random() * (offspring_crossover.shape[1] - 1.0)
        gene_mutation_ind = int(np.round(gene_mutation_ind_float))
        for ind_mut in range(num_mutations):
            random_value = (rng.random() - 0.5) * mutation_amplitude
            offspring_crossover[idx, gene_mutation_ind] += random_value
            if offspring_crossover[idx, gene_mutation_ind] < pbounds[gene_mutation_ind, 0]:
                offspring_crossover[idx, gene_mutation_ind] = pbounds[gene_mutation_ind, 0]
            if offspring_crossover[idx, gene_mutation_ind] > pbounds[gene_mutation_ind, 1]:
                offspring_crossover[idx, gene_mutation_ind] = pbounds[gene_mutation_ind, 1]

    return offspring_crossover

#####################################################################################