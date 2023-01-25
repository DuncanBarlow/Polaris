# code copied from https://github.com/fmfn/BayesianOptimization edited by Duncan Barlow
from bayes_opt import BayesianOptimization, UtilityFunction
import numpy as np
import training_data_generation as tdg
import netcdf_read_write as nrw


def define_optimizer_dataset(X_all, Y_all, avg_powers_all):
    dataset = {}
    dataset["X_all"] = X_all
    dataset["Y_all"] = Y_all
    dataset["avg_powers_all"] = avg_powers_all
    return dataset



def define_optimizer_parameters(run_dir, num_inputs, num_modes,
                                num_init_examples, n_iter, num_parallel, random_seed):
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
    optimizer_params["random_generator"] = np.random.default_rng(random_seed)

    pbounds = np.zeros((optimizer_params["num_inputs"], 2))
    pbounds[:,1] = 1.0
    optimizer_params["pbounds"] = pbounds
    return optimizer_params

#################################### Bayesian Optimization ################################################

def define_bayesian_optimisation_params(target_set_undetermined, initial_mean_power):
    bo_params = {}
    bo_params["target_set_undetermined"] = bayesian_change_min2max(target_set_undetermined,
                                                                   initial_mean_power,
                                                                   initial_mean_power)
    bo_params["initial_mean_power"] = initial_mean_power
    return bo_params



def bayesian_change_min2max(Y, avg_power, initial_mean_power):
    maxi_func = np.exp(-Y/0.03) * avg_power/initial_mean_power
    return maxi_func



def initialize_unknown_func(input_data, target, pbounds, init_points, num_inputs):

    optimizer = BayesianOptimization(
      f=None,
      pbounds=pbounds,
      random_state=1,
    )

    utility = UtilityFunction(kind = "ucb", kappa = 2.5, xi = 0.0)

    # initial data points
    params = {}
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

    return optimizer, utility

######################################## Gradient Descent ############################################

def define_gradient_descent_params(num_steps_per_iter):
    gd_params = {}
    gd_params["learn_exp"] = -1.0
    gd_params["num_steps_per_iter"] = num_steps_per_iter
    return gd_params



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

def define_genetic_algorithm_params(init_points, num_parents_mating):
    ga_params = {}
    ga_params["num_parents_mating"] = num_parents_mating
    ga_params["initial_pop_size"] = init_points
    ga_params["num_mutations"] = 8
    ga_params["mutation_amplitude"] = 0.25 # multiplier for standard normal distribution
    return ga_params



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
        for ind_mut in range(num_mutations):
            gene_mutation_ind = int(np.round(rng.random()
                                * (offspring_crossover.shape[1] - 1.0)))
            random_value = rng.standard_normal() * mutation_amplitude
            offspring_crossover[idx, gene_mutation_ind] += random_value
            if offspring_crossover[idx, gene_mutation_ind] < pbounds[gene_mutation_ind, 0]:
                offspring_crossover[idx, gene_mutation_ind] = pbounds[gene_mutation_ind, 0]
            if offspring_crossover[idx, gene_mutation_ind] > pbounds[gene_mutation_ind, 1]:
                offspring_crossover[idx, gene_mutation_ind] = pbounds[gene_mutation_ind, 1]

    return offspring_crossover

#####################################################################################
