# code copied from https://github.com/fmfn/BayesianOptimization edited by Duncan Barlow
#from bayes_opt import BayesianOptimization, UtilityFunction
import numpy as np
import training_data_generation as tdg
import netcdf_read_write as nrw
import utils_deck_generation as idg
import time
import sys



def define_optimizer_dataset(X_all, Y_all, avg_powers_all):
    dataset = {}
    dataset["X_all"] = X_all
    dataset["Y_all"] = Y_all
    dataset["avg_powers_all"] = avg_powers_all
    return dataset



def define_optimizer_parameters(run_dir, num_optimization_params,
                                num_init_examples, n_iter,
                                random_seed, facility_spec, run_clean):
    optimizer_params = {}
    optimizer_params["run_dir"] = run_dir
    optimizer_params["num_optimization_params"] = num_optimization_params
    optimizer_params["num_init_examples"] = num_init_examples
    optimizer_params["n_iter"] = n_iter
    optimizer_params["run_clean"] = run_clean
    optimizer_params["random_generator"] = np.random.default_rng(random_seed)
    optimizer_params["fitness_desired_power_per_steradian"] = facility_spec['nbeams'] \
        * facility_spec['default_power'] * 1.0e12 / (4.0 * np.pi)
    optimizer_params["fitness_desired_pressure_mbar"] = 60.0
    optimizer_params["fitness_desired_rms"] = 0.03
    optimizer_params["fitness_norm_factor"] = 5.0
    optimizer_params["printout_iteration_skip"] = 1
    optimizer_params["profile_for_fitness"] = 0

    pbounds = np.zeros((optimizer_params["num_optimization_params"], 2))
    pbounds[:,1] = 1.0
    optimizer_params["pbounds"] = pbounds
    return optimizer_params



def fitness_function(dataset, opt_params):
    target_rms = opt_params["fitness_desired_rms"]
    if opt_params["profile_for_fitness"] == 0:
        target_flux = opt_params["fitness_desired_power_per_steradian"]
    else:
        target_flux = opt_params["fitness_desired_pressure_mbar"]
    norm_factor = opt_params["fitness_norm_factor"]
    profile_for_fitness = opt_params["profile_for_fitness"]

    rms = np.sqrt((dataset["rms"][:,0]**2 + dataset["rms"][:,1]**2) / 2.0)
    avg_flux = dataset["avg_flux"][:,profile_for_fitness]

    maxi_func = np.exp(-rms/target_rms + (avg_flux / target_flux)**6) * norm_factor
    return maxi_func



def run_ifriit_input(num_new_examples, X_all, opt_params):
    sys_params = tdg.define_system_params(opt_params["run_dir"])
    sys_params["run_clean"] = opt_params["run_clean"] # Create new run files

    dataset, dataset_params, deck_gen_params, facility_spec = idg.load_data_dicts_from_file(sys_params)
    dataset_params["num_examples"] = dataset["num_evaluated"] + num_new_examples

    num_evaluated = dataset["num_evaluated"]
    dataset = expand_dataset(dataset, dataset_params, num_evaluated)
    deck_gen_params = expand_deck_gen_params(deck_gen_params, dataset_params, facility_spec, num_evaluated)
    dataset["input_parameters"][dataset["num_evaluated"]:,:] = X_all

    deck_gen_params = idg.create_run_files(dataset, deck_gen_params, dataset_params, sys_params, facility_spec)

    tdg.generate_training_data(dataset, dataset_params, sys_params, facility_spec)
    return dataset



def expand_dataset(dataset_small, dataset_params, num_evaluated):
    dataset_big = tdg.define_dataset(dataset_params)
    dataset_big = expand_dict(dataset_big, dataset_small, num_evaluated)
    return dataset_big



def expand_deck_gen_params(deck_gen_params_small, dataset_params, facility_spec, num_evaluated):
    deck_gen_params_big = idg.define_deck_generation_params(dataset_params, facility_spec)
    deck_gen_params_big = expand_dict(deck_gen_params_big, deck_gen_params_small, num_evaluated)
    return deck_gen_params_big



def expand_dict(big_dictionary, small_dictionary, old_size):
    prohibited_list = small_dictionary["non_expand_keys"]
    for key, item in big_dictionary.items():
        dims = np.shape(item)
        total_dims = np.shape(dims)[0]
        if any(x in key for x in prohibited_list):#(key == "num_evaluated"):
            big_dictionary[key] = small_dictionary[key]
        else:
            if total_dims == 3:
                big_dictionary[key][:old_size,:,:] = small_dictionary[key][:old_size,:,:]
            if total_dims == 2:
                big_dictionary[key][:old_size,:] = small_dictionary[key][:old_size,:]
            if total_dims == 1:
                big_dictionary[key][:old_size] = small_dictionary[key][:old_size]
    small_dictionary.clear()
    return big_dictionary



def printout_optimizer_iteration(tic, dataset, opt_params):
    toc = time.perf_counter()
    print("{:0.4f} seconds".format(toc - tic))

    target = fitness_function(dataset, opt_params)
    maxdex = np.argmax(target)
    print(maxdex)
    print(target[maxdex])
    print(dataset["rms"][maxdex,:])

#################################### Bayesian Optimization #############################################

def define_bayesian_optimisation_params(ifriit_runs_per_iteration, target_set_undetermined, num_mutations):
    bo_params = {}
    bo_params["target_set_undetermined"] = target_set_undetermined
    bo_params["num_mutations"] = num_mutations
    bo_params["mutation_amplitude"] = 0.25
    bo_params["ifriit_runs_per_iteration"] = ifriit_runs_per_iteration

    return bo_params


def initialize_unknown_func(input_data, target, pbounds, init_points, num_inputs):

    sys.exit("Bayesian optimization intentionally removed! Find this line to re-add")
    optimizer = {}#BayesianOptimization(f=None,pbounds=pbounds,random_state=1)
    utility = {}#UtilityFunction(kind = "ucb", kappa = 2.5, xi = 0.0)

    # initial data points
    params = {}
    for ieval in range(init_points):
        # put data in dict
        for ii in range(num_inputs):
            params["x"+str(ii)] = input_data[ieval, ii]
        # add to optimizer
        try:
            optimizer.register(params = params, target = target[ieval])
        except:
            print("Broken input!", input_data[ieval, :])
        if ieval%100 <= 0.0:
            print(str(ieval) + " initialization data points added")
    print(str(ieval) + " initialization data points added")

    return optimizer, utility

######################################## Gradient Descent ############################################

def define_gradient_ascent_params(num_steps_per_iter):
    gd_params = {}
    gd_params["learn_exp"] = -1.0
    gd_params["num_steps_per_iter"] = num_steps_per_iter
    return gd_params



def gradient_stencil(X_new, learning_rate, pbounds, num_inputs, stencil_size):
    X_stencil = np.zeros((stencil_size, num_inputs))

    counter = 0
    for ii in range(num_inputs):
        X_stencil[counter,:] = X_new[0,:]
        X_stencil[counter,ii] = X_new[0,ii] - learning_rate
        if (X_stencil[counter,ii] < pbounds[ii,0]):
            X_stencil[counter,ii] = pbounds[ii,0] # to avoid stencil leaving domain
        counter += 1
        X_stencil[counter,:] = X_new[0,:]
        X_stencil[counter,ii] = X_new[0,ii] + learning_rate
        if (X_stencil[counter,ii] > pbounds[ii,1]):
            X_stencil[counter,ii] = pbounds[ii,1] # to avoid stencil leaving domain
        counter += 1

    return X_stencil



def determine_gradient(X_stencil, target_stencil, target, learning_rate, pbounds, num_inputs):

    grad = np.zeros(num_inputs)
    f_centre = target
    counter = 0
    for ii in range(num_inputs):

        centred_diff = True
        forward_diff = False
        backward_diff = False

        if (X_stencil[counter,ii] < pbounds[ii,0]):
            centred_diff = False
            forward_diff = True
        else:
            f_minus = target_stencil[counter]
        counter += 1

        if (X_stencil[counter,ii] > pbounds[ii,1]):
            centred_diff = False
            backward_diff = True
        else:
            f_plus = target_stencil[counter]
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



def grad_ascent(X_old, grad, step_size, pbounds, num_inputs, num_steps_per_iter):

    learning_rates = np.logspace(step_size[0], step_size[1], num_steps_per_iter)
    X_new = np.zeros((num_steps_per_iter, num_inputs))
    for ieval in range(num_steps_per_iter):
        X_new[ieval,:] = X_old[0,:]
        for ii in range(num_inputs):
            X_new[ieval,ii] = X_old[0,ii] + learning_rates[ieval] * grad[ii]
            if (X_new[ieval,ii] < pbounds[ii,0]):
                X_new[ieval,ii] = pbounds[ii,0]
            elif (X_new[ieval,ii] > pbounds[ii,1]):
                X_new[ieval,ii] = pbounds[ii,1]

    return X_new

###################################### Genetic Algorithm ##############################################
# Taken from https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/master/Tutorial%20Project/Example_GeneticAlgorithm.py
# https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6

def define_genetic_algorithm_params(init_points, num_parents_mating, num_mutations):
    ga_params = {}
    ga_params["num_parents_mating"] = num_parents_mating
    ga_params["initial_pop_size"] = init_points
    ga_params["num_mutations"] = num_mutations
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
