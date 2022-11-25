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
    dataset = {}
    dataset["X_all"] = X_all
    dataset["Y_all"] = Y_all
    dataset["avg_powers_all"] = avg_powers_all
    return dataset



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

    pbounds = np.zeros((optimizer_params["num_inputs"], 2))
    pbounds[:,1] = 1.0
    optimizer_params["pbounds"] = pbounds
    return optimizer_params

#################################### Bayesian Optimization ################################################


def wrapper_bayesian_optimisation(dataset, bo_params, opt_params):
    pbounds = {}
    for ii in range(opt_params["num_inputs"]):
        pbounds["x"+str(ii)] = opt_params["pbounds"][ii,:]

    target = -np.sqrt(np.sum(dataset["Y_all"]**2, axis=0)) # Critical to make negative (min not max)

    optimizer, utility = initialize_unknown_func(dataset["X_all"],
                                                      target, pbounds,
                                                      opt_params["num_init_examples"],
                                                      opt_params["num_inputs"])
    print(optimizer.max)

    tic = time.perf_counter()
    for it in range(opt_params["n_iter"]):
        optimizer2 = copy.deepcopy(optimizer)

        X_new = np.zeros((opt_params["num_inputs"], opt_params["num_parallel"]))
        for npar in range(opt_params["num_parallel"]):
            next_point = optimizer2.suggest(utility)
            for ii in range(opt_params["num_inputs"]):
                X_new[ii,npar] = next_point["x"+str(ii)]

            try:
                optimizer2.register(params=next_point, target=bo_params["target_set_undetermined"])
            except:
                print("Broken input!", next_point, target_set_undetermined)

        Y_new, avg_powers_new = tdg.run_ifriit_input(opt_params["num_parallel"],
                                                     X_new, opt_params["run_dir"],
                                                     opt_params["num_modes"],
                                                     opt_params["num_parallel"],
                                                     opt_params["hemisphere_symmetric"],
                                                     opt_params["run_clean"])
        dataset["X_all"] = np.hstack((dataset["X_all"], X_new))
        dataset["Y_all"] = np.hstack((dataset["Y_all"], Y_new))
        dataset["avg_powers_all"] = np.hstack((dataset["avg_powers_all"], avg_powers_new))

        for npar in range(opt_params["num_parallel"]):
            ieval = it * opt_params["num_parallel"] + npar
            os.rename(opt_params["run_dir"] + "/run_" + str(npar),
                      opt_params["run_dir"] + "/" + opt_params["iter_dir"]
                      + str(ieval+opt_params["num_init_examples"]))

            target = -np.sqrt(np.sum(Y_new[:,npar]**2))
            for ii in range(opt_params["num_inputs"]):
                next_point["x"+str(ii)] = X_new[ii,npar]
            try:
                optimizer.register(params=next_point, target=target)
            except:
                print("Broken input!", next_point, target)

        if (it+1)%1 <= 0.0:
            toc = time.perf_counter()
            print("{:0.4f} seconds".format(toc - tic))
            print(str(ieval+1) + " Bayesian data points added, saving to .nc")
            filename_trainingdata = opt_params["run_dir"] + '/' + opt_params["trainingdata_filename"]
            nrw.save_training_data(dataset["X_all"], dataset["Y_all"],
                                   dataset["avg_powers_all"], filename_trainingdata)
            print(optimizer.max)
            mindex = np.argmin(np.mean(dataset["Y_all"], axis=0))
            print(mindex)
            print(np.sum(dataset["Y_all"][:,mindex]))
            print(np.sqrt(np.sum(dataset["Y_all"][:,mindex]**2)))
            mindex = np.argmin(np.sqrt(np.sum(dataset["Y_all"]**2, axis=0)))
            print(mindex)
            print(np.sum(dataset["Y_all"][:,mindex]))
            print(np.sqrt(np.sum(dataset["Y_all"][:,mindex]**2)))
    print(next_point)
    return dataset



def define_bayesian_optimisation_params(target_set_undetermined):
    bo_params = {}
    bo_params["target_set_undetermined"] = target_set_undetermined
    return bo_params


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
    gd_params = {}
    gd_params["learn_exp"] = -1.0
    gd_params["num_steps_per_iter"] = num_steps_per_iter
    return gd_params



def wrapper_gradient_descent(dataset, gd_params, opt_params):
    learning_rate = 10.0**gd_params["learn_exp"]
    step_size = np.array([gd_params["learn_exp"] - 1.0, gd_params["learn_exp"] + 1.0])
    stencil_size = opt_params["num_inputs"] * 2 + 1

    X_old = np.zeros((opt_params["num_inputs"], 1))
    Y_old = np.zeros((opt_params["num_modes"], 1))
    avg_powers_old = np.array([0.0])

    mindex = np.argmin(np.sqrt(np.sum(dataset["Y_all"]**2, axis=0)))
    X_old[:,0] = dataset["X_all"][:, mindex]

    tic = time.perf_counter()
    for ieval in range(opt_params["n_iter"]):

        if (sum(abs(dataset["X_all"][:,-1] - dataset["X_all"][:,-2])) <= 0.0):
            gd_params["learn_exp"] = gd_params["learn_exp"]-0.5
            learning_rate = 10.0**(gd_params["learn_exp"])
            step_size = step_size - 0.5
            print("Reducing step size to: " + str(learning_rate))
            if learning_rate < 1.0e-4:
                print(str(ieval+1) + " Bayesian data points added, saving to .nc")
                print("Early stopping due to repeated results")
                filename_trainingdata = opt_params["run_dir"] + '/' + opt_params["trainingdata_filename"]
                nrw.save_training_data(dataset["X_all"], dataset["Y_all"],
                                       dataset["avg_powers_all"], filename_trainingdata)
                break

        X_stencil = gradient_stencil(X_old, learning_rate, opt_params["pbounds"],
                                     opt_params["num_inputs"], stencil_size)
        Y_stencil, avg_powers_stencil = tdg.run_ifriit_input(stencil_size, X_stencil,
                                                             opt_params["run_dir"],
                                                             opt_params["num_modes"],
                                                             opt_params["num_parallel"],
                                                             opt_params["hemisphere_symmetric"],
                                                             opt_params["run_clean"])
        target_stencil = np.sqrt(np.sum(Y_stencil**2, axis=0))
        mindex_stencil = np.argmin(target_stencil)
        print("The minimum in the stencil", np.min(target_stencil), mindex_stencil)
        print("The previous value was: ", target_stencil[0], 0)
        print(X_stencil[:,0])
        os.rename(opt_params["run_dir"]  + "/run_" + str(mindex_stencil),
                  opt_params["run_dir"] + "/" + opt_params["iter_dir"] 
                  + str(ieval+opt_params["num_init_examples"]))

        grad = determine_gradient(X_stencil, target_stencil, learning_rate,
                                  opt_params["pbounds"], opt_params["num_inputs"])
        X_new = grad_descent(X_old, grad, step_size, opt_params["pbounds"],
                             opt_params["num_inputs"],
                             gd_params["num_steps_per_iter"])

        Y_new, avg_powers_new = tdg.run_ifriit_input(gd_params["num_steps_per_iter"],
                                                     X_new, opt_params["run_dir"],
                                                     opt_params["num_modes"],
                                                     opt_params["num_parallel"],
                                                     opt_params["hemisphere_symmetric"],
                                                     opt_params["run_clean"])
        target_downhill = np.sqrt(np.sum(Y_new**2, axis=0))
        mindex_downhill = np.argmin(target_downhill)
        print("The minimum downhill", np.min(target_downhill), mindex_downhill)

        if target_downhill[mindex_downhill] < target_stencil[mindex_stencil]:
            shutil.rmtree(opt_params["run_dir"] + "/" + opt_params["iter_dir"] 
                          + str(ieval+opt_params["num_init_examples"]))
            os.rename(opt_params["run_dir"] + "/run_" + str(mindex_downhill),
                      opt_params["run_dir"] + "/" + opt_params["iter_dir"] 
                      + str(ieval+opt_params["num_init_examples"]))
            X_old[:,0] = X_new[:,mindex_downhill]
            Y_old[:,0] = Y_new[:,mindex_downhill]
            avg_powers_old = avg_powers_new[mindex_downhill]
        else:
            X_old[:,0] = X_stencil[:,mindex_stencil]
            Y_old[:,0] = Y_stencil[:,mindex_stencil]
            avg_powers_old = avg_powers_stencil[mindex_stencil]

        dataset["X_all"] = np.hstack((dataset["X_all"], X_old))
        dataset["Y_all"] = np.hstack((dataset["Y_all"], Y_old))
        dataset["avg_powers_all"] = np.hstack((dataset["avg_powers_all"], avg_powers_old))

        print("Iteration {} with learn rate {} value:{}".format(ieval, learning_rate, np.sqrt(np.sum(Y_old**2))))
        print(X_old[:,0])

        if (np.sqrt(np.sum(dataset["Y_all"][:,-1]**2)) 
            > np.sqrt(np.sum(dataset["Y_all"][:,-2]**2))):
            print("Bug! Ascending slope!")
            print(np.sqrt(np.sum(dataset["Y_all"][:,-1]**2)), np.sqrt(np.sum(dataset["Y_all"][:,-2]**2)))
            break

        if (ieval+1)%10 <= 0.0:
            toc = time.perf_counter()
            print("{:0.4f} seconds".format(toc - tic))
            print(str(ieval+1) + " Bayesian data points added, saving to .nc")
            filename_trainingdata = optimizer_params["run_dir"] + '/' + optimizer_params["trainingdata_filename"]
            nrw.save_training_data(dataset["X_all"], dataset["Y_all"],
                                   dataset["avg_powers_all"], filename_trainingdata)
            mindex = np.argmin(np.mean(dataset["Y_all"], axis=0))
            print(mindex)
            print(np.sum(dataset["Y_all"][:,mindex]))
            print(np.sqrt(np.sum(dataset["Y_all"][:,mindex]**2)))
            mindex = np.argmin(np.sqrt(np.sum(dataset["Y_all"]**2, axis=0)))
            print(mindex)
            print(np.sum(dataset["Y_all"][:,mindex]))
            print(np.sqrt(np.sum(dataset["Y_all"][:,mindex]**2)))
    for isten in range(stencil_size):
        try:
            shutil.rmtree(optimizer_params["run_dir"] + "/run_" + str(isten))
        except:
            print("File: " + optimizer_params["run_dir"] + "/run_" + str(isten) + ", already deleted.")
    return dataset



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
    return ga_params



def wrapper_genetic_algorithm(dataset, ga_params, opt_params):
    #Creating the initial population.
    X_pop = opt_params["random_generator"].random((opt_params["num_inputs"], ga_params["initial_pop_size"]))

    best_outputs = []
    tic = time.perf_counter()
    for generation in range(opt_params["n_iter"]-1):
        print("Generation : ", generation)
        # Measuring the fitness of each chromosome in the population.
        Y_pop, avg_powers_pop = tdg.run_ifriit_input(ga_params["initial_pop_size"] , X_pop,
                                                     opt_params["run_dir"],
                                                     opt_params["num_modes"],
                                                     opt_params["num_parallel"],
                                                     opt_params["hemisphere_symmetric"],
                                                     opt_params["run_clean"])
        dataset["X_all"] = np.hstack((dataset["X_all"], X_pop))
        dataset["Y_all"] = np.hstack((dataset["Y_all"], Y_pop))
        dataset["avg_powers_all"] = np.hstack((dataset["avg_powers_all"], avg_powers_pop))
        for irun in range(ga_params["initial_pop_size"] ):
            os.rename(opt_params["run_dir"] + "/run_" + str(irun), opt_params["run_dir"]
                      + "/" + opt_params["iter_dir"] + str(irun+ga_params["initial_pop_size"] *generation))

        fitness_pop = -np.sqrt(np.sum(Y_pop**2, axis=0))
        mindex_pop = np.argmax(fitness_pop)

        if (generation+1)%10 <= 0.0:
            toc = time.perf_counter()
            print("{:0.4f} seconds".format(toc - tic))
            print(str(generation+1) + " genetic algorithm data points added, saving to .nc")
            filename_trainingdata = opt_params["run_dir"] + '/' + opt_params["trainingdata_filename"]
            nrw.save_training_data(dataset["X_all"], dataset["Y_all"],
                                   dataset["avg_powers_all"], filename_trainingdata)
            mindex = np.argmin(np.mean(dataset["Y_all"], axis=0))
            print(mindex)
            print(np.sum(dataset["Y_all"][:,mindex]))
            print(np.sqrt(np.sum(dataset["Y_all"][:,mindex]**2)))
            mindex = np.argmin(np.sqrt(np.sum(dataset["Y_all"]**2, axis=0)))
            print(mindex)
            print(np.sum(dataset["Y_all"][:,mindex]))
            print(np.sqrt(np.sum(dataset["Y_all"][:,mindex]**2)))

        best_outputs.append(np.max(fitness_pop))
        # The best result in the current iteration.
        print("Best result : ", best_outputs[generation])

        # Selecting the best parents in the population for mating.
        parents = select_mating_pool(X_pop.T, fitness_pop, ga_params["num_parents_mating"])

        # Generating next generation using crossover.
        offspring_crossover = crossover(parents,
                                             offspring_size=(ga_params["initial_pop_size"]
                                                             - ga_params["num_parents_mating"],
                                                             opt_params["num_inputs"]))

        # Adding some variations to the offspring using mutation.
        offspring_mutation = mutation(offspring_crossover, opt_params["random_generator"],
                                           opt_params["pbounds"], num_mutations=2)

        # Creating the new population based on the parents and offspring.
        X_pop[:,0:ga_params["num_parents_mating"]] = parents.T
        X_pop[:,ga_params["num_parents_mating"]:] = offspring_mutation.T
    print("Generation : ", generation+1)
    # Getting the best solution after iterating finishing all generations.
    #At first, the fitness is calculated for each solution in the final generation.
    Y_pop, avg_powers_pop = tdg.run_ifriit_input(ga_params["initial_pop_size"], X_pop,
                                                 opt_params["run_dir"],
                                                 opt_params["num_modes"],
                                                 opt_params["num_parallel"],
                                                 opt_params["hemisphere_symmetric"],
                                                 opt_params["run_clean"])
    dataset["X_all"] = np.hstack((dataset["X_all"], X_pop))
    dataset["Y_all"] = np.hstack((dataset["Y_all"], Y_pop))
    dataset["avg_powers_all"] = np.hstack((dataset["avg_powers_all"], avg_powers_pop))
    for irun in range(ga_params["initial_pop_size"] ):
        os.rename(opt_params["run_dir"] + "/run_" + str(irun),
                  opt_params["run_dir"] + "/" + opt_params["iter_dir"]
                  + str(irun+ga_params["initial_pop_size"] *(generation+1)))

    # Then return the index of that solution corresponding to the best fitness.
    fitness_pop = -np.sqrt(np.sum(Y_pop**2, axis=0))
    mindex_pop = np.argmax(fitness_pop)

    print("Best solution : ", X_pop[:, mindex_pop])
    print("Best solution fitness : ", fitness_pop[mindex_pop])

    toc = time.perf_counter()
    print("{:0.4f} seconds".format(toc - tic))
    print(str(generation+1) + " genetic algorithm data points added, saving to .nc")
    filename_trainingdata = opt_params["run_dir"] + '/' + opt_params["trainingdata_filename"]
    nrw.save_training_data(dataset["X_all"], dataset["Y_all"],
                           dataset["avg_powers_all"], filename_trainingdata)
    mindex = np.argmin(np.mean(dataset["Y_all"], axis=0))
    print(mindex)
    print(np.sum(dataset["Y_all"][:,mindex]))
    print(np.sqrt(np.sum(dataset["Y_all"][:,mindex]**2)))
    mindex = np.argmin(np.sqrt(np.sum(dataset["Y_all"]**2, axis=0)))
    print(mindex)
    print(np.sum(dataset["Y_all"][:,mindex]))
    print(np.sqrt(np.sum(dataset["Y_all"][:,mindex]**2)))

    #plt.plot(best_outputs)
    #plt.xlabel("Iteration")
    #plt.ylabel("Fitness")
    #plt.show()



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
