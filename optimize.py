import training_data_generation as tdg
import netcdf_read_write as nrw
import utils_optimizers as uopt
import utils_deck_generation as idg
import numpy as np
import sys
import time
import os
import shutil
import copy


def wrapper_bayesian_optimisation(dataset, bo_params, opt_params):
    pbounds = {}
    for ii in range(opt_params["num_optimization_params"]):
        pbounds["x"+str(ii)] = opt_params["pbounds"][ii,:]

    target = uopt.fitness_function(dataset, opt_params)

    optimizer, utility = uopt.initialize_unknown_func(dataset["input_parameters"],
                                                      target, pbounds,
                                                      opt_params["num_init_examples"],
                                                      opt_params["num_optimization_params"])
    print("Starting Bayesian optimizer")

    tic = time.perf_counter()
    for it in range(opt_params["n_iter"]):
        optimizer2 = copy.deepcopy(optimizer)

        X_new = np.zeros((bo_params["ifriit_runs_per_iteration"], opt_params["num_optimization_params"]))
        old_max_eval = dataset["num_evaluated"]

        for npar in range(bo_params["ifriit_runs_per_iteration"]):
            ieval = old_max_eval + npar
            next_point = optimizer2.suggest(utility)
            for ii in range(opt_params["num_optimization_params"]):
                X_new[npar,ii] = next_point["x"+str(ii)]

            try:
                optimizer2.register(params=next_point, target=bo_params["target_set_undetermined"])
            except:
                boolean_array = np.isclose(X_new[npar,0], dataset["input_parameters"][:,0], rtol=1.0e-5)
                print("Broken input! Likely degererate, index: ", ieval)
                print("First parameter degenerate with: ", np.where(boolean_array))

                print("Trying random mutation:")
                for_mutation = X_new[npar,:].reshape((1,opt_params["num_optimization_params"]))
                print("old: ", for_mutation)
                after_mutation = uopt.mutation(for_mutation, opt_params["random_generator"],
                                           opt_params["pbounds"],
                                           num_mutations=bo_params["num_mutations"],
                                           mutation_amplitude=bo_params["mutation_amplitude"])
                print("new: ", after_mutation)
                X_new[npar,:] = after_mutation[0,:]

        old_max_eval = dataset["num_evaluated"]
        dataset = uopt.run_ifriit_input(bo_params["ifriit_runs_per_iteration"], X_new, opt_params)

        target = uopt.fitness_function(dataset, opt_params)
        for npar in range(bo_params["ifriit_runs_per_iteration"]):
            ieval = old_max_eval + npar

            for ii in range(opt_params["num_optimization_params"]):
                next_point["x"+str(ii)] = dataset["input_parameters"][ieval,ii]

            try:
                optimizer.register(params=next_point, target=target[ieval])
            except:
                print("Broken input!", next_point, target[ieval])

        if (it+1)%opt_params["printout_iteration_skip"] <= 0.0:
            uopt.printout_optimizer_iteration(tic, dataset, opt_params)
    return dataset



def wrapper_gradient_descent(dataset, gd_params, opt_params):
    learning_rate = 10.0**gd_params["learn_exp"]
    step_size = np.array([gd_params["learn_exp"] - 1.0, gd_params["learn_exp"] + 1.0])
    stencil_size = opt_params["num_inputs"] * 2 + 1

    X_old = np.zeros((opt_params["num_inputs"], 1))
    Y_old = np.zeros((opt_params["num_modes"], 1))
    avg_powers_old = np.array([0.0])

    fitness_pop = -uopt.fitness_function(dataset["Y_all"], dataset["avg_powers_all"],
                                         opt_params)
    mindex = np.argmin(fitness_pop)
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

        X_stencil = uopt.gradient_stencil(X_old, learning_rate, opt_params["pbounds"],
                                     opt_params["num_inputs"], stencil_size)
        Y_stencil, avg_powers_stencil = tdg.run_ifriit_input(stencil_size, X_stencil,
                                                             opt_params["run_dir"],
                                                             opt_params["num_modes"],
                                                             opt_params["num_parallel"],
                                                             opt_params["hemisphere_symmetric"],
                                                             opt_params["run_clean"])
        target_stencil = -uopt.fitness_function(Y_stencil, avg_powers_stencil, opt_params)
        mindex_stencil = np.argmin(target_stencil)
        print("The minimum in the stencil", np.min(target_stencil), mindex_stencil)
        print("The previous value was: ", target_stencil[0], 0)
        print(X_stencil[:,0])
        os.rename(opt_params["run_dir"]  + "/run_" + str(mindex_stencil),
                  opt_params["run_dir"] + "/" + opt_params["iter_dir"] 
                  + str(ieval+opt_params["num_init_examples"]))

        grad = uopt.determine_gradient(X_stencil, target_stencil, learning_rate,
                                  opt_params["pbounds"], opt_params["num_inputs"])
        grad = grad / np.sum(np.abs(grad))
        X_new = uopt.grad_descent(X_old, grad, step_size, opt_params["pbounds"],
                             opt_params["num_inputs"],
                             gd_params["num_steps_per_iter"])

        Y_new, avg_powers_new = tdg.run_ifriit_input(gd_params["num_steps_per_iter"],
                                                     X_new, opt_params["run_dir"],
                                                     opt_params["num_modes"],
                                                     opt_params["num_parallel"],
                                                     opt_params["hemisphere_symmetric"],
                                                     opt_params["run_clean"])
        target_downhill = -uopt.fitness_function(Y_new, avg_powers_new, opt_params)
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

        fit_func_all = -uopt.fitness_function(dataset["Y_all"], dataset["avg_powers_all"],
                                              opt_params)

        if fit_func_all[-1] > fit_func_all[-2]:
            print("Bug! Ascending slope!")
            print(fit_func_all[-1], fit_func_all[-2])
            break

        if (ieval+1)%1 <= 0.0:
            toc = time.perf_counter()
            print("{:0.4f} seconds".format(toc - tic))
            print(str(ieval+1) + " data points added, saving to .nc")
            filename_trainingdata = opt_params["run_dir"] + '/' + opt_params["trainingdata_filename"]
            nrw.save_training_data(dataset["X_all"], dataset["Y_all"],
                                   dataset["avg_powers_all"], filename_trainingdata)
            mindex = np.argmin(fit_func_all)
            print(mindex)
            print(np.sum(fit_func_all[mindex]))
            print(np.sum(dataset["Y_all"][:,mindex]))
            print(np.sqrt(np.sum(dataset["Y_all"][:,mindex]**2)))
    for isten in range(stencil_size):
        try:
            shutil.rmtree(opt_params["run_dir"] + "/run_" + str(isten))
        except:
            print("File: " + opt_params["run_dir"] + "/run_" + str(isten) + ", already deleted.")
    return dataset



def wrapper_genetic_algorithm(dataset, ga_params, opt_params):
    X_pop = dataset["input_parameters"]

    generation = -1
    tic = time.perf_counter()
    for generation in range(opt_params["n_iter"]-1):
        print("Generation : ", generation)
        target = uopt.fitness_function(dataset, opt_params)

        # Selecting the best parents in the population for mating.
        parents = uopt.select_mating_pool(X_pop, target, ga_params["num_parents_mating"])

        # Generating next generation using crossover.
        offspring_crossover = uopt.crossover(parents,
                                             offspring_size=(ga_params["initial_pop_size"]
                                                             - ga_params["num_parents_mating"],
                                                             opt_params["num_optimization_params"]))

        # Adding some variations to the offspring using mutation.
        offspring_mutation = uopt.mutation(offspring_crossover, opt_params["random_generator"],
                                           opt_params["pbounds"],
                                           num_mutations=ga_params["num_mutations"],
                                           mutation_amplitude=ga_params["mutation_amplitude"])

        # Creating the new population based on the parents and offspring.
        X_pop[0:ga_params["num_parents_mating"],:] = parents
        X_pop[ga_params["num_parents_mating"]:,:] = offspring_mutation

        dataset = uopt.run_ifriit_input(ga_params["initial_pop_size"], X_pop, opt_params)

        if (generation+1)%opt_params["printout_iteration_skip"] <= 0.0:
            print(str((generation+1) * ga_params["initial_pop_size"]) + " data points added")
            uopt.printout_optimizer_iteration(tic, dataset, opt_params)
    
    return dataset



def main(argv):
    """ 
                       dir         iex  init_type  bayes_opt grad_descent random_sampler random_seed  dir
    python optimize.py Data_output 100   0-2 10     0-1 10     0-1  10        0           12345      Data_input
    python optimize.py Data_output 100 2 10 1 10 1 10 0 12345 Data_input
    index:                       1  2  3 4  5  6 7 8  9   10      11
    """
    #
    data_init_type = int(argv[3])
    input_dir = argv[11]
    output_dir = argv[1]
    num_examples = int(argv[2])
    #random_seed = int(argv[10])
    #random_sampling = int(argv[9])

    sys_params = tdg.define_system_params(output_dir)

    if data_init_type == 1: # Generate new initialization dataset
        print("Generating data!")

        dataset, dataset_params, sys_params, facility_spec = tdg.main((None, sys_params["root_dir"], num_examples))

    elif data_init_type == 2: # Genetic algorithm
        print("Using a genetic algorithm!")
        ga_n_iter = int(argv[4])
        initial_pop_size = num_examples
        dataset, dataset_params, sys_params, facility_spec = tdg.main((None, sys_params["root_dir"], initial_pop_size))

        num_parents_mating = int(initial_pop_size / 10.0)
        if (num_parents_mating % 2) != 0:
            num_parents_mating -=1
        if num_parents_mating < 2:
            num_parents_mating = 2
        num_init_examples = 0 # genetic algorithm generates its own initial data

        opt_params = uopt.define_optimizer_parameters(output_dir, dataset_params["num_input_params"],
                                                     num_init_examples,
                                                     ga_n_iter, sys_params["num_parallel_ifriits"],
                                                     dataset_params["random_seed"], facility_spec)
        num_mutations = int(opt_params["num_optimization_params"] / 2)

        ga_params = uopt.define_genetic_algorithm_params(initial_pop_size, num_parents_mating, num_mutations)
        dataset = wrapper_genetic_algorithm(dataset, ga_params, opt_params)

    elif data_init_type == 0:
        print("Importing pre-generated data!")
        # copy across dataset_params and facility_spec
        shutil.copyfile(input_dir + "/" + sys_params["dataset_params_filename"],
                        output_dir + "/" + sys_params["dataset_params_filename"])
        shutil.copyfile(input_dir + "/" + sys_params["facility_spec_filename"],
                        output_dir + "/" + sys_params["facility_spec_filename"])
        shutil.copyfile(input_dir + "/" + sys_params["trainingdata_filename"],
                        output_dir + "/" + sys_params["trainingdata_filename"])
        shutil.copyfile(input_dir + "/" + sys_params["deck_gen_params_filename"],
                        output_dir + "/" + sys_params["deck_gen_params_filename"])
    else:
        print("")
        sys.exit("Dataset not properly specified")

    print("Importing data!")
    dataset_params = nrw.read_general_netcdf(sys_params["root_dir"] + "/" + sys_params["dataset_params_filename"])
    facility_spec = nrw.read_general_netcdf(sys_params["root_dir"] + "/" + sys_params["facility_spec_filename"])
    dataset = nrw.read_general_netcdf(sys_params["root_dir"] + "/" + sys_params["trainingdata_filename"])
    num_init_examples = dataset["num_evaluated"]

    use_bayesian_optimization = bool(int(argv[5]))
    if use_bayesian_optimization: # Bayesian optimization
        bo_n_iter = int(argv[6])
        opt_params = uopt.define_optimizer_parameters(output_dir, dataset_params["num_input_params"],
                                                     num_init_examples,
                                                     bo_n_iter, sys_params["num_parallel_ifriits"],
                                                     dataset_params["random_seed"], facility_spec)
        ifriit_runs_per_bo_iteration = sys_params["num_parallel_ifriits"]

        target = uopt.fitness_function(dataset, opt_params)
        target_set_undetermined = np.mean(target) / 2.0 # half mean for all undetermined BO values
        num_mutations = int(opt_params["num_optimization_params"] / 2)
        bo_params = uopt.define_bayesian_optimisation_params(ifriit_runs_per_bo_iteration, target_set_undetermined, num_mutations)
        dataset = wrapper_bayesian_optimisation(dataset, bo_params, opt_params)
        num_init_examples = dataset["num_evaluated"]

    use_gradient_descent = bool(int(argv[7]))
    if use_gradient_descent: # Gradient descent
        print("Using gradient descent!")
        gd_n_iter = int(argv[8])
        opt_params = uopt.define_optimizer_parameters(output_dir, dataset_params["num_input_params"],
                                                     num_init_examples,
                                                     gd_n_iter, sys_params["num_parallel_ifriits"],
                                                     dataset_params["random_seed"], facility_spec)

        gd_params = uopt.define_gradient_descent_params(num_parallel)
        dataset = wrapper_gradient_descent(dataset, gd_params, opt_params)
        num_init_examples = dataset["num_evaluated"]

    return


if __name__ == "__main__":
    main(sys.argv)
    print("terminated normally")
