import training_data_generation as tdg
import netcdf_read_write as nrw
import utils_optimizers as uopt
import numpy as np
import sys
import time
import os
import shutil
import copy


def wrapper_bayesian_optimisation(dataset, bo_params, opt_params):
    pbounds = {}
    for ii in range(opt_params["num_inputs"]):
        pbounds["x"+str(ii)] = opt_params["pbounds"][ii,:]

    # Critical to make negative (min not max)
    target = uopt.bayesian_change_min2max(np.sqrt(np.sum(dataset["Y_all"]**2, axis=0)),
                                          dataset["avg_powers_all"],
                                          bo_params["initial_mean_power"])

    optimizer, utility = uopt.initialize_unknown_func(dataset["X_all"],
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
                print("Broken input!", next_point, bo_params["target_set_undetermined"])

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

            target = uopt.bayesian_change_min2max(np.sqrt(np.sum(Y_new[:,npar]**2)),
                                                  avg_powers_new[npar],
                                                  bo_params["initial_mean_power"])
            for ii in range(opt_params["num_inputs"]):
                next_point["x"+str(ii)] = X_new[ii,npar]
            try:
                optimizer.register(params=next_point, target=target)
            except:
                print("Broken input!", next_point, target)

        if (it+1)%1 <= 0.0:
            toc = time.perf_counter()
            print("{:0.4f} seconds".format(toc - tic))
            print(str(ieval+1) + " data points added, saving to .nc")
            filename_trainingdata = opt_params["run_dir"] + '/' + opt_params["trainingdata_filename"]
            nrw.save_training_data(dataset["X_all"], dataset["Y_all"],
                                   dataset["avg_powers_all"], filename_trainingdata)
            print(optimizer.max)

            fit_func_all = uopt.bayesian_change_min2max(np.sqrt(np.sum(dataset["Y_all"]**2, axis=0)),
                                                        dataset["avg_powers_all"],
                                                        bo_params["initial_mean_power"])
            mindex = np.argmax(fit_func_all)
            print(mindex)
            print(np.sum(fit_func_all[mindex]))
            print(np.sum(dataset["Y_all"][:,mindex]))
            print(np.sqrt(np.sum(dataset["Y_all"][:,mindex]**2)))
            mindex = np.argmin(np.mean(dataset["Y_all"], axis=0))
            print(mindex)
            print(np.sum(fit_func_all[mindex]))
            print(np.sum(dataset["Y_all"][:,mindex]))
            print(np.sqrt(np.sum(dataset["Y_all"][:,mindex]**2)))
            mindex = np.argmin(np.sqrt(np.sum(dataset["Y_all"]**2, axis=0)))
            print(mindex)
            print(np.sum(fit_func_all[mindex]))
            print(np.sum(dataset["Y_all"][:,mindex]))
            print(np.sqrt(np.sum(dataset["Y_all"][:,mindex]**2)))
    print(next_point)
    return dataset



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

        X_stencil = uopt.gradient_stencil(X_old, learning_rate, opt_params["pbounds"],
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

        grad = uopt.determine_gradient(X_stencil, target_stencil, learning_rate,
                                  opt_params["pbounds"], opt_params["num_inputs"])
        X_new = uopt.grad_descent(X_old, grad, step_size, opt_params["pbounds"],
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

        if (ieval+1)%1 <= 0.0:
            toc = time.perf_counter()
            print("{:0.4f} seconds".format(toc - tic))
            print(str(ieval+1) + " data points added, saving to .nc")
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
    for isten in range(stencil_size):
        try:
            shutil.rmtree(opt_params["run_dir"] + "/run_" + str(isten))
        except:
            print("File: " + opt_params["run_dir"] + "/run_" + str(isten) + ", already deleted.")
    return dataset



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

        if (generation+1)%1 <= 0.0:
            toc = time.perf_counter()
            print("{:0.4f} seconds".format(toc - tic))
            print(str((generation+1) * ga_params["initial_pop_size"]) + " data points added, saving to .nc")
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
        parents = uopt.select_mating_pool(X_pop.T, fitness_pop, ga_params["num_parents_mating"])

        # Generating next generation using crossover.
        offspring_crossover = uopt.crossover(parents,
                                             offspring_size=(ga_params["initial_pop_size"]
                                                             - ga_params["num_parents_mating"],
                                                             opt_params["num_inputs"]))

        # Adding some variations to the offspring using mutation.
        offspring_mutation = uopt.mutation(offspring_crossover, opt_params["random_generator"],
                                           opt_params["pbounds"],
                                           num_mutations=ga_params["num_mutations"],
                                           mutation_amplitude=ga_params["mutation_amplitude"])

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
    if (data_init_type == 0):
        input_dir = argv[11]
    else:
        input_dir = argv[1]
    output_dir = argv[1]
    num_examples = int(argv[2])
    trainingdata_filename = "flipped_training_data_and_labels.nc"
    iter_dir = "iter_"
    num_modes = 30
    num_parallel = 12
    hemisphere_symmetric = True
    random_seed = int(argv[10])
    random_sampling = int(argv[9])
    run_clean = True

    sys_params = tdg.define_system_params(input_dir)
    sys_params["run_clean"] = run_clean

    if data_init_type == 1: # Generate new initialization dataset
        print("Generating data!")
        sys_params["root_dir"] = output_dir

        dataset_params, facility_spec = tdg.define_dataset_params(num_examples, random_sampling=random_sampling, random_seed=random_seed)
        dataset_params["hemisphere_symmetric"] = hemisphere_symmetric
        dataset_params["run_clean"] = run_clean

        tdg.generate_training_data(dataset_params, sys_params, facility_spec)
        # choose test data set
        X_all, Y_all, avg_powers_all = nrw.import_training_data_reversed(sys_params, num_modes)
        filename_trainingdata = output_dir + '/' + trainingdata_filename 
        nrw.save_training_data(X_all, Y_all, avg_powers_all, filename_trainingdata)
        for ieval in range(num_examples):
            os.rename(output_dir + "/run_" + str(ieval),
                      output_dir + "/" + iter_dir
                      + str(ieval))
    elif data_init_type == 2: # Genetic algorithm
        print("Using a genetic algorithm!")
        ga_n_iter = int(argv[4])
        init_points = num_examples
        num_inputs = dataset_params["num_output"]

        num_parents_mating = int(init_points / 10.0)
        if (num_parents_mating % 2) != 0:
            num_parents_mating -=1
        if num_parents_mating < 2:
            num_parents_mating = 2
        num_init_examples = 0 # genetic algorithm generates it's own intial data
        X_all = np.array([], dtype=np.int64).reshape(num_inputs,0)
        Y_all= np.array([], dtype=np.int64).reshape(num_modes,0)
        avg_powers_all = np.array([], dtype=np.int64)

        opt_params = uopt.define_optimizer_parameters(output_dir, num_inputs,
                                                      num_modes, num_init_examples,
                                                      ga_n_iter, num_parallel, random_seed)
        opt_params["run_clean"] = run_clean
        dataset = uopt.define_optimizer_dataset(X_all, Y_all, avg_powers_all)
        ga_params = uopt.define_genetic_algorithm_params(init_points, num_parents_mating)
        dataset = wrapper_genetic_algorithm(dataset, ga_params, opt_params)
    elif data_init_type == 0:
        print("Importing pre-generated data!")
        # copy across dataset_params and facility_spec
        shutil.copyfile(input_dir + "/dataset_params.nc", output_dir + "/dataset_params.nc")
        shutil.copyfile(input_dir + "/facility_spec.nc", output_dir + "/facility_spec.nc")
    else:
        print("")
        sys.exit("Dataset not properly specified")

    print("Importing data!")
    sys_params["trainingdata_filename"] = trainingdata_filename
    X_all, Y_all, avg_powers_all = nrw.import_training_data(sys_params)
    num_init_examples = np.shape(X_all)[1]
    num_inputs = np.shape(X_all)[0]
    dataset = uopt.define_optimizer_dataset(X_all, Y_all, avg_powers_all)

    use_bayesian_optimization = bool(int(argv[5]))
    if use_bayesian_optimization: # Bayesian optimization
        bo_n_iter = int(argv[6])
        target_all = np.sqrt(np.sum(dataset["Y_all"]**2, axis=0))
        target_mean = np.mean(target_all)
        target_set_undetermined = target_mean / 2.0 # half mean for all undetermined BO values
        opt_params = uopt.define_optimizer_parameters(output_dir, num_inputs,
                                                      num_modes, num_init_examples,
                                                      bo_n_iter, num_parallel, random_seed)
        opt_params["run_clean"] = run_clean

        bo_params = uopt.define_bayesian_optimisation_params(target_set_undetermined, np.mean(dataset["avg_powers_all"]))
        dataset = wrapper_bayesian_optimisation(dataset, bo_params, opt_params)
        num_init_examples = np.shape(dataset["X_all"])[1]

    use_gradient_descent = bool(int(argv[7]))
    if use_gradient_descent: # Gradient descent
        print("Using gradient descent!")
        gd_n_iter = int(argv[8])
        opt_params = uopt.define_optimizer_parameters(output_dir, num_inputs,
                                                            num_modes, num_init_examples,
                                                            gd_n_iter, num_parallel, random_seed)
        opt_params["run_clean"] = run_clean

        gd_params = uopt.define_gradient_descent_params(num_parallel)
        dataset = wrapper_gradient_descent(dataset, gd_params, opt_params)
        num_init_examples = np.shape(dataset["X_all"])[1]

    return


if __name__ == "__main__":
    main(sys.argv)
    print("terminated normally")
