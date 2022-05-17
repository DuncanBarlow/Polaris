from netCDF4 import Dataset
import healpy as hp
import numpy as np
import training_data_generation as tdg
import tf_neural_network as tfnn
import matplotlib.pyplot as plt
import nn_plots as nnp
import netcdf_read_write as nrw
import sys


def define_nn_params(num_nn):
    nn_params = {}
    
    nn_params["run_shuffle"] = True
    nn_params["random_seed"] = 12345
    nn_params["test_fraction"] = 1.0 / 100.0
    nn_params["dir_nn_weights"] = "neural_network_weights"
    nn_params["num_nn"] = num_nn
    nn_params["filename_hyperparams"] = "NN_hyper_parameters"
    
    return nn_params



def define_nn_hyperparams(num_epochs, num_nn):
    nn_hyperparams = {}

    nn_hyperparams["num_epochs"] = [num_epochs] * num_nn
    nn_hyperparams["learning_rate"] = [0.001] * num_nn
    nn_hyperparams["hidden_units1"] = [600] * num_nn
    nn_hyperparams["hidden_units2"] = [600] * num_nn
    nn_hyperparams["hidden_units3"] = [600] * num_nn
    nn_hyperparams["cost"] = np.zeros(num_nn)
    nn_hyperparams["train_acc"] = np.zeros(num_nn)
    nn_hyperparams["test_acc"] = np.zeros(num_nn)
    nn_hyperparams["initialize_seed"] = np.arange(num_nn)

    return nn_hyperparams



def import_training_data(nn_params, sys_params):
    training_data = Dataset(sys_params["root_dir"] + "/" + sys_params["trainingdata_filename"])
    X_all = training_data.variables["X_train"][:]
    Y_all = training_data.variables["Y_train"][:]
    avg_powers_all = training_data.variables["avg_powers"][:]
    training_data.close()

    print(np.shape(X_all), np.shape(Y_all))
    nn_params["num_examples"] = np.shape(X_all)[1]
    nn_params["input_size"] = np.shape(X_all)[0]
    nn_params["output_size"] = np.shape(Y_all)[0]

    test_size = int(nn_params["num_examples"] * nn_params["test_fraction"])
    if test_size == 0:
        test_size = 1
    nn_params["test_size"] = test_size

    return X_all, Y_all, avg_powers_all, nn_params



def seperate_test_set(X_all, Y_all, avg_powers_all, nn_params):
    nn_dataset = {}
    test_size = nn_params["test_size"]
    if nn_params["run_shuffle"]:
        index_shuf = list(range(nn_params["num_examples"]))
        rng = np.random.default_rng(nn_params["random_seed"])
        rng.shuffle(index_shuf)
        index_shuf = np.array(index_shuf)
        X_train = []
        Y_train = []
        train_avg_powers = []
        X_test = []
        Y_test = []
        test_avg_powers = []
        j=0
        for i in index_shuf:
            if (j < test_size):
                X_test.append(X_all[:,i])
                Y_test.append(Y_all[:,i])
                test_avg_powers.append(avg_powers_all[i])
            else:
                X_train.append(X_all[:,i])
                Y_train.append(Y_all[:,i])
                train_avg_powers.append(avg_powers_all[i])
            j = j + 1
        nn_dataset["index_shuf"] = index_shuf
        nn_dataset["X_test"] = np.array(X_test)
        nn_dataset["Y_test"] = np.array(Y_test)
        nn_dataset["X_train"] = np.array(X_train)
        nn_dataset["Y_train"] = np.array(Y_train)
        nn_dataset["test_avg_powers"] = test_avg_powers
        nn_dataset["train_avg_powers"] = train_avg_powers
    else:
        nn_dataset["X_test"] = X_all.T[:test_size,:]
        nn_dataset["Y_test"] = Y_all.T[:test_size,:]
        nn_dataset["test_avg_powers"] = avg_powers_all[:test_size]

        nn_dataset["X_train"] = X_all.T[test_size:,:]
        nn_dataset["Y_train"] = Y_all.T[test_size:,:]
        nn_dataset["train_avg_powers"] = avg_powers_all[test_size:]

    print("Train shape for input ", np.shape(nn_dataset["X_train"]),
          "Train shape for output ", np.shape(nn_dataset["Y_train"]))
    print("Test shape for input ", np.shape(nn_dataset["X_test"]),
          "Test shape for output ", np.shape(nn_dataset["Y_test"]))
    
    return nn_dataset



def normalise(nn_dataset):
    nn_dataset["mu"] = np.mean(nn_dataset["X_train"])
    nn_dataset["sigma"] = np.std(nn_dataset["X_train"])

    nn_dataset["X_train"] = (nn_dataset["X_train"] - nn_dataset["mu"]) / nn_dataset["sigma"]
    nn_dataset["X_test"] = (nn_dataset["X_test"] - nn_dataset["mu"]) / nn_dataset["sigma"]
    return nn_dataset



def multiple_nn(nn_params, nn_dataset, sys_params, nn_hyperparams):

    print_cost = False
    if nn_params["num_nn"] == 1:
        print_cost = True
    
    for inn in range(nn_params["num_nn"]):
        parameters, costs, train_acc, test_acc, epochs = tfnn.model_wrapper(nn_params, nn_dataset, nn_hyperparams["num_epochs"][inn], nn_hyperparams["learning_rate"][inn], nn_hyperparams["hidden_units1"][inn], nn_hyperparams["hidden_units2"][inn], nn_hyperparams["hidden_units3"][inn], initialize_seed = nn_hyperparams["initialize_seed"][inn], print_cost = print_cost)
        filename_nn_weights = nn_params["dir_nn_weights"] + "/NN" + str(inn)
        nrw.save_nn_weights(parameters, filename_nn_weights)
        nn_hyperparams["cost"][inn] = costs[-1]
        nn_hyperparams["train_acc"][inn] = train_acc[-1]
        nn_hyperparams["test_acc"][inn] = test_acc[-1]
        print("Trained neural network index/seed: ", inn)
    filename_hyperparams = nn_params["dir_nn_weights"] + "/" + nn_params["filename_hyperparams"]
    nrw.save_nn_weights(nn_hyperparams, filename_hyperparams)

    if nn_params["num_nn"] == 1:
        nnp.plotting(epochs, costs, train_acc, test_acc, nn_hyperparams["learning_rate"], sys_params["figure_location"])

    return nn_hyperparams



def main(argv):
    sys_params = tdg.define_system_params(argv[1])
    nn_params = define_nn_params(int(argv[3]))
    X_all, Y_all, avg_powers_all, nn_params = import_training_data(nn_params, sys_params)
    nn_dataset = seperate_test_set(X_all, Y_all, avg_powers_all, nn_params)
    nn_dataset = normalise(nn_dataset)

    nn_hyperparams = {}
    if (nn_params["num_nn"] > 0):
        nn_hyperparams = define_nn_hyperparams(int(argv[2]), int(argv[3]))
        nn_hyperparams = multiple_nn(nn_params, nn_dataset, sys_params, nn_hyperparams)
    return nn_params, nn_dataset, sys_params, nn_hyperparams



if __name__ == "__main__":
    _, _, _, _ = main(sys.argv)