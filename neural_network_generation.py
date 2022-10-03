from netCDF4 import Dataset
import numpy as np
import training_data_generation as tdg
import tf_neural_network as tfnn
import matplotlib.pyplot as plt
import nn_plots as nnp
import netcdf_read_write as nrw
import utils_intensity_map as uim
import sys
import healpy as hp


def define_nn_params(num_nn):
    nn_params = {}

    nn_params["random_seed"] = 12345
    nn_params["test_fraction"] = 1.0 / 100.0
    nn_params["dir_nn_weights"] = "neural_network_weights"
    nn_params["num_nn"] = num_nn
    nn_params["filename_hyperparams"] = "NN_hyper_parameters"

    return nn_params



def define_nn_hyperparams(num_epochs, num_nn, **kwargs):
    mean = kwargs.get("mean", 0.0)
    std_dev = kwargs.get("std_dev", 1.0)

    nn_hyperparams = {}

    nn_hyperparams["num_epochs"] = [num_epochs] * num_nn
    nn_hyperparams["learning_rate"] = [0.001] * num_nn
    nn_hyperparams["hidden_units1"] = [600] * num_nn
    nn_hyperparams["hidden_units2"] = [600] * num_nn
    nn_hyperparams["hidden_units3"] = [600] * num_nn
    nn_hyperparams["mu"] = [mean] * num_nn
    nn_hyperparams["sigma"] = [std_dev] * num_nn

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



def change_number_modes(Y_train, avg_powers_all, LMAX):

    num_examples = np.shape(Y_train)[1]
    Y_train2 = np.zeros((LMAX, num_examples))
    num_coeff = int(((LMAX + 2) * (LMAX + 1))/2.0)
    np_complex = np.vectorize(complex)
    for ie in range(num_examples):
        Y_train_real = np.squeeze(Y_train[:,ie] / avg_powers_all[ie])
        Y_train_complex = np_complex(Y_train_real[:num_coeff], Y_train_real[num_coeff:])
        var = abs(Y_train_complex)**2
        the_modes = np.zeros(LMAX)
        for l in range(LMAX):
            for m in range(l):
                if (m>0):
                    the_modes[l] = the_modes[l] + 2.*var[hp.sphtfunc.Alm.getidx(LMAX, l, m)]
                else:
                    the_modes[l] = the_modes[l] + var[hp.sphtfunc.Alm.getidx(LMAX, l, m)]
        power_spectrum_unweighted = np.sqrt(the_modes)
        Y_train2[:,ie] = power_spectrum_unweighted

    return Y_train2


def import_training_data_reversed(nn_params, sys_params, LMAX):
    training_data = Dataset(sys_params["root_dir"] + "/" + sys_params["trainingdata_filename"])
    X_all = training_data.variables["Y_train"][:]
    Y_all = training_data.variables["X_train"][:]
    avg_powers_all = training_data.variables["avg_powers"][:]
    training_data.close()
    print(np.shape(X_all), np.shape(Y_all))

    Y_mag = nng.change_number_modes(Y_all, avg_powers_all, LMAX)

    nn_params["num_examples"] = np.shape(X_all)[1]
    nn_params["input_size"] = np.shape(X_all)[0]
    nn_params["output_size"] = np.shape(Y_mag)[0]

    test_size = int(nn_params["num_examples"] * nn_params["test_fraction"])
    if test_size == 0:
        test_size = 1
    nn_params["test_size"] = test_size

    return X_all, Y_mag, avg_powers_all, nn_params



def seperate_test_set(X_all, Y_all, avg_powers_all, nn_params):
    nn_dataset = {}
    test_size = nn_params["test_size"]
    if test_size > 0:
        nn_dataset["X_test"] = X_all.T[:test_size,:]
        nn_dataset["Y_test"] = Y_all.T[:test_size,:]
        nn_dataset["test_avg_powers"] = avg_powers_all[:test_size]

        nn_dataset["X_train"] = X_all.T[test_size:,:]
        nn_dataset["Y_train"] = Y_all.T[test_size:,:]
        nn_dataset["train_avg_powers"] = avg_powers_all[test_size:]
    else:
        nn_dataset["X_test"] = [0.0]
        nn_dataset["Y_test"] = [0.0]
        nn_dataset["test_avg_powers"] = [0.0]

        nn_dataset["X_train"] = X_all.T
        nn_dataset["Y_train"] = Y_all.T
        nn_dataset["train_avg_powers"] = avg_powers_all

    print("Train shape for input ", np.shape(nn_dataset["X_train"]),
          "Train shape for output ", np.shape(nn_dataset["Y_train"]))
    print("Test shape for input ", np.shape(nn_dataset["X_test"]),
          "Test shape for output ", np.shape(nn_dataset["Y_test"]))

    return nn_dataset



def normalise(nn_dataset, **kwargs):
    nn_dataset["mu"] = kwargs.get("mean", np.mean(nn_dataset["X_train"]))
    nn_dataset["sigma"] = kwargs.get("std_dev", np.std(nn_dataset["X_train"]))

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
    root_dir = argv[1]
    num_epochs = int(argv[2])
    num_nn = int(argv[3])

    sys_params = tdg.define_system_params(root_dir)
    nn_params = define_nn_params(num_nn)
    X_all, Y_all, avg_powers_all, nn_params = import_training_data(nn_params, sys_params)
    nn_dataset = seperate_test_set(X_all, Y_all, avg_powers_all, nn_params)
    nn_dataset = normalise(nn_dataset)

    if (nn_params["num_nn"] > 0):
        nn_hyperparams = define_nn_hyperparams(num_epochs, num_nn, mean=nn_dataset["mu"], std_dev=nn_dataset["sigma"])
        nn_hyperparams = multiple_nn(nn_params, nn_dataset, sys_params, nn_hyperparams)
    return nn_params, nn_dataset, sys_params, nn_hyperparams



if __name__ == "__main__":
    _, _, _, _ = main(sys.argv)
