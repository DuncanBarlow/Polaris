from netCDF4 import Dataset
import numpy as np
from os import path
import os
import glob
from healpy_pointings import rot_mat
import utils_intensity_map as uim



def read_nn_weights(filename_nn_weights):
    parameters = {}

    rootgrp = Dataset(filename_nn_weights + ".nc")
    keys = list(rootgrp["parameters"].variables.keys())
    for key in keys:
        if np.shape(np.shape(rootgrp["parameters"][key]))[0] == 2:
            parameters[key] = rootgrp["parameters"][key][:,:]
        if np.shape(np.shape(rootgrp["parameters"][key]))[0] == 1:
            parameters[key] = rootgrp["parameters"][key][:]
    rootgrp.close()

    return parameters



def retrieve_xtrain_and_delete(iex, dataset_params, sys_params):
    run_location = sys_params["root_dir"] + "/" + sys_params["sim_dir"] + str(iex)
    if sys_params["run_compression"]:
        intensity_map = read_intensity(run_location, dataset_params["imap_nside"])
        X_train1, avg_power1 = uim.create_xtrain(intensity_map, dataset_params["LMAX"])

    if sys_params["run_clean"]:
        os.remove(run_location + '/main')
        os.remove(run_location + '/p_in_z1z2_beam_all.nc')
    return X_train1, avg_power1



def save_nn_weights(parameters, filename_nn_weights):
    if path.exists(filename_nn_weights + '.nc'):
        os.remove(filename_nn_weights + '.nc')

    rootgrp = Dataset(filename_nn_weights + '.nc', 'w')
    parms = rootgrp.createGroup('parameters')
    for key, item in parameters.items():
        dims = np.shape(item)
        if np.shape(dims)[0] == 2:
            parms.createDimension(key+'_'+'item_dim1', dims[0])
            parms.createDimension(key+'_'+'item_dim2', dims[1])
            variable = parms.createVariable(key, 'f4', (key+'_'+'item_dim1',key+'_'+'item_dim2'))
            variable[:,:] = item
        if np.shape(dims)[0] == 1:
            parms.createDimension(key+'_'+'item_dim1', dims[0])
            variable = parms.createVariable(key, 'f4', (key+'_'+'item_dim1'))
            variable[:] = item

    rootgrp.close()



def read_intensity(data_location, nside):
    file_name = data_location + '/p_in_z1z2_beam_all.nc'

    b='Reading from: ' + file_name + "  "
    print("\r", b, end="")
    cone_data = Dataset(file_name)
    intensity_data = cone_data.variables["intensity"][:]
    theta = cone_data.variables["theta"][:]
    phi = cone_data.variables["phi"][:]
    cone_data.close()

    indices = np.argsort(phi + theta * nside**2*12)
    intensity_map = intensity_data[indices]

    return intensity_map



def create_training_data(the_data, num_cones, pointing_nside, num_defocus, num_powers, num_coeff, num_output, num_examples, power_range, LMAX, savename_trainingdata, filename_pointing, filename_defocus):

    pointing_per_cone = [0,0,0,0]
    defocus_per_cone = [0,0,0,0]
    power_per_cone = [0,0,0,0]
    X_train = np.zeros((num_coeff * 2, num_examples))
    Y_train = np.zeros((num_output, num_examples))
    avg_powers = np.zeros(num_examples)

    i = 0
    for pind in range(pointing_nside):
        print("Currently assembling data for pointing: " + str(pind+1) + " of " + str(pointing_nside))
        for dind in range(num_defocus):
            for pwind in range(num_powers):
                for cind in range(num_cones):
                    pointing_per_cone[cind] = pind
                    defocus_per_cone[cind] = dind
                    power_per_cone[cind] = pwind

                    Y_train1, Y_norms = uim.create_ytrain(pointing_per_cone, pointing_nside, defocus_per_cone, num_defocus, power_per_cone, num_powers)

                    intensity_map, _, _ = assemble_full_sphere(Y_train1, Y_norms, the_data, filename_pointing, filename_defocus, power_range)

                    X_train1, avg_power = uim.create_xtrain(intensity_map, LMAX)

                    Y_train[:,i] = Y_train1
                    X_train[:,i] = X_train1
                    avg_powers[i] = avg_power
                    i = i + 1

    save_training_data(X_train, Y_train, avg_powers, savename_trainingdata)



def save_training_data(X_train, Y_train, avg_powers, filename_trainingdata):
    num_examples = np.shape(X_train)[1]
    num_inputs = np.shape(X_train)[0]
    num_output = np.shape(Y_train)[0]

    if path.exists(filename_trainingdata):
        os.remove(filename_trainingdata)
    rootgrp = Dataset(filename_trainingdata, "w", format="NETCDF4")
    rootgrp.createDimension('num_examples', num_examples)
    rootgrp.createDimension('num_coeff_ir', num_inputs)
    rootgrp.createDimension('num_output', num_output)

    X_train_save = rootgrp.createVariable('X_train', 'f4', ('num_coeff_ir','num_examples'))
    X_train_save[:,:] = X_train

    Y_train_save = rootgrp.createVariable('Y_train', 'f4', ('num_output','num_examples'))
    Y_train_save[:,:] = Y_train

    avg_powers_save = rootgrp.createVariable('avg_powers', 'f4', ('num_examples'))
    avg_powers_save[:] = avg_powers

    rootgrp.close()