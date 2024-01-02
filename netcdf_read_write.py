from netCDF4 import Dataset
import numpy as np
import os
import glob
from healpy_pointings import rot_mat
import utils_intensity_map as uim
import utils_healpy as uhp


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



def read_general_netcdf(filename):
    parameters = {}

    rootgrp = Dataset(filename)
    keys = list(rootgrp.variables.keys())
    for key in keys:
        if np.shape(np.shape(rootgrp[key]))[0] == 3:
            parameters[key] = rootgrp[key][:,:,:]
        if np.shape(np.shape(rootgrp[key]))[0] == 2:
            parameters[key] = rootgrp[key][:,:]
        if np.shape(np.shape(rootgrp[key]))[0] == 1:
            parameters[key] = rootgrp[key][:]
        #print(key, parameters[key])

    keys = list(rootgrp.__dict__.keys())
    for key in keys:
        parameters[key] = getattr(rootgrp, key)

    rootgrp.close()

    return parameters



def save_general_netcdf(parameters, filename):
    if os.path.exists(filename):
        os.remove(filename)

    rootgrp = Dataset(filename, 'w')
    for key, item in parameters.items():
        dims = np.shape(item)
        total_dims = np.shape(dims)[0]
        #print(key, type(item))
        if isinstance(item, tuple):
            var_type = 'f4'
        if isinstance(item, np.ndarray):
            #print(item.dtype)
            if item.dtype == "i":
                var_type = 'i4'
            if "float" in str(item.dtype):# == "float64":
                var_type = 'f4'
            if "<U" in str(item.dtype):
                # this is designed to catch strings use np.array(my_array, dtype='<U*')
                str_length = len(item[0])
                item = np.array(item, dtype='S'+str(str_length))
                var_type = 'S1'
                dims = dims + (str_length,)
                total_dims += 1
        if isinstance(item, list):
            var_type = 'S1'
            try: # if string
                str_length = len(item[0])
            except:
                str_length = 10 # this is definitely not robust
            item = np.array(item, dtype='S'+str(str_length))
            dims = dims + (str_length,)
            total_dims += 1
        if total_dims == 3:
            rootgrp.createDimension(key+'_'+'item_dim1', dims[0])
            rootgrp.createDimension(key+'_'+'item_dim2', dims[1])
            rootgrp.createDimension(key+'_'+'item_dim3', dims[2])
            variable = rootgrp.createVariable(key, var_type,
                                            (key+'_'+'item_dim1',
                                             key+'_'+'item_dim2',
                                             key+'_'+'item_dim3'))
            variable._Encoding = 'ascii' # this enables automatic conversion of strings
            variable[:,:,:] = item
        if total_dims == 2:
            rootgrp.createDimension(key+'_'+'item_dim1', dims[0])
            rootgrp.createDimension(key+'_'+'item_dim2', dims[1])
            variable = rootgrp.createVariable(key, var_type,
                                            (key+'_'+'item_dim1',
                                             key+'_'+'item_dim2'))
            variable._Encoding = 'ascii' # this enables automatic conversion
            variable[:,:] = item
        if total_dims == 1:
            rootgrp.createDimension(key+'_'+'item_dim1', dims[0])
            variable = rootgrp.createVariable(key, var_type,
                                            (key+'_'+'item_dim1'))
            variable._Encoding = 'ascii' # this enables automatic conversion
            variable[:] = item
        if total_dims == 0:
            if item == True:
                item = 1
            if item == False:
                item = 0
            setattr(rootgrp, key, item)
    rootgrp.close()



def retrieve_xtrain_and_delete(min_parallel, max_parallel, dataset, dataset_params, sys_params, facility_spec):

    for iex in range(min_parallel, max_parallel+1):
        run_location = sys_params["root_dir"] + "/" + sys_params["sim_dir"] + str(iex)

        dir_illumination = run_location + "/" + sys_params["ifriit_ouput_name"]
        if os.path.exists(dir_illumination):
            parameters = read_general_netcdf(dir_illumination)
            intensity_map = parameters["intensity"] * (facility_spec["target_radius"] / 10000.0)**2

            intensity_map_normalized, dataset["avg_flux"][iex,0] = uim.imap_norm(intensity_map)
            dataset["real_modes"][iex,0,:], dataset["imag_modes"][iex,0,:] = uhp.imap2modes(intensity_map_normalized, dataset_params["LMAX"])
            dataset["rms"][iex,0] = uim.alms2rms(dataset["real_modes"][iex,0,:], dataset["imag_modes"][iex,0,:], dataset_params["LMAX"])
        else:
            print("Broken solid sphere! Probably due to CBET convergence?")

        print("Without density profiles:")
        print('Intensity per steradian, {:.2e}W/sr^-1'.format(dataset["avg_flux"][iex, 0]))
        print("The rms is: ", dataset["rms"][iex,0]*100.0, "%")

        if dataset_params["run_plasma_profile"]:
            dir_illumination = run_location+"/"+sys_params["heat_source_nc"]
            if os.path.exists(dir_illumination):
                hs_and_modes = read_general_netcdf(dir_illumination)
                dataset["real_modes"][iex,1,:], dataset["imag_modes"][iex,1,:], dataset["avg_flux"][iex,1] = uim.heatsource_analysis(hs_and_modes)
                dataset["rms"][iex,1] = uim.alms2rms(dataset["real_modes"][iex,1,:], dataset["imag_modes"][iex,1,:], dataset_params["LMAX"])
            else:
                print("Broken with profiles! Probably due to CBET convergence?")

            print("With density profiles:")
            print('Intensity per steradian, {:.2e}W/sr^-1'.format(dataset["avg_flux"][iex, 1]))
            print("The rms is: ", dataset["rms"][iex,1]*100.0, "%")

        if sys_params["run_clean"]:
            #os.remove(run_location + "/" + sys_params["ifriit_binary_filename"])
            #os.remove(run_location + "/" + sys_params["ifriit_ouput_name"])
            for filename in glob.glob(run_location + "/fort.*"):
                os.remove(filename)
            for filename in glob.glob(run_location + "/abs_beam_*"):
                os.remove(filename)
    return dataset



def save_nn_weights(parameters, filename_nn_weights):
    if os.path.exists(filename_nn_weights + '.nc'):
        os.remove(filename_nn_weights + '.nc')

    rootgrp = Dataset(filename_nn_weights + '.nc', 'w')
    parms = rootgrp.createGroup('parameters')
    for key, item in parameters.items():
        dims = np.shape(item)
        if np.shape(dims)[0] == 3:
            parms.createDimension(key+'_'+'item_dim1', dims[0])
            parms.createDimension(key+'_'+'item_dim2', dims[1])
            parms.createDimension(key+'_'+'item_dim3', dims[2])
            variable = parms.createVariable(key, 'f4',
                                            (key+'_'+'item_dim1',
                                             key+'_'+'item_dim2',
                                             key+'_'+'item_dim3'))
            variable[:,:,:] = item
        if np.shape(dims)[0] == 2:
            parms.createDimension(key+'_'+'item_dim1', dims[0])
            parms.createDimension(key+'_'+'item_dim2', dims[1])
            variable = parms.createVariable(key, 'f4',
                                            (key+'_'+'item_dim1',
                                             key+'_'+'item_dim2'))
            variable[:,:] = item
        if np.shape(dims)[0] == 1:
            parms.createDimension(key+'_'+'item_dim1', dims[0])
            variable = parms.createVariable(key, 'f4',
                                            (key+'_'+'item_dim1'))
            variable[:] = item

    rootgrp.close()



def read_intensity(data_location, nside, target_radius_microns):
    file_name = data_location + '/p_in_z1z2_beam_all.nc'

    cone_data = Dataset(file_name)
    intensity_data = cone_data.variables["intensity"][:]
    theta = cone_data.variables["theta"][:]
    phi = cone_data.variables["phi"][:]
    cone_data.close()

    indices = np.argsort(phi + theta * nside**2*12)
    intensity_map = intensity_data[indices]

    # convert from W/cm^2 to W/sr
    intensity_map = intensity_map * (target_radius_microns / 10000.0)**2

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

    if os.path.exists(filename_trainingdata):
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



def import_training_data_reversed(sys_params, LMAX):
    training_data = Dataset(sys_params["root_dir"] + "/" + sys_params["trainingdata_filename"])
    X_all = training_data.variables["Y_train"][:]
    Y_all = training_data.variables["X_train"][:]
    avg_powers_all = training_data.variables["avg_powers"][:]
    training_data.close()

    Y_mag = uim.change_number_modes(Y_all, avg_powers_all, LMAX)

    return X_all, Y_mag, avg_powers_all



def import_training_data(sys_params):
    training_data = Dataset(sys_params["root_dir"] + "/" + sys_params["trainingdata_filename"])
    X_all = training_data.variables["X_train"][:]
    Y_all = training_data.variables["Y_train"][:]
    avg_powers_all = training_data.variables["avg_powers"][:]
    training_data.close()

    return X_all, Y_all, avg_powers_all