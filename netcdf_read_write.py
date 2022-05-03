from netCDF4 import Dataset
import numpy as np
from os import path
import os
import healpy as hp
import glob
from healpy_pointings import rot_mat
import utils_intensity_map as uim

def read_nn_weights(filename_nn_weights):
    parameters = {}

    rootgrp = Dataset(filename_nn_weights + ".nc")
    keys = list(rootgrp["parameters"].variables.keys())
    for key in keys:
        parameters[key] = rootgrp["parameters"][key][:,:]
    rootgrp.close()

    return parameters



def save_nn_weights(parameters, filename_nn_weights):
    if path.exists(filename_nn_weights + '.nc'):
        os.remove(filename_nn_weights + '.nc')

    rootgrp = Dataset(filename_nn_weights + '.nc', 'w')
    parms = rootgrp.createGroup('parameters')
    for key, item in parameters.items():
        dims = np.shape(item)
        parms.createDimension(key+'_'+'item_dim1', dims[0])
        parms.createDimension(key+'_'+'item_dim2', dims[1])
        variable = parms.createVariable(key, 'f4', (key+'_'+'item_dim1',key+'_'+'item_dim2'))
        variable[:,:] = item
    rootgrp.close()



def read_intensity(data_location, run_type, beam_names, nside):
    if (run_type == "nif"):
        start = [data_location + '/p_in_z1z2_beam_NIF-'] * len(beam_names)
        end = ['.nc'] * len(beam_names)
        files = [i + j for i, j in zip(start, beam_names)]
        files = [i + j for i, j in zip(files, end)]
    elif (run_type == "test"):
        files = glob.glob(data_location + '/p_in_*.nc')
    intensity_map = 0.0

    for file_name in files:
        b='Reading from: ' + file_name + "  "
        print("\r", b, end="")
        cone_data = Dataset(file_name)
        intensity_data = cone_data.variables["intensity"][:]
        intensity_map = intensity_data + intensity_map
        theta = cone_data.variables["theta"][:]
        phi = cone_data.variables["phi"][:]
        cone_data.close()

    indices = np.argsort(phi + theta * nside**2*12)
    intensity_map = intensity_map[indices]

    return intensity_map



def rotate_cone_and_save(the_data, quad_name, hpxmap, imap_nside, pointing_theta, pointing_phi, data_location):
    quad_start_ind = the_data["Quad"].index(quad_name)
    quad_slice = slice(quad_start_ind, quad_start_ind+4)
    old_imap_theta = np.mean(the_data['Theta'][quad_slice])
    old_imap_phi = np.mean(the_data['Phi'][quad_slice])

    cone_name = the_data['Cone'][quad_slice]
    cone_name = cone_name[0]
    quad_name = the_data['Quad'][quad_slice]
    quad_name = quad_name[0]

    quad_count = int(the_data["Cone"].count(cone_name)/2)
    cone_slice = slice(the_data["Cone"].index(cone_name),the_data["Quad"].index(quad_name)+quad_count,4)
    quad_list_in_cone = the_data["Quad"][cone_slice]

    cone_name = str(cone_name)
    cone_name = cone_name.replace(".", "_")

    imap_npix = imap_nside**2 * 12
    index = np.arange(imap_npix)
    imap_theta, imap_phi = hp.pix2ang(imap_nside, index)

    save_name = data_location + "/" + data_location + "_cone_" + cone_name + ".nc"
    if path.exists(save_name):
        os.remove(save_name)
    b="Rotating cone: " + cone_name + " with " + data_location
    print("\r", b, end="")
    rootgrp = Dataset(save_name, "w", format="NETCDF4")

    coord_o = np.zeros([3, imap_npix])
    for i in range(len(quad_list_in_cone)):
        quad_start_ind = the_data["Quad"].index(quad_list_in_cone[i])
        quad_slice = slice(quad_start_ind,quad_start_ind+4)
        quad_grp = rootgrp.createGroup(quad_list_in_cone[i])

        quad_grp.createDimension('pointing_dim', 3)
        port_loc = quad_grp.createVariable('port_loc', 'f4', ('pointing_dim',))
        port_loc[0] = the_data['target_radius']
        port_loc[1] = np.mean(the_data['Theta'][quad_slice])
        port_loc[2] = np.mean(the_data['Phi'][quad_slice])

        ############## rotation by interpolation ##################
        rotate_theta = port_loc[1] - old_imap_theta
        rotate_phi = port_loc[2] - old_imap_phi

        theta = old_imap_theta
        phi = old_imap_phi
        # I do not know the reason for the different signs in front of "rotate_theta"
        # but i checked and this works
        alpha = old_imap_theta + rotate_theta
        beta = old_imap_phi - rotate_phi

        # rotate so base port location aligns with z axis and then rotate to new port
        rotation_matrix = np.matmul(np.matmul(rot_mat(beta, "z"), rot_mat(alpha, "y")),
                                    np.matmul(rot_mat(-theta, "y"), rot_mat(-phi, "z")))

        # Use cartesian to apply rotation
        coord_o[0,:] = np.cos(imap_phi) * np.sin(imap_theta)
        coord_o[1,:] = np.sin(imap_phi) * np.sin(imap_theta)
        coord_o[2,:] = np.cos(imap_theta)

        # rotate intensity map coordinates
        coord_n = np.matmul(rotation_matrix, coord_o)

        # convert back to spherical polar
        imap_theta_new = np.arccos(coord_n[2])
        imap_phi_new = np.arctan2(coord_n[1], coord_n[0])
        ###########################################################

        intensity_map_rotate = hp.get_interp_val(hpxmap, imap_theta_new, imap_phi_new)

        quad_pointing = quad_grp.createVariable('quad_pointing', 'f4', ('pointing_dim',))
        quad_pointing[0] = the_data['target_radius']
        quad_pointing[1] = pointing_theta + rotate_theta
        quad_pointing[2] = pointing_phi + rotate_phi

        quad_grp.createDimension('intensity_dim', imap_npix)
        intensity_map = quad_grp.createVariable('intensity_map', 'f4', ('intensity_dim',))
        intensity_map.units = "W/cm^2"
        intensity_map[:] = intensity_map_rotate[:]

    rootgrp.close()



def assemble_full_sphere(Y_train, Y_norms, the_data, filename_pointing, filename_defocus, power_range):
    portloc_theta = np.zeros(the_data['num_quads'])
    portloc_phi = np.zeros(the_data['num_quads'])
    quad_pointing_theta = np.zeros(the_data['num_quads'])
    quad_pointing_phi = np.zeros(the_data['num_quads'])
    intensity_map = 0.0

    pointing_nside = Y_norms[0]
    num_defocus = Y_norms[1]
    num_powers = Y_norms[2]
    pointing_per_cone = (Y_train[0:4] * (pointing_nside-1)).astype(int)
    defocus_per_cone = (Y_train[4:8] * (num_defocus-1)).astype(int)
    power_per_cone = (Y_train[8:] * (num_powers-1)).astype(int)

    power_fractions = np.linspace(power_range, 1, num_powers)

    iquad = 0
    icone = 0

    for quad_name in the_data['quad_from_each_cone']:
        quad_slice = slice(the_data["Quad"].index(quad_name),the_data["Quad"].index(quad_name)+4)

        cone_name = the_data['Cone'][quad_slice]
        cone_name = cone_name[0]
        quad_name = the_data['Quad'][quad_slice]
        quad_name = quad_name[0]

        beam_count = int(the_data["Cone"].count(cone_name)/2)
        cone_slice = slice(the_data["Cone"].index(cone_name),the_data["Quad"].index(quad_name)+beam_count,4)
        quad_list_in_cone = the_data["Quad"][cone_slice]

        cone_name = str(cone_name)
        cone_name = cone_name.replace(".", "_")
        run_location = filename_pointing + str(pointing_per_cone[icone]) + "_" + filename_defocus + str(defocus_per_cone[icone])
        save_name = run_location + "/" + run_location + "_cone_" + cone_name + ".nc"

        for i in range(int(beam_count/4)):
            cone_data = Dataset(save_name)
            intensity_data = cone_data[quad_list_in_cone[i]]
            port_loc = intensity_data.variables["port_loc"][:]
            portloc_theta[iquad] = port_loc[1]
            portloc_phi[iquad] = port_loc[2]
            quad_pointing = intensity_data.variables["quad_pointing"][:]
            quad_pointing_theta[iquad] = quad_pointing[1]
            quad_pointing_phi[iquad] = quad_pointing[2]
            power_mult = power_fractions[power_per_cone[icone]]
            intensity_map = intensity_map + intensity_data.variables["intensity_map"][:] * power_mult
            cone_data.close()

            iquad = iquad + 1
        icone = icone + 1

    intensity_map = intensity_map + np.flip(intensity_map)

    port_loc = np.vstack((portloc_theta, portloc_phi))
    quad_pointing = np.vstack((quad_pointing_theta, quad_pointing_phi))

    return intensity_map, port_loc, quad_pointing



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
    num_coeff = int(np.shape(X_train)[0] / 2.0)
    num_output = np.shape(Y_train)[0]

    if path.exists(filename_trainingdata):
        os.remove(filename_trainingdata)
    rootgrp = Dataset(filename_trainingdata, "w", format="NETCDF4")
    rootgrp.createDimension('num_examples', num_examples)
    rootgrp.createDimension('num_coeff_ir', num_coeff*2)
    rootgrp.createDimension('num_output', num_output)

    X_train_save = rootgrp.createVariable('X_train', 'f4', ('num_coeff_ir','num_examples'))
    X_train_save[:,:] = X_train

    Y_train_save = rootgrp.createVariable('Y_train', 'f4', ('num_output','num_examples'))
    Y_train_save[:,:] = Y_train

    avg_powers_save = rootgrp.createVariable('avg_powers', 'f4', ('num_examples'))
    avg_powers_save[:] = avg_powers

    rootgrp.close()