from netCDF4 import Dataset
import numpy as np
from os import path
import os
import healpy as hp
import glob
from healpy_pointings import rot_mat


def read_intensity(data_location, run_type, beam_names, nside):
    if (run_type == "nif"):
        start = [data_location + '/p_in_z1z2_beam_NIF-'] * 4
        end = ['.nc']*4
        files = [i + j for i, j in zip(start, beam_names)]
        files = [i + j for i, j in zip(files, end)]
    elif (run_type == "test"):
        files = glob.glob(data_location + '/p_in_*.nc')
    intensity_map = 0.0
    n_beams = len(files)

    for file_name in files:
        b='Reading from: ' + file_name + "  "
        print("\r", b, end="")
        cone_data = Dataset(file_name)
        intensity_data = cone_data.variables["intensity"][:]
        intensity_map = intensity_data + intensity_map
        theta = cone_data.variables["theta"][:]
        phi = cone_data.variables["phi"][:]

    indices = np.argsort(phi+theta*nside**2*12)
    intensity_map = intensity_map[indices]

    return intensity_map, n_beams



def rotate_cone_and_save(the_data, quad_name, hpxmap, imap_nside, pointing_theta, pointing_phi, pointing_ind, file_naming):
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

    save_name = file_naming + str(pointing_ind) + "/theta_" + str(pointing_ind) + "_cone_" + cone_name + ".nc"
    if path.exists(save_name):
        os.remove(save_name)
    b="Rotating cone: " + cone_name + " for on healpy grid point: " + str(pointing_ind) + "     "
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