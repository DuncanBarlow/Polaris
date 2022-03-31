from netCDF4 import Dataset
import numpy as np
from os import path
import healpy as hp
import glob



def read_intensity(data_location):
    files=glob.glob(data_location+'/p_in_*.nc')
    intensity_map = 0.0
    n_beams = len(files)
    for file_name in files:
        b='Reading from: ' + file_name + "  "
        print("\r", b, end="")
        cone_data = Dataset(file_name)
        intensity_data = cone_data.variables["intensity"]
        intensity_map = intensity_data[:] + intensity_map

    return intensity_map, n_beams



def rotate_cone_and_save(the_data, quad_name, hpxmap, imap_nside, pointing_theta, pointing_phi, pointing_ind):
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

    save_name = "test_hpi_"+str(pointing_ind)+"/hpi_" + str(pointing_ind) + "_cone_"+cone_name+".nc"
    if path.exists(save_name):
        os.remove(save_name)
    b="Rotating cone: " + cone_name + " for on healpy grid point: " + str(pointing_ind) + "     "
    print("\r", b, end="")
    rootgrp = Dataset(save_name, "w", format="NETCDF4")

    for i in range(len(quad_list_in_cone)):
        quad_start_ind = the_data["Quad"].index(quad_list_in_cone[i])
        quad_slice = slice(quad_start_ind,quad_start_ind+4)
        quad_grp = rootgrp.createGroup(quad_list_in_cone[i])

        quad_grp.createDimension('pointing_dim', 3)
        port_loc = quad_grp.createVariable('port_loc', 'f4', ('pointing_dim',))
        port_loc[0] = the_data['target_radius']
        port_loc[1] = np.mean(the_data['Theta'][quad_slice])
        port_loc[2] = np.mean(the_data['Phi'][quad_slice])

        rotate_theta = - old_imap_theta + np.mean(the_data['Theta'][quad_slice])
        imap_theta_new = imap_theta - np.radians(rotate_theta)
        rotate_phi = - old_imap_phi + np.mean(the_data['Phi'][quad_slice])
        imap_phi_new = imap_phi - np.radians(rotate_phi)

        quad_pointing = quad_grp.createVariable('quad_pointing', 'f4', ('pointing_dim',))
        quad_pointing[0] = the_data['target_radius']
        quad_pointing[1] = pointing_theta + np.radians(rotate_theta)
        quad_pointing[2] = pointing_phi + np.radians(rotate_phi)

        intensity_map_rotate = hp.get_interp_val(hpxmap, imap_theta_new, imap_phi_new)

        quad_grp.createDimension('intensity_dim', imap_npix)
        intensity_map = quad_grp.createVariable('intensity_map', 'f4', ('intensity_dim',))
        intensity_map.units = "W/cm^2"
        intensity_map[:] = intensity_map_rotate[:]

    rootgrp.close()