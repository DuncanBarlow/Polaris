import numpy as np
import os


def angle2moll(theta, phi):
    
    latitude = np.pi / 2.0 - theta
    if phi < np.pi:
        longitude = phi
    else:
        longitude = phi - 2.0 * np.pi
    
    rad = 1.0 / np.sqrt(2.0)
    longitude0 = 0.0
    i=0
    angle1 = latitude
    dangle = 0.1
    while (i < 100) and (abs(dangle) > 0.01):
        angle2 = angle1 - (2.0 * angle1 + np.sin(2.0 * angle1) - np.pi * np.sin(latitude)) / (4.0 * np.cos(angle1)**2)
        dangle = abs(angle2 - angle1)
        angle1 = angle2
        i+=1
    x = rad * 2.0 * np.sqrt(2.0) / np.pi * (longitude - longitude0) * np.cos(angle2)
    y = rad * np.sqrt(2.0) * np.sin(angle2)
    
    return x, y



def extract_rms(intensity_map_normalized):

    rms = np.sqrt(np.mean(intensity_map_normalized**2))

    return rms



def print_save_readout(print_list, stats_filename):
    if os.path.exists(stats_filename):
        os.remove(stats_filename)
    file1 = open(stats_filename,"a")
    for line in range(len(print_list)):
        print(print_list[line])
        file1.writelines(print_list[line]+"\n")
    file1.close()



def readout_intensity(facility_spec, intensity_map, use_ablation_pressure=0):
    n_beams = facility_spec['nbeams']
    total_TW = np.mean(intensity_map)*10**(-12) * 4.0 * np.pi
    mean_intensity_cm = np.mean(intensity_map) / (facility_spec['target_radius'] / 10000.0)**2

    #rms
    intensity_map_normalised, avg_flux = imap_norm(intensity_map)
    imap_pn = np.sign(intensity_map_normalised)
    intensity_map_rms = 100.0 * np.sqrt(np.mean(intensity_map_normalised**2))

    print_line = []
    print_line.append('Number of beams ' + str(n_beams))
    #print_line.append('Max power per beam {:.2f}TW, '.format(facility_spec['default_power']))
    print_line.append('Target radius {:.2f}um, '.format(facility_spec['target_radius']))

    print_line.append('RMS is {:.4f}%, '.format(intensity_map_rms))
    if use_ablation_pressure == 0:
        print_line.append('Mean intensity, {:.2e}W/cm2'.format(mean_intensity_cm))
        print_line.append('Mean intensity per steradian, {:.2e}W/sr'.format(avg_flux))
        print_line.append('The power per beam deposited is {:.4f}TW, '.format(total_TW / n_beams))
        print_line.append('The total power deposited is {:.2f}TW, '.format(total_TW))
    else:
        print_line.append('Mean ablation pressure: {:.2f}Mbar, '.format(avg_flux))

    return print_line, total_TW



def heatsource_analysis(hs_and_modes):

    avg_flux = hs_and_modes["average_flux"][0]
    real_modes = hs_and_modes["complex_modes"][0,:]
    imag_modes = hs_and_modes["complex_modes"][1,:]

    return real_modes, imag_modes, avg_flux




def extract_run_parameters(iex, power_deposited, dataset_params, facility_spec, sys_params, deck_gen_params, use_ablation_pressure=0):

    total_power = 0
    print_line = []
    beam_count = 0
    num_vars = dataset_params["num_variables_per_beam"]

    for icone in range(facility_spec['num_cones']):
        beams_per_cone = facility_spec['beams_per_cone'][icone]

        cone_theta_offset = deck_gen_params["sim_params"][iex,icone*num_vars+dataset_params["theta_index"]]
        cone_phi_offset = deck_gen_params["sim_params"][iex,icone*num_vars+dataset_params["phi_index"]]
        cone_defocus = deck_gen_params["defocus"][iex,beam_count]
        cone_powers = deck_gen_params["p0"][iex,beam_count] / (
                      facility_spec['default_power'] * facility_spec["beams_per_ifriit_beam"])

        total_power += cone_powers * beams_per_cone
        beam_count = beam_count + int(beams_per_cone / facility_spec["beams_per_ifriit_beam"])

        if icone < int(facility_spec['num_cones']/2):
            print_line.append("For cone " + str(icone+1) +
                  ": {:.2f}\N{DEGREE SIGN}, ".format(np.degrees(cone_theta_offset)) +
                  "{:.2f}\N{DEGREE SIGN}, ".format(np.degrees(cone_phi_offset)) +
                  "{:.2f}mm, ".format(cone_defocus) +
                  "{:.2f}% power, ".format(cone_powers * 100))

    mean_power_fraction = total_power / facility_spec['nbeams']
    print_line.append('The optimization selected a mean power percentage, {:.2f}%, '.format(mean_power_fraction * 100.0))

    print_line.append('Total power emitted {:.2f}TW, '.format(total_power))
    if use_ablation_pressure == 0:
        print_line.append('Percentage of emitted power deposited was {:.2f}%, '.format(power_deposited / (facility_spec["nbeams"] * facility_spec['default_power'] * mean_power_fraction) * 100.0))

    return print_line



def alms2rms(real_modes, imag_modes, lmax):

    # modes with m!=0 need to be x2 to account for negative terms
    # in healpix indexing the first lmax terms are all m=0
    pwr_spec_m0 = np.sum(np.abs(real_modes[:lmax]**2 + imag_modes[:lmax]**2))
    pwr_spec_rest = np.sum(np.abs(real_modes[lmax:]**2 + imag_modes[lmax:]**2)*2)
    rms = np.sqrt((pwr_spec_m0+pwr_spec_rest)/4.0/np.pi)

    return rms



def create_ytrain(pointing_per_cone, pointing_nside, defocus_per_cone, num_defocus, power_per_cone, num_powers):

    Y_train = np.hstack((np.array(pointing_per_cone)/(pointing_nside-1), np.array(defocus_per_cone)/(num_defocus-1)))
    Y_train = np.hstack((Y_train, np.array(power_per_cone)/(num_powers-1)))
    Y_norms = [pointing_nside, num_defocus, num_powers]

    return Y_train, Y_norms



def imap_norm(intensity_map):

    avg_flux = np.mean(intensity_map) # average power per steradian (i.e. a flux)
    intensity_map_normalized = intensity_map / avg_flux - 1.0

    return intensity_map_normalized, avg_flux