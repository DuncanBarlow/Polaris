import numpy as np
import healpy as hp
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



def readout_intensity(the_data, intensity_map, mean_power_fraction=-1.0, file_location="."):
    n_beams = the_data['nbeams']
    total_TW = np.mean(intensity_map)*10**(-12) * 4.0 * np.pi
    mean_intensity = np.mean(intensity_map) / (the_data['target_radius'] / 10000.0)**2

    #rms
    intensity_map_normalised, avg_power = imap_norm(intensity_map)
    imap_pn = np.sign(intensity_map_normalised)
    intensity_map_rms = 100.0 * np.sqrt(np.mean(intensity_map_normalised**2))
    intensity_map_rms_spatial = imap_pn * 100.0 * np.abs(intensity_map_normalised)

    print_line = []
    print('')
    print_line.append('RMS is {:.4f}%, '.format(intensity_map_rms))
    print_line.append('Number of beams ' + str(n_beams))
    print_line.append('Mean intensity is {:.2e}W/cm^2, '.format(mean_intensity))
    print_line.append('The total power deposited is {:.2f}TW, '.format(total_TW))
    print_line.append('The power per beam deposited is {:.4f}TW, '.format(total_TW / n_beams))
    if mean_power_fraction > 0.0:
        print_line.append('This is a drive efficiency of {:.2f}%, '.format(total_TW / (n_beams * the_data['default_power'] * mean_power_fraction) * 100.0))
        print_line.append('Mean power percentage {:.2f}%, '.format(mean_power_fraction * 100.0))
    print('')

    file1 = open(file_location+"/stats.txt","a")
    for line in range(len(print_line)):
        print(print_line[line])
        file1.writelines(print_line[line]+"\n")
    file1.close()

    return intensity_map_rms_spatial



def heatsource_analysis(hs_and_modes):

    avg_flux = hs_and_modes["average_flux"][0]
    real_modes = hs_and_modes["complex_modes"][0,:]
    imag_modes = hs_and_modes["complex_modes"][1,:]

    return real_modes, imag_modes, avg_flux




def extract_run_parameters(dataset_params, facility_spec, sys_params, file_location="."):

    total_power = 0
    print_line = []

    for icone in range(facility_spec['num_cones']):
        ind_cone_start = icone * dataset_params["num_sim_params"]

        cone_theta_offset = float(
            dataset_params["sim_params"][ind_cone_start+dataset_params["theta_index"]])
        cone_phi_offset = float(
            dataset_params["sim_params"][ind_cone_start+dataset_params["phi_index"]])
        if dataset_params["defocus_bool"]:
            cone_defocus = float(
                dataset_params["sim_params"][ind_cone_start+dataset_params["defocus_index"]])
        else:
            cone_defocus = dataset_params["defocus_default"]
        cone_powers = float(
            dataset_params["sim_params"][ind_cone_start+dataset_params["power_index"]])

        beams_per_cone = facility_spec['beams_per_cone'][icone]
        total_power += cone_powers * beams_per_cone

        if icone < int(facility_spec['num_cones']/2):
            print_line.append("For cone " + str(icone+1) +
                  ": {:.2f}\N{DEGREE SIGN}, ".format(np.degrees(cone_theta_offset)) +
                  "{:.2f}\N{DEGREE SIGN}, ".format(np.degrees(cone_phi_offset)) +
                  "{:.2f}mm, ".format(cone_defocus) +
                  "{:.2f}% power, ".format(cone_powers * 100))
    mean_power_fraction = total_power / facility_spec['nbeams']

    if os.path.exists(file_location+"/stats.txt"):
        os.remove(file_location+"/stats.txt")
    file1 = open(file_location+"/stats.txt","a")
    for line in range(len(print_line)):
        print(print_line[line])
        file1.writelines(print_line[line]+"\n")
    file1.close()

    return mean_power_fraction


def alms2power_spectrum(alms, LMAX):

    the_modes = np.zeros(LMAX)
    the_modes_full = np.zeros((LMAX,LMAX+1))
    for l in range(LMAX):
        for m in range(l+1):
            the_modes_full[l,m] = np.real(alms[hp.sphtfunc.Alm.getidx(LMAX, l, m)]*
                np.conjugate(alms[hp.sphtfunc.Alm.getidx(LMAX, l, m)]))
            if (m>0):
                the_modes[l] = the_modes[l] + 2.*the_modes_full[l,m]
            else:          
                the_modes[l] = the_modes[l] + the_modes_full[l,m]

    the_modes = the_modes / (4.*np.pi)

    return the_modes



def alms2rms(real_modes, imag_modes, lmax):

    # modes with m!=0 need to be x2 to account for negative terms
    # in healpix indexing the first lmax terms are all m=0
    pwr_spec_m0 = np.sum(np.abs(real_modes[:lmax]**2 + imag_modes[:lmax]**2))
    pwr_spec_rest = np.sum(np.abs(real_modes[lmax:]**2 + imag_modes[lmax:]**2)*2)
    rms = np.sqrt((pwr_spec_m0+pwr_spec_rest)/4.0/np.pi)

    return rms



def power_spectrum(intensity_map, LMAX, verbose=True):
    intensity_map_normalized, avg_power = imap_norm(intensity_map)
    alms = hp.sphtfunc.map2alm(intensity_map_normalized, lmax=LMAX)
    power_spectrum = alms2power_spectrum(alms, LMAX)

    if verbose:
        print("The LLE quoted rms cumulative over all modes is: ", np.sqrt(np.sum(power_spectrum))*100.0, "%")
    sqrt_power_spectrum = np.sqrt(power_spectrum)

    return sqrt_power_spectrum



def create_ytrain(pointing_per_cone, pointing_nside, defocus_per_cone, num_defocus, power_per_cone, num_powers):

    Y_train = np.hstack((np.array(pointing_per_cone)/(pointing_nside-1), np.array(defocus_per_cone)/(num_defocus-1)))
    Y_train = np.hstack((Y_train, np.array(power_per_cone)/(num_powers-1)))
    Y_norms = [pointing_nside, num_defocus, num_powers]

    return Y_train, Y_norms



def extract_modes_and_flux(intensity_map, LMAX):

    intensity_map_normalized, avg_flux = imap_norm(intensity_map)
    real_modes, imag_modes = imap2modes(intensity_map_normalized, LMAX, avg_flux)

    return real_modes, imag_modes, avg_flux



def imap2modes(intensity_map_normalized, LMAX, avg_power):

    modes_complex = hp.sphtfunc.map2alm(intensity_map_normalized, lmax=LMAX)

    return modes_complex.real, modes_complex.imag



def modes2imap(real_modes, imag_modes, imap_nside):

    np_complex = np.vectorize(complex)
    modes_complex = np_complex(real_modes, imag_modes)
    intensity_map_normalized = hp.alm2map(modes_complex, imap_nside)

    return intensity_map_normalized



def imap_norm(intensity_map):

    avg_flux = np.mean(intensity_map) # average power per steradian (i.e. a flux)
    intensity_map_normalized = intensity_map / avg_flux - 1.0

    return intensity_map_normalized, avg_flux



def change_number_modes(Y_train, avg_powers_all, LMAX):

    num_examples = np.shape(Y_train)[1]
    Y_train2 = np.zeros((LMAX, num_examples))
    num_coeff = int(((LMAX + 2) * (LMAX + 1))/2.0)
    np_complex = np.vectorize(complex)
    for ie in range(num_examples):
        # weighting to allow NN to adjust for mean flux
        #Y_train_real = np.squeeze(Y_train[:,ie] / avg_powers_all[ie])
        Y_train_complex = np_complex(Y_train_real[:num_coeff], Y_train_real[num_coeff:])

        power_spectrum = alms2power_spectrum(Y_train_complex, LMAX)
        Y_train2[:,ie] = np.sqrt(power_spectrum)

    return Y_train2