import numpy as np
import healpy as hp


def readout_intensity(the_data, intensity_map):
    n_beams = the_data['nbeams_total']
    r = the_data['target_radius'] / 1.0e4 # microns to cm
    total_TW = np.mean(intensity_map)*10**(-12)
    surface_area = 4.0 * np.pi * r**2

    #rms
    avg_power = np.mean(intensity_map)
    intensity_map_rms = 100.0 * np.sqrt(np.mean((intensity_map / avg_power - 1.0)**2))
    intensity_map_rms_spatial = 100.0 * np.sqrt((intensity_map / avg_power - 1.0)**2)

    print('')
    print('The total power deposited is ', total_TW * surface_area, 'TW')
    print('The power per beam deposited is ', total_TW * surface_area / n_beams, 'TW')
    print('This is a drive efficiency of ', total_TW * surface_area / (n_beams * 0.25) * 100.0, '%')
    print('RMS is ', intensity_map_rms, '%')
    print('Number of beams ', n_beams)
    print('')

    return intensity_map_rms_spatial



def power_spectrum(intensity_map, LMAX):
    avg_power = np.mean(intensity_map)
    intensity_map_normalized = (intensity_map / avg_power - 1.0)

    # Compute the corresponding normalized mode spectrum
    rmsalms = hp.sphtfunc.map2alm(intensity_map_normalized, lmax=LMAX)
    var = abs(rmsalms)**2
    the_modes = np.zeros(LMAX)
    power_spectrum = np.zeros(LMAX)
    for l in range(LMAX):
        for m in range(l):
            if (m>0):
                the_modes[l] = the_modes[l] + 2.*var[hp.sphtfunc.Alm.getidx(LMAX, l, m)]
            else:          
                the_modes[l] = the_modes[l] + var[hp.sphtfunc.Alm.getidx(LMAX, l, m)]
        power_spectrum[l] = (2.0 * l + 1.0) * the_modes[l] / (4.0 * np.pi)

    power_spectrum_unweighted = np.sqrt(the_modes)
    power_spectrum_weighted = np.sqrt(power_spectrum)
    print("The LLE quoted rms cumalitive over all modes is: ", np.sqrt(np.sum(the_modes**2)), "%")

    return power_spectrum_unweighted, power_spectrum_weighted