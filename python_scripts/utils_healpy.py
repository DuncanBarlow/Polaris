import numpy as np
import healpy as hp
import os
import utils_intensity_map as uim


def power_spectrum(intensity_map, LMAX, verbose=True):
    intensity_map_normalized, avg_power = uim.imap_norm(intensity_map)
    alms = hp.sphtfunc.map2alm(intensity_map_normalized, lmax=LMAX)
    power_spectrum = alms2power_spectrum(alms, LMAX)

    if verbose:
        print("The rms is: ", np.sqrt(np.sum(power_spectrum))*100.0, "%")
    sqrt_power_spectrum = np.sqrt(power_spectrum)

    return sqrt_power_spectrum



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



def imap2modes(intensity_map_normalized, LMAX):

    modes_complex = hp.sphtfunc.map2alm(intensity_map_normalized, lmax=LMAX)

    return modes_complex.real, modes_complex.imag



def modes2imap(real_modes, imag_modes, imap_nside):

    np_complex = np.vectorize(complex)
    modes_complex = np_complex(real_modes, imag_modes)
    intensity_map_normalized = hp.alm2map(modes_complex, imap_nside)

    return intensity_map_normalized



def change_number_modes(Y_train, avg_powers_all, LMAX):

    num_examples = np.shape(Y_train)[1]
    Y_train2 = np.zeros((LMAX, num_examples))
    num_coeff = int(((LMAX + 2) * (LMAX + 1))/2.0)
    np_complex = np.vectorize(complex)
    for ie in range(num_examples):
        Y_train_complex = np_complex(Y_train_real[:num_coeff], Y_train_real[num_coeff:])

        power_spectrum = alms2power_spectrum(Y_train_complex, LMAX)
        Y_train2[:,ie] = np.sqrt(power_spectrum)

    return Y_train2



def extract_modes_and_flux(intensity_map, LMAX):

    intensity_map_normalized, avg_flux = uim.imap_norm(intensity_map)
    real_modes, imag_modes = imap2modes(intensity_map_normalized, LMAX, avg_flux)

    return real_modes, imag_modes, avg_flux