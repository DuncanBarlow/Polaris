import glob
import numpy as np
import healpy as hp

def import_data(data_location, map_nside):
    files=glob.glob(data_location+'/p_in_*.txt')
    f= files[0]
    with open(f,'r') as ff:
        line = ff.readline()
        data = np.array(line.split(),dtype=np.int)
        npix = data[0]
        lines=ff.readlines()
        data = [line.split() for line in lines]
        data = np.array(data)

    power = np.zeros((npix))
    thetas = np.array(data[:,6],dtype=np.float)
    phis = np.array(data[:,7],dtype=np.float)
    n_beams = 0
    for f in files:
        b='Reading from: '+f+"  "
        print("\r", b, end="")
        with open(f,'r') as ff:
            line = ff.readline()
            lines=ff.readlines()
            data = [line.split() for line in lines]
            data = np.asfarray(data)
            power = power + data[:,2]
        n_beams = n_beams + 1

    nsources = len(power)
    npix = hp.nside2npix(map_nside)
    indices = hp.ang2pix(map_nside, thetas, phis)
    hpxmap = np.zeros(npix, dtype=np.float64)
    counts = np.zeros(npix)
    for i in range(nsources):
        hpxmap[indices[i]] = hpxmap[indices[i]] + power[i]
        counts[indices[i]] = counts[indices[i]] + 1
    index = np.where(counts > 0.)[0]
    #hpxmap[index] = hpxmap[index]/counts[index]
    index = np.where(counts == 0.)[0]
    hpxmap[index] = hp.pixelfunc.UNSEEN
    
    return hpxmap, n_beams