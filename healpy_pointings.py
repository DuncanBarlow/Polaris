import utils_deck_generation as idg
import numpy as np
import healpy as hp

small_num = 1.0e-10


def pointing_rotations(the_data, quad_slice, pointing_nside, surface_cover_degrees):
    npixels = hp.nside2npix(pointing_nside)
    img = np.linspace(0, npixels, num=npixels)
    index = np.arange(npixels)
    theta, phi = hp.pix2ang(pointing_nside,index)

    vec = hp.ang2vec(np.pi / 2, 0.0)
    pointing_ind = hp.query_disc(nside=pointing_nside, vec=vec,
                                 radius=np.radians(surface_cover_degrees))

    gamma = theta[pointing_ind] - np.pi /2
    beta = phi[pointing_ind]
    cos_g = np.cos(gamma)
    sin_g = np.sin(gamma)
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)

    delta = np.mean(np.radians(the_data['Theta'][quad_slice])) - np.pi /2
    phi = np.mean(np.radians(the_data['Phi'][quad_slice]))
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    term1 = cos_d * cos_b * cos_g - sin_d * sin_g

    x = the_data['target_radius'] * (cos_p * term1 - sin_p * sin_b * cos_g)
    y = the_data['target_radius'] * (sin_p * term1 + cos_p * sin_b * cos_g)
    z = the_data['target_radius'] * (-sin_d * cos_b * cos_g - cos_d * sin_g)

    x = np.where(abs(x) < small_num, 0.0, x)
    y = np.where(abs(y) < small_num, 0.0, y)
    z = np.where(abs(z) < small_num, 0.0, z)
    
    return x, y, z, pointing_ind



def rot_mat(theta, axis):
    mat = np.zeros([3,3])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    if (abs(sin_t) < small_num):
        sin_t = 0.0
        cos_t = 1.0
    if (abs(cos_t) < small_num):
        cos_t = 0.0
        sin_t = 1.0

    if (axis == "x"):
        mat[0,0] = 1.0
        mat[1,1] = cos_t
        mat[2,2] = cos_t
        mat[1,2] = -sin_t
        mat[2,1] = sin_t
    elif (axis == "y"):
        mat[0,0] = cos_t
        mat[1,1] = 1.0
        mat[2,2] = cos_t
        mat[0,2] = sin_t
        mat[2,0] = -sin_t
    elif (axis == "z"):
        mat[0,0] = cos_t
        mat[1,1] = cos_t
        mat[2,2] = 1.0
        mat[0,1] = -sin_t
        mat[1,0] = sin_t
    else:
        print("Invalid parameter 'axis' try setting string 'x', 'y', or 'z'")

    return mat