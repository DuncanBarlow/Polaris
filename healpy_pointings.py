import utils_deck_generation as idg
import numpy as np

small_num = 1.0e-10



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



def square2disk(a, b):
    """
    Shirley, P., Chiu, K. 1997. “A Low Distortion Map Between Disk and Square”
    Journal of Graphics Tools, volume 2 number 3.
    """

    if (a > -b):
        if (a > b):
            r = a
            phi = (np.pi / 4.0) * (b / a)
        else:
            r = b
            phi = (np.pi / 4.0) * (2.0 - (a / b))
    else:
        if (a < b):
            r = -a
            phi = (np.pi / 4.0) * (4.0 - (a / b))
        else:
            r = -b
            if (b != 0):
                phi = (np.pi / 4.0) * (6.0 - (a / b))
            else:
                phi = 0.0

    return r, phi