############################################
# Ancillary methods
############################################
import copy
import numpy as np

CONSTANTS = {
    "c":    299792458,      # speed of light, m/s
    "eps0": 8.854e-12       # permittivity of free space
}


def freq2wavelength(freq):
    """
    Converts frequency to wavelength

    Parameters:
        freq:       float, GHz

    Output:
        wavelength: float, meters
    """
    return CONSTANTS["c"] / (freq * 1e9)


def flip_stokes(stokes_in):
    """
    Flip a stokes matrix so that the h-polarization vector is flipped"""
    stokes_out = copy.copy(stokes_in)
    stokes_out[2:, :] = -1. * stokes_in[2:, :]
    return stokes_out



if __name__== "__main__":
    # test flip_stokes
    a = np.ones((4, 4))
    print a
    print flip_stokes(a)