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
    """ Flip a stokes matrix so that the h-polarization vector is flipped"""
    stokes_out = copy.copy(stokes_in)
    stokes_out[2:, :] = -1. * stokes_in[2:, :]
    return stokes_out


def phase_diff(stokes):
    """
    Compute phase difference, relative to VV, given a Stoke's matrix

    Returns a 3-element list:      [phi_hh_vv, phi_vh_vv, phi_hv_vv]
    """
    if (stokes[0, 3] != 0) or (stokes[0, 2] != 0):
        phi_vh_vv = np.arctan2(stokes[0, 3], stokes[0, 2])
    else:
        phi_vh_vv = 0.

    if (stokes[3, 0] != 0) or (stokes[2, 0] != 0):
        phi_hv_vv = np.arctan2(-stokes[3, 0], stokes[2, 0])
    else:
        phi_hv_vv = 0.

    x = -2. * stokes[0, 0] * stokes[3, 2] + stokes[2, 0] * stokes[0, 3] + stokes[0, 2] * stokes[3, 0]
    y = 2. * stokes[0, 0] * stokes[2, 2] - stokes[2, 0] * stokes[0, 2] + stokes[0, 3] * stokes[3, 0]
    phi_hh_vv = np.arctan2(x, y)

    return [np.rad2deg(phi_hh_vv), np.rad2deg(phi_vh_vv), np.rad2deg(phi_hv_vv)]


# *********************************************************************************
# Error handling
# *********************************************************************************

class CodeError(Exception):

    def __init__(self, error_text, logfile):
        if logfile is not None:
            writeText("ERROR: " + error_text, logfile)
        else:
            print error_text

        Exception.__init__(self)


def warning(inputText, logfile):
    writeText("WARNING: " + inputText, logfile)


def writeText(inputText, logfile):
    print inputText
    if logfile is not None:
        logfile.write(inputText)
        logfile.flush()


if __name__== "__main__":
    # test flip_stokes
    a = np.ones((4, 4))
    print a
    print flip_stokes(a)