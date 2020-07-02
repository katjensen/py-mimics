############################################
# Ancillary methods
############################################
from collections import defaultdict
import copy
import datetime
import numpy as np

CONSTANTS = {
    "c":    299792458,      # speed of light, m/s
    "eps0": 8.854e-12       # permittivity of free space
}


def pow2db(sigma0):
    return 10. * np.log10(sigma0)


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


def merge_dictionaries(dictionary_list):
    super_dict = {}
    for d in dictionary_list:
        super_dict.update(d)
    return super_dict


def mueller(s):
    """
    Calculates the 4x4 modified Mueller matrix of a 2x2 input scattering matrix
    See: Ulaby & Long 2015, pg 169

    Reminder:
    S = [s00, s01
         s10, s11]

    where:  s00 = sVV,
            s01 = sVH,
            s10 = sHV,
            s11 = sHH

    """
    sVV = s[0, 0]
    sVH = s[0, 1]
    sHV = s[1, 0]
    sHH = s[1, 1]

    m = np.zeros((4, 4), dtype=np.complex128)

    m[0, 0] = np.abs(sVV) * np.abs(sVV)
    m[0, 1] = np.abs(sVH) * np.abs(sVH)
    m[0, 2] = np.real(sVV * np.conj(sVH))
    m[0, 3] = -np.imag(sVV * np.conj(sVH))

    m[1, 0] = np.abs(sHV) * np.abs(sHV)
    m[1, 1] = np.abs(sHH) * np.abs(sHH)
    m[1, 2] = np.real(sHV * np.conj(sHH))
    m[1, 3] = -np.imag(sHV * np.conj(sHH))

    m[2, 0] = 2 * np.real(sVV * np.conj(sHV))
    m[2, 1] = 2 * np.real(sVH * np.conj(sHH))
    m[2, 2] = np.real(sVV * np.conj(sHH) + sVH * np.conj(sHV))
    m[2, 3] = -np.imag(sVV * np.conj(sHH) - sVH * np.conj(sHV))

    m[3, 0] = 2 * np.imag(sVV * np.conj(sHV))
    m[3, 1] = 2 * np.imag(sVH * np.conj(sHH))
    m[3, 2] = np.imag(sVV * np.conj(sHH) + sVH * np.conj(sHV))
    m[3, 3] = np.real(sVV * np.conj(sHH) - sVH * np.conj(sHV))

    return m


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
    phi_hh_vv = np.arctan2(np.real(x), np.real(y))

    return [np.rad2deg(phi_hh_vv), np.rad2deg(phi_vh_vv), np.rad2deg(phi_hv_vv)]


def str2date(datestr, fmt='%m/%d/%Y'):
    return datetime.datetime.strptime(datestr, fmt)

def update_column(dat, column_name, updated_values):
    dat[column_name] = updated_values
    return dat


# *********************************************************************************
# Error handling
# *********************************************************************************

class CodeError(Exception):

    def __init__(self, error_text, logfile=None):
        if logfile is not None:
            write_text('ERROR: ' + error_text + '\n' + '*** SIMULATION STOPPED ***', logfile)
        else:
            print error_text

        Exception.__init__(self)


def warning(inputText, logfile):
    write_text('WARNING: ' + inputText, logfile)


def write_text(inputText, logfile):
    print inputText
    if logfile is not None:
        logfile.write(inputText)
        logfile.flush()


if __name__== '__main__':
    # test flip_stokes
    a = np.ones((4, 4))
    print a
    print flip_stokes(a)