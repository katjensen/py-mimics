import numpy as np




def eps_soil(freq, temp, rho_b, sand_frac, clay_frac, mv):
    """ Relative Dielectric Constant of SOIL
    based on: Ulaby (Section 4-8)
    eps_soil = eps_r - j*eps_i

    Method computes the real and imaginary parts of the relative dielectric constant of soil at a given
    temperature 0 < t < 40C, frequency, volumetric moisture content, soil bulk density, sand and clay fractions.

    Input Variables:
        freq:       frequency, GHz
        temp:       temperature, degrees C
        rho_b:      bulk density, g/cm3 (typical value is 1.7 g/cm3)
        sand_frac:  sand fraction (0 < s < 1)
        clay_frac:  clay fraction (0 < c < 1)
        mv:         volumetric water content of soil, 0 < mv < 1

    Output Products:
        eps_r:      real part of dielectric constant
        eps_i:      imaginary part of dielectric constant
    """

    freq_hz = freq * 1.0e9                                  # convert GHz to Hz
    alpha = 0.65                                            # eq: 4.68a
    beta1 = 1.27 - 0.519 * sand_frac - 0.152 * clay_frac    # eq: 4.68b
    beta2 = 2.06 - 0.928 * sand_frac - 0.255 * clay_frac    # eq: 4.68c
    eps_0 = 8.854e-12                                       # permittivity of free space

    if freq > 1.3:
        sigma_s = -1.645 + 1.939 * rho_b - 2.256 * sand_frac + 1.594 * clay_frac    # eq. 4.68d
    elif (freq >= 0.3) and (freq <= 1.3):
        sigma_s = 0.0467 + 0.22 * rho_b - 0.411 * sand_frac + 0.661 * clay_frac     # eq. 4.70
    else:
        raise Exception("Selected frequency is not supported in dielectric constant calculation.")

    # Dielectric constant of pure water -------------------------------------------------------------------------------
    epsw_inf = 4.9  # eq: 4.15, magnitude of high-frequency eps_w

    # Static dielectric constant (at f = 0), dimensionless; eq. 4.18, Klein & Swift 1977
    epsw_0 = 88.045 - 0.4147 * temp + 6.295e-4 * np.power(temp, 2) + 1.075e-5 * np.power(temp, 3)

    # Relaxation time consant (s) of pure water; eq. 4.16
    tau_w = (1.1109e-10 - 3.824e-12 * temp + 6.938e-14 * np.power(temp, 2) - 5.096e-16 * np.power(temp, 3)) / \
            (2 * np.pi)

    # dielectric constant of water, real & imaginary parts; single Debye model, section 4-1
    epsw_r = epsw_inf + (epsw_0 - epsw_inf) / (1. + np.power(2. * np.pi * freq_hz * tau_w, 2))  # eq. 4.67a
    epsw_i = (2. * np.pi * freq_hz * tau_w * (epsw_0 - epsw_inf)) / (1. + np.power(2. * np.pi * freq_hz * tau_w, 2)) + \
             (2.65 - rho_b) / (2.65 * mv) * (sigma_s / (2. * np.pi * eps_0 * freq_hz))         # eq. 4.67b

    # Dielectric constant of soil -------------------------------------------------------------------------------------
    eps_r = np.power((1 + 0.66 * rho_b + np.power(mv, beta1) * np.power(epsw_r, alpha) - mv), (1 / alpha))
    eps_i = np.power(mv, beta2) * epsw_i

    return eps_r, eps_i
