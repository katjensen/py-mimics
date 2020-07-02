import numpy as np
import util



def eps_water(temp, freq, s):
    """
    Relative Dielectric Constant of Pure Water
    based on Ulaby & Long "Microwave Radar and Radiometric Remote Sensing" 2014 (Section 4-2)

    Computes the real and imaginary parts of the relative dielectric constant of water using
    the double-Debye (D3M) model (Matzler, 2006). Applicable for the following conditions:
        0 <= temp <= 30 C
        0 <= s <= 40 ppt
        0 <= f <= 1000 GHz

    Parameters:
        temp:       temperature in C
        freq:       frequency in GHz
        s:          salinity in parts per thousand

    Output:
        eps_r:      real part of relative dielectric constant
        eps_i:      imaginary part of relative dielectric constant
    """
    a = [0.46606917e-2, -0.26087876e-4, -0.63926782e-5, 0.63000075e1, 0.26242021e-2, -0.42984155e-2,
         0.34414691e-4, 0.17667420e-3, -0.20491560e-6, 0.58366888e3, 0.12634992e3, 0.69227972e-4, 0.38957681e-6,
         0.30742330e3, 0.12634992e3, 0.37245044e1, 0.92609781e-2, -0.26093754e-1]


    # Conductvity
    sigma35 = 2.903602 + 8.607e-2 * temp + 4.738817e-4 * np.power(temp, 2) - 2.991e-6 * np.power(temp, 3) + \
              4.3041e-9 * np.power(temp, 4)                                                                 # eq. 4.21g
    p = s * (37.5109 + 5.45216 * s + 0.014409 * np.power(s, 2)) / (1004.75 + 182.283 * s + np.power(s, 2))  # eq. 4.21h
    a0 = (6.9431 + 3.2841 * s - 0.099486 * np.power(s, 2)) / (84.85 + 69.024 * s + np.power(s, 2))          # eq. 4.21j
    a1 = 49.843 - 0.2276 * s + 0.00198 * np.power(s, 2)                                                     # eq. 4.21k
    q = 1. + (a0 * (temp - 15.)) / (temp + a1)                                                              # eq. 4.21i
    sigma = sigma35 * p * q     # eq. 4.21f

    epsw0 = 87.85306 * np.exp(-0.00456992 * temp - a[0] * s - a[1] * np.power(s, 2) - a[2] * s * temp)      # eq. 4.21a
    epsw1 = a[3] * np.exp(-a[4] * temp - a[5] * s - a[6] * temp * s)                                        # eq. 4.21b
    tau1 = (a[7] + a[8] * s) * np.exp(a[9] / (temp + a[10]))                                                # eq. 4.21c
    tau2 = (a[11] + a[12] * s) * np.exp(a[13] / (temp + a[14]))                                             # eq. 4.21d
    epsw_inf = a[15] + a[16] * temp + a[17] * s                                                             # eq. 4.21e

    # Calculate real and imaginary parts of epsilon
    #eps_r = epsw_inf + ((epsw0 - epsw1) / (1. + np.power(2. * np.pi * freq * tau1, 2))) + \
    #        ((epsw0 - epsw_inf) / (1. + np.power(2. * np.pi * freq * tau2, 2)))                          # eq. 4.19a
    #eps_i = ((2. * np.pi * freq * tau1 * (epsw0 - epsw1)) / (1. + np.power(2. * np.pi * freq * tau1, 2))) + \
    #        ((2. * np.pi * freq * tau2 * (epsw0 - epsw_inf)) / (1. + np.power(2. * np.pi * freq * tau2, 2))) + \
    #        (sigma / (2. * np.pi * eps0 * freq))
    # eq. 4.19b

    # Complex Permitivity Calculation
    eps = ((epsw0 - epsw1) / (1 - 1j * 2 * np.pi * freq * tau1)) + \
          ((epsw1 - epsw_inf) / (1. - 1j * 2 * np.pi * freq * tau2)) + \
          epsw_inf + 1j * ((17.9751 * sigma) / freq)

    eps_r = np.real(eps)
    eps_i = np.imag(eps)

    return eps_r, eps_i


def eps_water_single_debye(temp, freq):
    """
    Relative dielectric constant of Pure Water
    Computes the real and imaginary parts of the relative dielectric constant of pure water under
    these conditions:
        freq <= 50 GHz
        0 <= temp <= 30 degrees C

    Based on the single Debye model, found in Ulaby & Long 2014 (Section 4-1)

    Parameters:
        temp:       temperature in C
        freq:       frequency in GHz

    Output:
        eps_r:      real part of relative dielectric constant
        eps_i:      imaginary part of relative dielectric constant
    """
    freq_hz = freq * 1.0e9  # convert GHz to Hz

    epsw_inf = 4.9  # eq: 4.15, magnitude of high-frequency eps_w

    # Static dielectric constant (at f = 0), dimensionless; eq. 4.18, Klein & Swift 1977
    epsw_0 = 88.045 - 0.4147 * temp + 6.295e-4 * np.power(temp, 2) + 1.075e-5 * np.power(temp, 3)

    # Relaxation time consant (s) of pure water; eq. 4.16
    tau_w = (1.1109e-10 - 3.824e-12 * temp + 6.938e-14 * np.power(temp, 2) - 5.096e-16 * np.power(temp, 3)) / \
            (2 * np.pi)

    # dielectric constant of water, real & imaginary parts; single Debye model, section 4-1
    epsw_r = epsw_inf + (epsw_0 - epsw_inf) / (1. + np.power(2. * np.pi * freq_hz * tau_w, 2))  # eq. 4.67a
    epsw_i = (2. * np.pi * freq_hz * tau_w * (epsw_0 - epsw_inf)) / \
             (1. + np.power(2. * np.pi * freq_hz * tau_w, 2))
    return epsw_r, epsw_i


def eps_soil(freq_GHz, temp, sand_frac, clay_frac, mv, rho_b=1.7, logfile=None):
    """
    Relative Dielectric Constant of SOIL
    based on: Ulaby & Long "Microwave Radar and Radiometric Remote Sensing" 2014 (Section 4-8)

    Implemented from the following paper:
    M.C. Dobson, F.F. Ulaby, M.T. Hallikainen, M.A. El-Rayes, "Microwave dielectric behavior of wet soil - Part II:
    Dielectric mixing models," IEEE. Trans. Geosci. Remote Sens., vol. 23, no. 1, pp. 35-46, 1985.

    Method computes the real and imaginary parts of the relative dielectric constant of soil at a given
    temperature 0 < t < 40C, frequency, volumetric moisture content, soil bulk density, sand and clay fractions.

    eps_soil = eps_r - j*eps_i

    Parameters:
        freq_GHz:   frequency, GHz
        temp:       temperature, degrees C
        sand_frac:  sand fraction, 0 - 1.
        clay_frac:  clay fraction, 0 - 1.
        mv:         volumetric water content of soil, 0 - 1.
        rho_b:      bulk density, g/cm3 (typical value is 1.7 g/cm3)

    Returns:
        eps_r:      real part of dielectric constant
        eps_i:      imaginary part of dielectric constant
    """
    # Non-frozen water in soil
    if temp > 40:
        util.CodeError("Soil water temperature is too hot! Must be less than 40 C", logfile)

    elif temp > 0:
        freq_hz = freq_GHz * 1.0e9                              # convert GHz to Hz
        alpha = 0.65                                            # eq: 4.68a, optimized coefficient
        beta1 = 1.27 - 0.519 * sand_frac - 0.152 * clay_frac    # eq: 4.68b, optimized coefficient
        beta2 = 2.06 - 0.928 * sand_frac - 0.255 * clay_frac    # eq: 4.68c, optimized coefficient

        # Effective conductivity, empirically derived -----------------------------------------------------------------
        if freq_GHz > 1.3:
            sigma_s = -1.645 + 1.939 * rho_b - 2.256 * sand_frac + 1.594 * clay_frac    # eq. 4.68d
        elif (freq_GHz >= 0.3) and (freq_GHz <= 1.3):
            sigma_s = 0.0467 + 0.22 * rho_b - 0.411 * sand_frac + 0.661 * clay_frac     # eq. 4.70
        else:
            util.CodeError("Selected frequency is not supported in dielectric constant calculation.", logfile)

        # Dielectric constant of pure water ----------------------------------------------------------------------------
        epsw_inf = 4.9  # eq: 4.15, high-frequency limit of free water dieletric constant

        # Static dielectric constant (at f = 0), dimensionless; eq. 4.18, Klein & Swift 1977
        epsw_0 = 88.045 - 0.4147 * temp + 6.295e-4 * np.power(temp, 2) + 1.075e-5 * np.power(temp, 3)

        # Relaxation time consant (s) of pure water; eq. 4.16
        tau_w = (1.1109e-10 - 3.824e-12 * temp + 6.938e-14 * np.power(temp, 2) - 5.096e-16 * np.power(temp, 3)) / \
                (2 * np.pi)

        # dielectric constant of water, real & imaginary parts; single Debye model, section 4-1 + addded conductivity term
        epsw_r = epsw_inf + (epsw_0 - epsw_inf) / (1. + np.power(2. * np.pi * freq_hz * tau_w, 2))      # eq. 4.67a
        epsw_i = (2. * np.pi * freq_hz * tau_w * (epsw_0 - epsw_inf)) / \
                 (1. + np.power(2. * np.pi * freq_hz * tau_w, 2)) + (2.65 - rho_b) / (2.65 * mv) * \
                 (sigma_s / (2. * np.pi * util.CONSTANTS['eps0'] * freq_hz))                           # eq. 4.67b

        # Dielectric constant of soil ----------------------------------------------------------------------------------
        eps_r = np.power((1 + 0.66 * rho_b + np.power(mv, beta1) * np.power(epsw_r, alpha) - mv), (1 / alpha))  # eq. 4.66a
        eps_i = np.power(mv, beta2) * epsw_i                                                                    # eq. 4.66b


    else:   # frozen soil
        util.CodeError("Frozen soils are not currently supported by model", logfile)
        #eps_r = 3. + mv * (20. + (2./3.) * temp)           # from Kyle's code, dielectric.f, ln 1203-04
        #eps_i = -1. * (1. + temp / 50.) * 2. * mv / 0.15

    return eps_r, eps_i




if __name__ == "__main__":
    T = 20
    f = 10
    r, i = eps_water_single_debye(T, f)
    print r, i

    r, i = eps_water(T, f, 0.)
    print r, i

    r, i = eps_soil(f, T, rho_b=1.7, sand_frac=0.2, clay_frac=0.2, mv=0.2)
    print r, i