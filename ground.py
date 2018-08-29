import numpy as np

import ancil
import dielectric


class Ground():
    _compute_mm = True

    def __init__(self, epsilon, theta, rms, ls, freq, surface_model):
        self.epsilon = epsilon                          # dielectric of soil relative to snow or to free space, complex
        self.theta = theta                              # local incidence angle, radians
        self.rms = rms                                  # RMS height, cm
        self.ls = ls                                    # Correlation length, cm
        self.freq = freq                                # sensor frequency, GHz
        self.wavelength = ancil.freq2wavelength(freq)   # sensor wavelength, meters
        self.surface_model = surface_model              # surface scattering model (1-5)
        self.calc_back = True                           # need to change this in the future!! to allow forward scatter

        self.refl_matrix = None
        self.back_matrix = None

        # Look-up hashtable for surface scattering models
        self.surface_models_LUT = {0:   self.specular,
                                   1:   self.geometric_optics,
                                   2:   self.physical_optics,
                                   3:   self.small_perturbation,
                                   4:   self.umich_empirical }


    def solve(self):
        print ".... working on Ground Layer ..."

        # Compute reflectivity matrix of specular surface
        self.refl_matrix = self.specular_reflection()

        # Compute backscattering matrix of rough ground surface
        self.back_matrix = ancil.flip_stokes(self.backscatter())

        # If snow layer present ## need to implement this!!
        # lines 108-115 (ground_layer.f)

        # Phase difference
        ground_spec_phase_diff = ancil.phase_diff(self.refl_matrix)
        ground_back_phase_diff = ancil.phase_diff(self.back_matrix)



    def specular_reflection(self):
        """
        Computes the reflectivity matrix of a specular ground surface

        see: McDonald dissertation (eq. 3.20)
        """
        stheta = np.sin(self.theta)
        ctheta = np.cos(self.theta)
        cwork = np.sqrt(self.epsilon - np.power(stheta, 2))

        # Fresnel reflection coefficients
        rperp = (ctheta - cwork) / (ctheta + cwork)                                 # perpendicular pol
        rpar = (self.epsilon * ctheta - cwork) / (self.epsilon * ctheta + cwork)    # parallel pol

        # Reflectivity matrix
        work1 = np.absolute(rpar)
        work2 = np.absolute(rperp)
        cwork = rpar * np.conj(rperp)
        work3 = np.real(cwork)
        work4 = np.imag(cwork)

        refl_matrix = np.zeros((4, 4), dtype=np.complex_)
        refl_matrix[0, 0] = work1 * work1
        refl_matrix[1, 1] = work2 * work2
        refl_matrix[2, 2] = -work3
        refl_matrix[3, 2] = -work4
        refl_matrix[2, 3] = work4
        refl_matrix[3, 3] = -work3

        return self.modify_refl_roughness(refl_matrix)


    def modify_refl_roughness(self, gamma):
        """
        Modify reflectivity matrix to account for rough ground.
        Modification first implemented by Dr. Leland Pierce, 1-14-1993

        gamma_new = gamma_old * exp(-(2 * k * cos(theta)) ** 2)

        where k = 2 * pi * R / lambda
            where:  R = RMS height of soil, in meters
                    lambda = free-space wavelength in meters

        """
        rms = self.rms * 0.01   # convert cm --> m
        k = 2. * np.pi * rms / self.wavelength
        roughness_factor = np.exp(-np.power(2. * k * rms * np.cos(self.theta), 2))
        return gamma * roughness_factor


    def rough_ground(self, args):
        """
        Compute the backscatter matrix of a rough ground surface

        Parameters found in args (not used by all surface scattering models:
            theta:      Local incidence angle
            epsilon:    Relative dielectric constant, complex
            wavelength: Wavelength in free space, meters
            thi:        Incident elevation angle in radians (measured from z-axis to the unit vector in the direction
                            of propogation of the incident wave)
            phi:        Incident azimuth angle in radians (measured from the x-axis to the projection on the x-y plane
                            of the unit vector in the direction of propogation of the incident wave)
            ths:        Scattered elevation angle in radians (measured from z-axis to the unit vector in the direction
                            of propogation of the incident wave)
            phs:        Scattered azimuth angle in radians (measured from the x-axis to the projection on the x-y plane
                            of the unit vector in the direction of propogation of the incident wave)
            m:          RMS surface slope
            s:          standard deviation of surface heights
            ls:         large scale surface correlation length
            calc_back:  bool, if True backscatter is desired. False if not
            wftn:       Function representing the Fourier transform of the correlation function of the surface
                            (function of K1X, K1Y, K1XI, K1YI, LS

        Output:
            back:       4x4 real matrix relating the incident and scattered intensity vectors (without range dependence)
        """
        back = self.surface_models_LUT[self.surface_model]()

        return ancil.flip_stokes(back)

    def specular(self, args):
        return np.zeros((4, 4))


    def geometric_optics(self, args):
        """
        Returns:
            ww:     4x4 complex W matrix
            mm:     4x4 real modified Mueller matrix relating the incident and scattered intesity vectors (without range dependence)
        """
        theta, epsilon, wvl, thi, phi, ths, phs, m, s, ls, calc_back, wftn = args

        k1 = 2. * np.pi / wvl
        k2 = k1 * np.sqrt(epsilon)

        qx = k1 * (np.sin(thi) * np.cos(phi) - np.sin(ths) * np.cos(phs))
        qy = k1 * (np.sin(thi) * np.sin(phi) - np.sin(ths) * np.sin(phs))
        qz = k1 * (np.cos(thi) - np.cos(ths))
        qt = np.sqrt(qx * qx + qy * qy)
        q = np.sqrt(qx * qx + qy * qy + qz * qz)

        # Compute dot products and reflection coefficients
        vins, vsni, hins, hsni = self.dot_comp(thi, phi, ths, phs)
        rper, rpar = self.refl_comp(k1, k2, q, qz, calc_back)

        # Compute the terms uvv, uvh, uhv, uhh
        # These correspond to upq[0, 0], upq[0,1], upq[1, 0], upq[1, 1]
        upq = np.zeros((2, 2), dtype=np.complex_)
        d2 = np.power(hsni, 2) + np.power(vsni, 2)
        m1p = q * np.abs(qz) / (k1 * qz * d2)
        upq[0, 0] = m1p * (rper * hins * hsni + rpar * vins * vsni)
        upq[0, 1] = m1p * (rper * vins * hsni - rpar * hins * vsni)
        upq[1, 0] = m1p * (rper * hins * vsni - rpar * vins * hsni)
        upq[1, 1] = -m1p * (rper * vins * vsni + rpar * hins * hsni)

        # Compute W matrix
        ww = np.zeros((4, 4), dtype=np.complex_)
        hxs = -qx / qz
        hys = -qy / qz
        u0 = q * q * k1 * k1 * self.surface_slope_prob(hxs, hys, m) / (4. * np.power(qz, 4))
        ww[0, 0] = u0 * upq[0, 0] * np.conj(upq[0, 0])
        ww[0, 1] = u0 * upq[0, 1] * np.conj(upq[0, 1])
        ww[0, 2] = u0 * upq[0, 1] * np.conj(upq[0, 0])
        ww[0, 3] = u0 * upq[0, 0] * np.conj(upq[0, 1])
        ww[1, 0] = u0 * upq[1, 0] * np.conj(upq[1, 0])
        ww[1, 1] = u0 * upq[1, 1] * np.conj(upq[1, 1])
        ww[1, 2] = u0 * upq[1, 1] * np.conj(upq[1, 0])
        ww[1, 3] = u0 * upq[1, 0] * np.conj(upq[1, 1])
        ww[2, 0] = u0 * upq[1, 0] * np.conj(upq[0, 0])
        ww[2, 1] = u0 * upq[1, 1] * np.conj(upq[0, 1])
        ww[2, 2] = u0 * upq[1, 1] * np.conj(upq[0, 0])
        ww[2, 3] = u0 * upq[1, 0] * np.conj(upq[0, 1])
        ww[3, 0] = u0 * upq[0, 0] * np.conj(upq[1, 0])
        ww[3, 1] = u0 * upq[0, 1] * np.conj(upq[1, 1])
        ww[3, 2] = u0 * upq[0, 1] * np.conj(upq[1, 0])
        ww[3, 3] = u0 * upq[0, 0] * np.conj(upq[1, 1])

        # Compute modified Muller matrix relating the incident and scatter intensities for the rough surface
        if self._compute_mm:
            return self.mueller(ww)
        else:
            return ww

    def dot_comp(self, thi, phi, ths, phs):
        """
        Compute dot products between polarization unit vectors and the unit vectors in the incident and scattered
        directions

        Parameters:
            thi:    Incident elevation angle in radians (measured from z-axis to the unit vector in the direction of
                    propagation of the incident wave
            phi:    Incident azimuth angle in radians (measured from the x-axis to the projection on the xy-plane of
                    the unit vector in the direction of propagation of the incident wave)
            ths:    Scattered elevation angle in radians (measured from z-axis to the unit vector in the direction
                    of propagation of the incident wave)
            phs:    Scattered azimuth angle in radians (measured from the x-axis to the projection on the xy-plane of
                    the unit vector in the direction of propagation of the incident wave)

        Returns:
            vins:   vi dot ns
            vsni:   vs dot ni
            hins:   hi dot ns
            hsni:   hs dot ni

        """
        vins = np.cos(thi) * np.sin(ths) * np.cos(phs - phi) - np.sin(thi) * np.cos(ths)
        vsni = np.cos(ths) * np.sin(thi) * np.cos(phs - phi) - np.sin(ths) * np.cos(thi)
        hins = np.sin(ths) * np.sin(phs - phi)
        hsni = -np.sin(thi) * np.sin(phs - phi)
        return vins, vsni, hins, hsni

    def refl_comp(self, k1, k2, q, qz, calc_back):
        """
        Computes the reflection coefficients for perpendicular and parallel polarizations.
        The parameters qx, qy, qz are the x, y, z components of the term -k1*(nshat - nihat)

        Parameters:
            k1:         Wave number in free space
            k2:         Wave number in dielectric
            q:          SQRT(qx**2 + qy**2 + qz**2)
            qz:         z component of the term -k1 * (nshat - nihat)
            calc_back:  bool; True if backscatter is desired, False if not

        Returns:
            rper:       Reflection coefficient for perpendicular polarization
            rpar:       Reflection coefficient for parallel polarization
        """
        if calc_back:
            cthil = 1.
            sthil = 0.
            sthtl = 0.
            cthtl = 1.

        else:
            cthil = -q * np.abs(qz) / (2. * k1 * qz)
            sthil = np.sqrt(1. - cthil * cthil)
            sthtl = (k1 / k2) * sthil
            cthtl = np.sqrt(1. - sthtl * sthtl)

        den1 = k1 * cthil + k2 * cthtl
        den2 = k2 * cthil + k1 * cthtl
        rper = (k1 * cthil - k2 * cthtl) / den1
        rpar = (k2 * cthil - k1 * cthtl) / den2

        return rper, rpar

    def surface_slope_prob(self, hx, hy, m):
        """
        Computes the surface slope probability density at the X and Y slopes hx & hy, respectively

        Parameters:
            hx:     slope in x-direction
            hy:     slope in y-direction
            m:      RMS surface slope
        """
        return np.exp(-(np.power(hx, 2) + np.power(hy, 2)) / 2. * np.power(m, 2)) / (2. * np.pi * np.power(m, 2))

    def mueller(self, w):
        """
        Compute the real 4x4 modified Mueller matrix from a given complex 4x4 W matrix.
        Both matrices are assumed to be in the same convention (either BSA or FSA)

        (see: Radar Polarimetry for Geoscience Applications, pg. 27)

        Parameters:
            w:  complex 4x4 W matrix
        Returns:
            m:  real 4x4 modified Mueller matrix
        """
        v = np.zeros((4, 4), dtype=np.complex_)
        v[0, 0] = 1. + 0.j
        v[1, 1] = 1. + 0.j
        v[2, 2] = 1. + 0.j
        v[2, 3] = 1. + 0.j
        v[3, 2] = 0. - 1.j
        v[3, 3] = 0. + 1.j

        vi = np.zeros((4, 4), dtype=np.complex_)
        vi[0, 0] = 1. + 0.j
        vi[1, 1] = 1. + 0.j
        vi[2, 2] = 0.5 + 0.j
        vi[2, 3] = 0. + 0.5j
        vi[3, 2] = 0.5 + 0.j
        vi[3, 3] = 0. - 0.5j

        tmp = np.matmul(v, w)
        tmp1 = np.matmul(tmp, vi)

        return np.real(tmp1)


    def physical_optics(self):
        pass

    def small_perturbation(self):
        pass

    def umich_empirical(self):
        pass


class Soil(Ground):

    def __init__(self, freq, temp, sand_frac, clay_frac, mv, snow_layer, snow_depth, theta):
        eps_r, eps_i = dielectric.eps_soil(freq, temp, sand_frac, clay_frac, mv)
        eps = eps_r + 1j * eps_i

        if snow_layer and (snow_depth > 0):
            pass # need to implement  snow dielectric calculation! (lines 66-86, ground_layer.f)
            # eps_snow_r, eps_snow_i = ????
            # eps = eps / (eps_snow_r + 1j * eps_snow_i)

        Ground.__init__(self, epsilon=eps, theta=theta)

    def backscatter(self):
        s = self.rms / 100.  # convert cm to m
        ls = self.corr_length / 100.  # convert cm to m
        m = np.sqrt(2.) * s / ls
        thi = np.pi - self.theta
        ths = self.theta
        phs = np.pi
        phi = 0.

        args = (self.theta, self.epsilon, self.wavelength, thi, phi, ths, phs, m, s, ls, self.calc_back, None)
        return self.rough_ground(args)



class Water(Ground):

    """ Standing water """

    def __init__(self, snow_layer, snow_depth):
        if snow_layer and (snow_depth > 0):
            raise Exception("")


    def backscatter(self):
        return np.zeros((4, 4))


class Ice(Ground):

    def __init__(self):
        raise Exception("Ice ground surface not implemented yet!!")
