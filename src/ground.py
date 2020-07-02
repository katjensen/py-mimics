import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
import util
import dielectric


'''
class Specular():
    pass

class RoughSurfaceModel():
    _compute_mm = True

    def __init__(self, soil):
        self.__dict__.update(soil.__dict__)
        self.rms = soil.rms / 100.                  # convert cm to m
        self.corr_length = soil.corr_length / 100.  # convert cm to m

        self.thi = np.pi - soil.theta
        self.ths = soil.theta
        self.phs = np.pi
        self.phi = 0.


    def w2mm(self, w):
        """
        Compute the real 4x4 modified Mueller matrix from a given complex 4x4 W matrix.
        Both matrices are assumed to be in the same convention (either BSA or FSA)

        (see: Radar Polarimetry for Geoscience Applications, pg. 27)

        m = V W V-1

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
        v_inv = np.linalg.inv(v)
        return np.dot(np.dot(v, w), v_inv)


    def surface_slope_prob(self, hx, hy, m):
        """
        Computes the surface slope probability density at the X and Y slopes hx & hy, respectively
        Based on Gaussian distribution

        Parameters:
            hx:     slope in x-direction
            hy:     slope in y-direction
            m:      RMS surface slope
        """
        return np.exp(-(np.power(hx, 2) + np.power(hy, 2)) / 2. * np.power(m, 2)) / (2. * np.pi * np.power(m, 2))



class GeometricOptics(RoughSurfaceModel):

    def __init__(self, soil):
        RoughSurfaceModel.__init__(soil)

    def backscatter(self):
        """
        Computes transformation matrix relating the incident and scattered intensity vectors
        for a rough surface under the Kirchoff and stationary phase approximations (ie. geometric optics model)
        The incident medium is free space.

        [see: Ulaby etal. Vol. 2, p. 929 & p. 999]

        Currently set up to use a surface height with a Gaussian distribution and zero mean. This can be modified
        in the 'surface_slope_prob' function.

        Inputs:
            epsr:       Complex dielectric constant of the ground surface relative to the medium above the surface
            wvl:        Wavelength in free space
            thi:        Incident elevation angle in radians (measured from z-axis to the unit vector in the
                            direction of propagation of the incident wave)
            phi:        Incident azimuth angle in radians (measured from x-axis to the projection on the xy-place
                            of the unit vector in the direction of propagation of the incident wave)
            ths:        Scattered elevation angle in radians (measured from z-axis to the unit vector in the
                            direction of propagation of the incident wave)
            phs:        Scattered azimuth angle in radians  (measured from x-axis to the projection on the xy-place
                            of the unit vector in the direction of propagation of the incident wave)
            s:          Standard deviation of surface heights
            l:          Large scale surface correlation length

        Returns:
            ww:     4x4 complex W matrix

            OR

            mm:     4x4 real modified Mueller matrix relating the incident and scattered intesity vectors
                    (without range dependence)

        ** CHECK Ulaby et al. Vol 2 Fig. 12.2
        """
        m = np.sqrt(2.) * self.rms / self.corr_length   # RMS surface slope
        k1 = 2. * np.pi / self.wavelength
        k2 = k1 * np.sqrt(self.epsilon)

        qx = k1 * (np.sin(self.thi) * np.cos(self.phi) - np.sin(self.ths) * np.cos(self.phs))
        qy = k1 * (np.sin(self.thi) * np.sin(self.phi) - np.sin(self.ths) * np.sin(self.phs))
        qz = k1 * (np.cos(self.thi) - np.cos(self.ths))
        qt = np.sqrt(qx * qx + qy * qy)
        q = np.sqrt(qx * qx + qy * qy + qz * qz)

        # Compute dot products and reflection coefficients
        vins, vsni, hins, hsni = self.dot_comp(self.thi, self.phi, self.ths, self.phs)
        rper, rpar = self.reflect_comp(k1, k2, q, qz)

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

        # Compute modified Mueller matrix relating the incident and scatter intensities for the rough surface
        if self._compute_mm:
            return self.w2mm(ww)
        else:
            return ww
'''


class Ground():
    _compute_mm = True      # Modified mueller matrix

    def __init__(self, simulation, log):
        self.log = log
        self.surface_model = simulation['surface_model']    # surface scattering model (1-5)
        self.theta = simulation['theta']                    # local incidence angle, radians
        self.freq = simulation['frequency']                 # sensor frequency, GHz
        self.wavelength = util.freq2wavelength(self.freq)   # sensor wavelength, meters
        self.epsilon = self.get_eps(simulation)             # dielectric of soil relative to free space, complex

        # Look-up hashtable for surface scattering models
        self.surface_models_LUT = {0:   self.specular,
                                   1:   self.geometric_optics,
                                   2:   self.physical_optics,
                                   3:   self.small_perturbation,
                                   4:   self.umich_empirical,
                                   5:   self.i2em}

        print ".... working on Ground Layer ..."
        self.reflect_matrix = self.specular_reflection()
        self.back_matrix = self.backscatter()

        print self.reflect_matrix
        print self.back_matrix

        self.ground_spec_phase_diff = util.phase_diff(self.reflect_matrix)
        self.ground_back_phase_diff = util.phase_diff(self.back_matrix)
        print " Ground layer computations COMPLETE!"


    def backscatter(self):
        util.CodeError('Attempt to calculate backscatter of general Ground class -- MUST only use subclass', self.log)


    def specular_reflection(self):
        """
        Computes the reflectivity matrix of a specular ground surface

        see: McDonald dissertation (eq. 3.20)
        """
        stheta = np.sin(self.theta)
        ctheta = np.cos(self.theta)
        cwork = np.sqrt(self.epsilon - stheta * stheta)

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

        # print refl_matrix

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
        roughness_factor = np.exp(-np.power(2. * k * np.cos(self.theta), 2))
        return gamma * roughness_factor


    def backscatter_rough_surface(self, rms, corr_length):
        """
        Compute the backscatter matrix of a rough ground surface

        Surface models:
            (0) Specular
            (1) Geometric optics:
                Conditions:     [(2 * pi * RMS_HT) / lamdba] >= 3
            (2) Physical Optics:
                Conditions:     [(2 * pi * RMS_HT) / lamdba] < 3
            (3) Small Perturbations:
                Conditions:     [(2 * pi * RMS_HT) / lamdba] < 0.3
                                [(2 * pi * CORR_LENGTH) / lamdba] < 3
                                (RMS_HT / CORR_LENGTH) < 0.3
            (4) UMich Empirical Model
            (5) I2EM Model

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
            s:          standard deviation of surface heights
            ls:         large scale surface correlation length

        Output:
            back:       4x4 real matrix relating the incident and scattered intensity vectors (without range dependence)
        """
        s = rms / 100.              # convert cm to m
        l = corr_length / 100.      # convert cm to m
        thi = np.pi - self.theta
        ths = self.theta
        phs = np.pi
        phi = 0.

        back = self.surface_models_LUT[self.surface_model](theta=self.theta, epsr=self.epsilon, wvl=self.wavelength,
                                                           thi=thi, phi=phi, ths=ths, phs=phs, s=s, l=l)
        return util.flip_stokes(back)


    def specular(self, **kwargs):
        return np.zeros((4, 4))


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

        # Note: casting trig fxn outputs as float32, to match results from F77 Mimics code

        """
        cthi = np.cos(thi, dtype=np.float32)
        sthi = np.sin(thi, dtype=np.float32)
        cths = np.cos(ths, dtype=np.float32)
        sths = np.sin(ths, dtype=np.float32)
        cphsi = np.cos(phs - phi, dtype=np.float32)
        sphsi = np.sin(phs - phi, dtype=np.float32)

        vins = cthi * sths * cphsi - sthi * cths
        vsni = cths * sthi * cphsi - sths * cthi
        hins = sths * sphsi
        hsni = -sthi * sphsi

        return vins, vsni, hins, hsni


    def reflect_comp(self, k1, k2, q, qz):
        """
        Computes the reflection coefficients for perpendicular and parallel polarizations.
        The parameters qx, qy, qz are the x, y, z components of the term -k1*(nshat - nihat)

        Parameters:
            k1:         Wave number in free space
            k2:         Wave number in dielectric
            q:          SQRT(qx**2 + qy**2 + qz**2)
            qz:         z component of the term -k1 * (nshat - nihat)

        Returns:
            rper:       Reflection coefficient for perpendicular polarization
            rpar:       Reflection coefficient for parallel polarization
        """
        # Backscatter
        cthil = 1.
        sthil = 0.
        sthtl = 0.
        cthtl = 1.

        """
        # specular direction
        else:
            cthil = -q * np.abs(qz) / (2. * k1 * qz)
            sthil = np.sqrt(1. - cthil * cthil)
            sthtl = (k1 / k2) * sthil
            cthtl = np.sqrt(1. - sthtl * sthtl)
        """

        den1 = k1 * cthil + k2 * cthtl
        den2 = k2 * cthil + k1 * cthtl
        rper = (k1 * cthil - k2 * cthtl) / den1
        rpar = (k2 * cthil - k1 * cthtl) / den2

        return np.complex64(rper), np.complex64(rpar)


    def gaussian(self, x, mean, stdev):
        return stats.norm.pdf(x, mean, stdev)


    def surface_slope_prob(self, hx, hy, m):
        """
        Computes the surface slope probability density at the X and Y slopes hx & hy, respectively

        Parameters:
            hx:     slope in x-direction
            hy:     slope in y-direction
            m:      RMS surface slope
        """
        prob = np.exp(-(hx * hx + hy * hy) / (2. * m * m), dtype=np.float32)
        return prob / (2. * np.pi * m * m)


    def w2mm(self, w):
        """
        Compute the real 4x4 modified Mueller matrix from a given complex 4x4 W matrix.
        Both matrices are assumed to be in the same convention (either BSA or FSA)

        (see: Radar Polarimetry for Geoscience Applications, pg. 27)

        Parameters:
            w:  complex 4x4 W matrix
        Returns:
            m:  real 4x4 modified Mueller matrix
        """
        v = np.zeros((4, 4), dtype=np.complex64)
        v[0, 0] = 1. + 0.j
        v[1, 1] = 1. + 0.j
        v[2, 2] = 1. + 0.j
        v[2, 3] = 1. + 0.j
        v[3, 2] = 0. - 1.j
        v[3, 3] = 0. + 1.j

        vi = np.zeros((4, 4), dtype=np.complex64)
        vi[0, 0] = 1. + 0.j
        vi[1, 1] = 1. + 0.j
        vi[2, 2] = 0.5 + 0.j
        vi[2, 3] = 0. + 0.5j
        vi[3, 2] = 0.5 + 0.j
        vi[3, 3] = 0. - 0.5j

        tmp = np.matmul(v, w)
        tmp1 = np.matmul(tmp, vi)

        print 'mm= ', np.real(tmp1)
        return np.real(tmp1)


    def geometric_optics(self, theta, epsr, wvl, thi, phi, ths, phs, s, l):
        """
        Computes transformation matrix relating the incident and scattered intensity vectors
        for a rough surface under the Kirchoff and stationary phase approximations (ie. geometric optics model)
        The incident medium is free space.

        [see: Ulaby etal. Vol. 2, p. 929 & p. 999]

        Currently set up to use a surface height with a Gaussian distribution and zero mean. This can be modified
        in the 'surface_slope_prob' function.

        Inputs:
            epsr:       Complex dielectric constant of the ground surface relative to the medium above the surface
            wvl:        Wavelength in free space
            thi:        Incident elevation angle in radians (measured from z-axis to the unit vector in the
                            direction of propagation of the incident wave)
            phi:        Incident azimuth angle in radians (measured from x-axis to the projection on the xy-place
                            of the unit vector in the direction of propagation of the incident wave)
            ths:        Scattered elevation angle in radians (measured from z-axis to the unit vector in the
                            direction of propagation of the incident wave)
            phs:        Scattered azimuth angle in radians  (measured from x-axis to the projection on the xy-place
                            of the unit vector in the direction of propagation of the incident wave)
            s:          Standard deviation of surface heights
            l:          Large scale surface correlation length

        Returns:
            ww:     4x4 complex W matrix

            OR

            mm:     4x4 real modified Mueller matrix relating the incident and scattered intesity vectors
                    (without range dependence)

        ** CHECK Ulaby et al. Vol 2 Fig. 12.2
        """
        m = np.float32(np.sqrt(2.) * s / l)    # RMS surface slope
        k1 = np.float32(2. * np.pi / wvl)
        k2 = k1 * np.sqrt(epsr, dtype=np.complex64)

        qx = k1 * (np.sin(thi, dtype=np.float32) * np.cos(phi, dtype=np.float32) - np.sin(ths, dtype=np.float32) * np.cos(phs, dtype=np.float32))
        qy = k1 * (np.sin(thi, dtype=np.float32) * np.sin(phi, dtype=np.float32) - np.sin(ths, dtype=np.float32) * np.sin(phs, dtype=np.float32))
        qz = k1 * (np.cos(thi, dtype=np.float32) - np.cos(ths, dtype=np.float32))
        qt = np.sqrt(qx * qx + qy * qy, dtype=np.float32)
        q = np.sqrt(qx * qx + qy * qy + qz * qz, dtype=np.float32)

        # Compute dot products and reflection coefficients
        vins, vsni, hins, hsni = self.dot_comp(thi, phi, ths, phs)
        rper, rpar = self.reflect_comp(k1, k2, q, qz)

        print '****'
        print vins.dtype, vsni.dtype
        print hins.dtype, hsni.dtype
        print rper.dtype, rpar.dtype

        # Compute the terms uvv, uvh, uhv, uhh
        # These correspond to upq[0, 0], upq[0,1], upq[1, 0], upq[1, 1]
        upq = np.zeros((2, 2), dtype=np.complex64)
        d2 = np.float32(np.power(hsni, 2) + np.power(vsni, 2))
        m1p = q * np.abs(qz) / (k1 * qz * d2)
        print "d2= ", d2
        print "m1p= ", m1p
        upq[0, 0] = -m1p * (rper * hins * hsni + rpar * vins * vsni)
        upq[0, 1] = m1p * (rper * vins * hsni - rpar * hins * vsni)
        upq[1, 0] = m1p * (rper * hins * vsni - rpar * vins * hsni)
        upq[1, 1] = -m1p * (rper * vins * vsni + rpar * hins * hsni)
        print "UPQ: ", upq, upq.dtype

        # Compute W matrix
        ww = np.zeros((4, 4), dtype=np.complex64)
        hxs = -qx / qz
        hys = -qy / qz
        u0 = q * q * k1 * k1 * self.surface_slope_prob(hxs, hys, m) / (4. * np.power(qz, 4, dtype=np.float32))
        print 'PROB: ', self.surface_slope_prob(hxs, hys, m)
        print 'hxs: ', hxs
        print 'hys: ', hys
        print 'u0: ', u0
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

        print "WW: ", ww

        # Compute modified Mueller matrix relating the incident and scatter intensities for the rough surface
        if self._compute_mm:
            return self.w2mm(ww)
        else:
            return ww


    def physical_optics(self, theta, epsr, wvl, thi, phi, ths, phs, s, l):
        """
        Computes the transformation matrix relating the incident and scattered intensity vectors for a
        rough surface under the Kirchoff and scalar approximations (physical optics model)

        The incident medium is free space.

        Currently only implements Gaussian correlation function

        Currently set up to use a surface height with a Gaussian distribution and zero mean. This can be modified
        in the 'surface_slope_prob' function.

        Inputs:
            epsr:       Complex dielectric constant of the ground surface relative to the medium above the surface
            wvl:        Wavelength in free space
            thi:        Incident elevation angle in radians (measured from z-axis to the unit vector in the
                            direction of propagation of the incident wave)
            phi:        Incident azimuth angle in radians (measured from x-axis to the projection on the xy-place
                            of the unit vector in the direction of propagation of the incident wave)
            ths:        Scattered elevation angle in radians (measured from z-axis to the unit vector in the
                            direction of propagation of the incident wave)
            phs:        Scattered azimuth angle in radians  (measured from x-axis to the projection on the xy-place
                            of the unit vector in the direction of propagation of the incident wave)
            s:          Standard deviation of surface heights
            l:          Large scale surface correlation length
            ls:         Small scale surface correlation length

        Returns:
            ww:     4x4 complex W matrix

            OR

            mm:     4x4 real modified Mueller matrix relating the incident and scattered intesity vectors
                    (without range dependence)

        ** CHECK Ulaby et al. Vol 2 Fig. 12.4, 12.3a-c. 12.4 is the result of setting RPER1=RPAR1=0
        """
        k1 = np.float32(2. * np.pi / wvl)
        k2 = k1 * np.sqrt(epsr)
        ls = 1.

        qx = k1 * (np.sin(thi, dtype=np.float32) * np.cos(phi, dtype=np.float32) - np.sin(ths, dtype=np.float32) * np.cos(phs, dtype=np.float32))
        qy = k1 * (np.sin(thi, dtype=np.float32) * np.sin(phi, dtype=np.float32) - np.sin(ths, dtype=np.float32) * np.sin(phs, dtype=np.float32))
        qz = k1 * (np.cos(thi, dtype=np.float32) - np.cos(ths, dtype=np.float32))
        qt = np.sqrt(qx * qx + qy * qy)

        # Compute APQ and BPQ coefficients
        apq, bpq, cpq = self.po_coefs(k1, k2, thi, phi, ths, phs, epsr)

        # specular scattering (?)
        # ..... add this when implementing Bistatic (see: lines 1080-1090 in ground_layer.f)
        IC = np.zeros((2, 2, 2, 2), dtype=np.complex64)

        # Compute terms IN (incoherent) and IS (due to surface slope). Involves numerical integration
        IN, IS = self.estimate_gauss_IN_IS(apq, bpq, cpq, qx, qy, qz, qt, s, l)

        # Compute the W matrix for both incoherent and surface slope contributions
        c0 = k1 ** 2 / (4. * np.pi) ** 2
        ww = np.zeros((4, 4), dtype=np.complex64)

        ww[0, 0] = c0 * (IN[0, 0, 0, 0] + IS[0, 0, 0, 0] + IC[0, 0, 0, 0])
        ww[0, 1] = c0 * (IN[0, 1, 0, 1] + IS[0, 1, 0, 1] + IC[0, 1, 0, 1])
        ww[0, 2] = c0 * (IN[0, 1, 0, 0] + IS[0, 1, 0, 0] + IC[0, 1, 0, 0])
        ww[0, 3] = c0 * (IN[0, 0, 0, 1] + IS[0, 0, 0, 1] + IC[0, 0, 0, 1])
        ww[1, 0] = c0 * (IN[1, 0, 1, 0] + IS[1, 0, 1, 0] + IC[1, 0, 1, 0])
        ww[1, 1] = c0 * (IN[1, 1, 1, 1] + IS[1, 1, 1, 1] + IC[1, 1, 1, 1])
        ww[1, 2] = c0 * (IN[1, 1, 1, 0] + IS[1, 1, 1, 0] + IC[1, 1, 1, 0])
        ww[1, 3] = c0 * (IN[1, 0, 1, 1] + IS[1, 0, 1, 1] + IC[1, 0, 1, 1])
        ww[2, 0] = c0 * (IN[1, 0, 0, 0] + IS[1, 0, 0, 0] + IC[1, 0, 0, 0])
        ww[2, 1] = c0 * (IN[1, 1, 0, 1] + IS[1, 1, 0, 1] + IC[1, 1, 0, 1])
        ww[2, 2] = c0 * (IN[1, 1, 0, 0] + IS[1, 1, 0, 0] + IC[1, 1, 0, 0])
        ww[2, 3] = c0 * (IN[1, 0, 0, 1] + IS[1, 0, 0, 1] + IC[1, 0, 0, 1])
        ww[3, 0] = c0 * (IN[0, 0, 1, 0] + IS[0, 0, 1, 0] + IC[0, 0, 1, 0])
        ww[3, 1] = c0 * (IN[0, 1, 1, 1] + IS[0, 1, 1, 1] + IC[0, 1, 1, 1])
        ww[3, 2] = c0 * (IN[0, 1, 1, 0] + IS[0, 1, 1, 0] + IC[0, 1, 1, 0])
        ww[3, 3] = c0 * (IN[0, 0, 1, 1] + IS[0, 0, 1, 1] + IC[0, 0, 1, 1])

        # Compute modified Mueller matrix relating the incident and scatter intensities for the rough surface
        if self._compute_mm:
            return self.w2mm(ww)
        else:
            return ww



    def estimate_gauss_IN_IS(self, apq, bpq, cpq, qx, qy, qz, qt, s, l):
        """
        Subroutine that numerically computes the IN and IS terms for a Gaussian distribution of
        surface height and Gaussian correlation function

        Estimate uses truncated summation instead of numerical integration
        """
        IN = np.zeros((2, 2, 2, 2), dtype=np.complex64)
        IS = np.zeros((2, 2, 2, 2), dtype=np.complex64)

        nmax = 100
        prec = 1.0e-6

        # Summation for IN and IS, using Gaussian correlation function
        sum_value = 0.
        term = 1.

        for n in xrange(1, nmax + 1):
            chk = np.abs(sum_value)
            xn = np.float(n)
            term = (term / xn) * (qz * s) ** 2
            xint = np.exp(-(qt * l) ** 2 / (4. * xn)) / xn
            add = term * xint
            sum_value += add

            if np.abs(add) < (prec * chk):
                break
            else:
                if n == nmax:
                    util.CodeError('Truncated Gaussian summation in Geometric Optics model is not making good '
                                   'progress ... please check parameters', self.log)
        l2 = l * l
        tn = np.pi * l2 * np.exp(-(qz * s) ** 2)
        ts = -tn / qz
        sumn = tn * sum_value
        sumsx = ts * qx * sum_value
        sumsy = ts * qy * sum_value

        # Multiply by polarization dependent coefficients and set up matrix
        # i1 = P, i2 = Q, i3 = M, i4 = N
        for i1 in xrange(2):
            for i2 in xrange(2):
                for i3 in xrange(2):
                    for i4 in xrange(2):
                        IN[i1, i2, i3, i4] = sumn * apq[i1, i2] * np.conj(apq[i3, i4])
                        IS[i1, i2, i3, i4] = sumsx * (bpq[i1, i2] * np.conj(apq[i3, i4]) + apq[i1, i2] *
                                                      np.conj(bpq[i3, i4])) + \
                                             sumsy * (cpq[i1, i2] * np.conj(apq[i3, i4]) + apq[i1, i2] *
                                                      np.conj(cpq[i3, i4]))
        return IN, IS


    def po_coefs(self, k1, k2, thi, phi, ths, phs, epsr):
        apq = np.zeros((2, 2), dtype=np.complex64)
        bpq = np.zeros((2, 2), dtype=np.complex64)
        cpq = np.zeros((2, 2), dtype=np.complex64)

        c1 = (k1 * np.cos(thi, dtype=np.float32)) + (k2 * (np.sqrt(1. - (k1 * np.sin(thi, dtype=np.float32) / k2) * (k1 * np.sin(thi, dtype=np.float32) / k2))) )
        c2 = (k1 * np.cos(thi, dtype=np.float32)) - (k2 * (np.sqrt(1. - (k1 * np.sin(thi, dtype=np.float32) / k2) * (k1 * np.sin(thi, dtype=np.float32) / k2))) )
        c3 = (k1 * np.sin(thi, dtype=np.float32)) + (k2 * (k1 * np.sin(thi, dtype=np.float32) / k2))
        c4 = (k2 * np.cos(thi, dtype=np.float32)) + (k1 * (np.sqrt(1. - (k1 * np.sin(thi, dtype=np.float32) / k2) * (k1 * np.sin(thi, dtype=np.float32) / k2))) )
        c5 = (k2 * np.cos(thi, dtype=np.float32)) - (k1 * (np.sqrt(1. - (k1 * np.sin(thi, dtype=np.float32) / k2) * (k1 * np.sin(thi, dtype=np.float32) / k2))) )
        c6 = (k2 * np.sin(thi, dtype=np.float32)) - (k1 * (k1 * np.sin(thi, dtype=np.float32) / k2))
        c7 = (k2 * np.sin(thi, dtype=np.float32)) + (k1 * (k1 * np.sin(thi, dtype=np.float32) / k2))
        c8 = np.cos(thi, dtype=np.float32) - np.cos(ths, dtype=np.float32)
        c9 = 1. - np.cos(thi, dtype=np.float32) * np.cos(ths, dtype=np.float32)
        c10 = np.sin(ths, dtype=np.float32) - np.sin(thi, dtype=np.float32) * np.cos(phs - phi, dtype=np.float32)

        rper0 = c1 / c2
        rpar0 = c4 / c5

        if np.absolute(epsr) < 50.:      # if epsr is 'small' -- need to check on this
            rpar1 = np.complex(0., 0.)
            rper1 = np.complex(0., 0.)
        else:
            rper1 = rper0 * c3 / c2
            rpar1 = -(c6 - rpar0 * c7) / c5

        apq[0, 0] = rpar0 * c8 * np.cos(phs - phi, dtype=np.float32)
        apq[0, 1] = rper0 * c9 * np.sin(phs - phi, dtype=np.float32)
        apq[1, 0] = rpar0 * c9 * np.sin(phs - phi, dtype=np.float32)
        apq[1, 1] = rper0 * c8 * np.cos(phs - phi, dtype=np.float32)

        zvv = rpar0 * c10 + rpar1 * c8 * np.cos(phs - phi, dtype=np.float32)
        zvh = -np.sin(phs - phi, dtype=np.float32) * (rper0 * np.sin(thi, dtype=np.float32) * np.cos(ths, dtype=np.float32) + rper1 * c9)
        zhv = np.sin(phs - phi, dtype=np.float32) * (rpar0 * np.sin(thi, dtype=np.float32) * np.cos(ths, dtype=np.float32) + rpar1 * c9)
        zhh = rper0 * c10 + rper1 * c8 * np.cos(phs - phi, dtype=np.float32)

        bpq[0, 0] = zvv * np.cos(phi, dtype=np.float32)
        bpq[0, 1] = zvh * np.cos(phi, dtype=np.float32)
        bpq[1, 0] = zhv * np.cos(phi, dtype=np.float32)
        bpq[1, 1] = zhh * np.cos(phi, dtype=np.float32)

        cpq[0, 0] = zvv * np.sin(phi, dtype=np.float32)
        cpq[0, 1] = zvh * np.sin(phi, dtype=np.float32)
        cpq[1, 0] = zhv * np.sin(phi, dtype=np.float32)
        cpq[1, 1] = zhh * np.sin(phi, dtype=np.float32)

        return apq, bpq, cpq


    def small_perturbation(self, theta, epsr, wvl, thi, phi, ths, phs, s, l):
        """
        Computes the transformation matrix relating the incident and scattered intensity vectors for a
        rough surface under the Kirchoff and scalar approximations (physical optics model)

        The incident medium is free space.

        Currently only implements Gaussian correlation function

        Currently set up to use a surface height with a Gaussian distribution and zero mean. This can be modified
        in the 'surface_slope_prob' function.

        Inputs:
            epsr:       Complex dielectric constant of the ground surface relative to the medium above the surface
            wvl:        Wavelength in free space
            thi:        Incident elevation angle in radians (measured from z-axis to the unit vector in the
                            direction of propagation of the incident wave)
            phi:        Incident azimuth angle in radians (measured from x-axis to the projection on the xy-place
                            of the unit vector in the direction of propagation of the incident wave)
            ths:        Scattered elevation angle in radians (measured from z-axis to the unit vector in the
                            direction of propagation of the incident wave)
            phs:        Scattered azimuth angle in radians  (measured from x-axis to the projection on the xy-place
                            of the unit vector in the direction of propagation of the incident wave)
            s:          Standard deviation of surface heights
            l:          Large scale surface correlation length
            ls:         Small scale surface correlation length

        Returns:
            ww:     4x4 complex W matrix

            OR

            mm:     4x4 real modified Mueller matrix relating the incident and scattered intesity vectors
                    (without range dependence)

        ** CHECK Ulaby et al. Vol 2 Fig. 12.6a, 12.6b, 12.7a, 12.7b
        """
        k1 = np.float32(2. * np.pi / wvl)
        k2 = k1 * np.sqrt(epsr)

        print "***!! k1: ", k1, '    k2: ', k2
        print "cths= ", np.cos(ths, dtype=np.float32)

        # Compute the polarization dependent coefficients and Fourier transform of the correlation function
        fpq, wcor = self.sp_coef(k1, k2, thi, phi, ths, phs, l, s)

        print "fpq : ", fpq
        print "wcor = ", wcor

        # Compute the W matrix
        f0 = k1 * k1 * np.cos(ths, dtype=np.float32) * np.cos(ths, dtype=np.float32) * wcor
        ww = np.zeros((4, 4), dtype=np.complex64)

        ww[0, 0] = f0 * fpq[0, 0] * np.conj(fpq[0, 0])
        ww[0, 1] = f0 * fpq[0, 1] * np.conj(fpq[0, 1])
        ww[0, 2] = f0 * fpq[0, 1] * np.conj(fpq[0, 0])
        ww[0, 3] = f0 * fpq[0, 0] * np.conj(fpq[0, 1])
        ww[1, 0] = f0 * fpq[1, 0] * np.conj(fpq[1, 0])
        ww[1, 1] = f0 * fpq[1, 1] * np.conj(fpq[1, 1])
        ww[1, 2] = f0 * fpq[1, 1] * np.conj(fpq[1, 0])
        ww[1, 3] = f0 * fpq[1, 0] * np.conj(fpq[1, 1])
        ww[2, 0] = f0 * fpq[1, 0] * np.conj(fpq[0, 0])
        ww[2, 1] = f0 * fpq[1, 1] * np.conj(fpq[0, 1])
        ww[2, 2] = f0 * fpq[1, 1] * np.conj(fpq[0, 0])
        ww[2, 3] = f0 * fpq[1, 0] * np.conj(fpq[0, 1])
        ww[3, 0] = f0 * fpq[0, 0] * np.conj(fpq[1, 0])
        ww[3, 1] = f0 * fpq[0, 1] * np.conj(fpq[1, 1])
        ww[3, 2] = f0 * fpq[0, 1] * np.conj(fpq[1, 0])
        ww[3, 3] = f0 * fpq[0, 0] * np.conj(fpq[1, 1])

        # Compute modified Mueller matrix relating the incident and scatter intensities for the rough surface
        if self._compute_mm:
            return self.w2mm(ww)
        else:
            return ww


    def sp_coef(self, k1, k2, thi, phi, ths, phs, l, s):
        """ Computes the coefficients for FPQ for polarizations VV, VH, HV, HH """
        fpq = np.zeros((2, 2), dtype=np.complex64)
        k1xi = k1 * np.sin(thi, dtype=np.float32) * np.cos(phi, dtype=np.float32)
        k1yi = k1 * np.sin(thi, dtype=np.float32) * np.sin(phi, dtype=np.float32)
        k1zi = k1 * np.cos(thi, dtype=np.float32)
        k2zi = np.sqrt(k2 ** 2 - (k1 * np.sin(ths, dtype=np.float32)) ** 2)

        k1x = k1 * np.sin(ths, dtype=np.float32) * np.cos(phs, dtype=np.float32)
        k1y = k1 * np.sin(ths, dtype=np.float32) * np.sin(phs, dtype=np.float32)
        k1z = k1 * np.cos(ths, dtype=np.float32)
        k2z = np.sqrt(k2 ** 2 - (k1 * np.sin(ths, dtype=np.float32)) ** 2)

        k1p = k1 * np.sin(ths, dtype=np.float32)
        k1pi = k1 * np.sin(thi, dtype=np.float32)

        kk1 = (k1x * k1xi + k1y * k1yi) / (k1p * k1pi)
        kk2 = (k1y * k1xi - k1x * k1yi) / (k1p * k1pi)
        c1 = k2 * k2 - k1 * k1
        c2 = k1 * k1 * k2z + k2 * k2 * k1z
        c3 = k1 * k1 * k2zi - k2* k2 * k1zi
        c4 = k2zi - k1zi
        c5 = k2z + k1z
        c6 = k1p * k1pi - (k1 * k1 / k2 ** 2) * k2z * k2zi * kk1

        fpq[0, 0] = 2. * k2 * k2 * k1zi * c1 * c6 / (c2 * c3)
        fpq[0, 1] = 2. * k1 * k2z * k1zi * c1 * kk2 / (c2 * c4)
        fpq[1, 0] = -2. * k1 * k2zi * k1zi * c1 * kk2 / (c5 * c3)
        fpq[1, 1] = 2. * k1zi * c1 * kk1 / (c5 * c4)

        return fpq, self.fcorr(k1x, k1y, k1xi, k1yi, l, s)


    def fcorr(self, k1x, k1y, k1xi, k1yi, l, s):
        """
        Fourier transform of the correlation function of a rough surface (ie. normalized roughness spectrum).

        This is particularly for a Gaussian correlation function

        Params:
            k1x:        Wave number in x-direction (scattered wave)
            k1y:        Wave number in y-direction (scattered wave)
            k1xi:       Wave number in x-direction (incident wave)
            k1yi:       Wave number in y-direction (incident wave)
            l:          Correlation length of the surface
            s:          Standard deviation of surface heights

        """
        kdp2 = (k1x - k1xi) ** 2 + (k1y - k1yi) ** 2
        return (l * l * s * s / (4. * np.pi)) * np.exp(-0.25 * kdp2 * l ** 2, dtype=np.float32)


    def umich_empirical(self, theta, epsr, wvl, thi, phi, ths, phs, s, l):
        """
        Semi-empirical model developed at UMich, detailed:
        Yisok Oh, K. Sarabandi, Fawwaz Ulaby: An Empirical Model and an Inversion Technique
        for Radar Scattering from Bare Soil Surfaces, IEEE Transactions on Geoscience and Remote Sensing, vol 30.,
        No. 2, 1992. pages 370-381.

        Check: figs 13 and 14 in above paper

        Works only in backscatter.
        """
        wvl_cm = 100. * wvl     # wavelength, cm
        scm =100. * s           # RMS surface height, cm

        ks = (2. * np.pi / wvl_cm) * scm
        n = np.sqrt(epsr)
        ctmp = (n - 1.) / (n + 1.)
        gam0 = np.abs(ctmp) ** 2
        ct = np.cos(theta)
        ct2 = np.sqrt(1. - np.sin(theta) ** 2 / epsr)

        gamh = np.abs((ct - n * ct2) / (ct + n * ct2)) ** 2
        gamv = np.abs((n * ct - ct2) / (n * ct + ct2)) ** 2

        g = 0.7 * (1. - np.exp(-0.65 * (ks ** 1.8)))
        rootp = 1. - ((2. * theta / np.pi) ** (1. / (3. * gam0))) * np.exp(-ks)

        p = rootp * rootp
        q = 0.23 * np.sqrt(gam0) * (1. - np.exp(-ks))

        sig0vv = g * (ct ** 3) * (gamv + gamh) / rootp
        vvmag = np.sqrt(sig0vv / (4. * np.pi))
        hhmag = np.sqrt(p * sig0vv / (4. * np.pi))
        hvmag = np.sqrt(q * sig0vv / (4. * np.pi))

        # Compute the W matrix (eq. 4.139)
        ww = np.zeros((4, 4), dtype=np.complex128)
        ww[0, 0] = vvmag ** 2 + 0j
        ww[0, 1] = hvmag ** 2 + 0j
        ww[1, 0] = hvmag ** 2 + 0j
        ww[1, 1] = hhmag ** 2 + 0j
        ww[2, 2] = hhmag * vvmag + 0j
        ww[3, 3] = hhmag * vvmag + 0j

        # Compute modified Mueller matrix relating the incident and scatter intensities for the rough surface
        if self._compute_mm:
            return self.w2mm(ww)
        else:
            return ww


    def i2em(self, theta, epsr, wvl, thi, phi, ths, phs, s, l, ls):
        pass



class Soil(Ground):

    def __init__(self, simulation, log):
        # Additional soil paramters
        self.rms = simulation['rms_soil']               # surface roughness, root mean sq height [cm]
        self.corr_length = simulation['ls_soil']        #

        # Suggested surface model:

        Ground.__init__(self, simulation, log)


    def backscatter(self):
        return self.backscatter_rough_surface(self.rms, self.corr_length)


    def get_eps(self, params):
        print params.columns
        if 'eps_soil' in params.columns:
            return params['eps_soil']
        else:
            eps_r, eps_i = dielectric.eps_soil(self.freq, temp=params['temp_soil'], sand_frac=params['sand'] / 100.,
                                               clay_frac=params['clay'] / 100., mv=params['mv_soil'],
                                               logfile=self.log)
            return eps_r - 1j * eps_i



class Water(Ground):

    """ Standing water """

    def __init__(self, simulation, log):
        Ground.__init__(self, simulation, log)


    def backscatter(self):
        return np.zeros((4, 4))


    def get_eps(self, params):
        eps_r, eps_i = dielectric.eps_water(temp=params['temp_water'], freq=self.freq, s=params['salinity'])
        return eps_r - 1j * eps_i



class Ice(Ground):

    def __init__(self):
        raise Exception("Snow / Ice ground surface not implemented yet!!")

    def backscatter(self):
        pass
        # If snow layer present ## need to implement this!!
        # lines 108-115 (ground_layer.f)
