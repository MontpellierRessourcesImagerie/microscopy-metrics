# This file is a part of the PSFGenerator toolbox (https://github.com/Biomedical-Imaging-Group/PSFGenerator) by Biomedical-Imaging-Group, which implements the Born & Wolf 3D optical model for generating point spread functions (PSFs) in microscopy.
# The code has been adapted to Python and structured as a class for easier integration into the microscopy metrics framework.

import math

from scipy.special import j0
from scipy.integrate import quad


class KirchhoffDiffractionSimpson(object):
    """Class for computing the Kirchhoff diffraction integral using Simpson's rule.
    This class provides methods for calculating the diffraction pattern based on the given parameters such as defocus, refractive index (ni), numerical aperture (NA), and wavelength (lmbda).
    The class includes methods for evaluating the integrand of the diffraction integral and for performing the integration to compute the intensity of the diffraction pattern.
    """

    TOL = 1e-1
    K = 0
    NA = 1.4
    lmbda = 610
    defocus = 1
    ni = 1.5

    def __init__(self, defocus, ni, accuracy, NA, lmbda):
        self.NA = NA
        self.lmbda = lmbda
        self.defocus = defocus
        self.ni = ni
        if accuracy == 0:
            self.K = 5
        elif accuracy == 1:
            self.K = 7
        elif accuracy == 2:
            self.K = 9
        else:
            self.K = 3

    def calculate(self, r):
        """Calculates the intensity of the diffraction pattern at a given radial distance r using the Kirchhoff diffraction integral.
        The method defines the integrand for the real and imaginary parts of the diffraction integral, performs the integration using the quad function from scipy, and computes the intensity based on the real and imaginary components of the integral.
        Args:
            r (float): The radial distance at which to calculate the intensity of the diffraction pattern.
        Returns:
            float: The calculated intensity of the diffraction pattern at the given radial distance r.
        """

        def real_integrand(rho):
            return self.integrand(rho, r)[0]

        def imag_integrand(rho):
            return self.integrand(rho, r)[1]

        real_val, _ = quad(real_integrand, 0.0, 1.0, limit=500, epsabs=1e-4)
        imag_val, _ = quad(imag_integrand, 0.0, 1.0, limit=500, epsabs=1e-4)
        curI = real_val**2 + imag_val**2
        return curI

    def integrand(self, rho, r):
        """Evaluates the integrand of the Kirchhoff diffraction integral for a given radial distance rho and r.
        The method calculates the Bessel function value, the optical path difference (OPD), and the phase term W, and returns the real and imaginary components of the integrand based on these calculations.
        Args:
            rho (float): The radial distance in the pupil plane.
            r (float): The radial distance at which to evaluate the integrand.
        Returns:
            list: A list containing the real and imaginary components of the integrand for the given rho and r.
        """
        k0 = 2.0 * math.pi / self.lmbda
        BesselValue = j0(k0 * self.NA * r * rho)
        OPD = 0.0
        W = 0.0
        I = [0.0, 0.0]
        OPD = self.NA**2 * self.defocus * rho**2 / (2.0 * self.ni)
        W = k0 * OPD
        I[0] = BesselValue * math.cos(W) * rho
        I[1] = -BesselValue * math.sin(W) * rho
        return I
