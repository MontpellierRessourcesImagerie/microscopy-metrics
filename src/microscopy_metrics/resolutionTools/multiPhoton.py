import math

from microscopy_metrics.resolutionTools.theoretical_resolution import (
    TheoreticalResolution,
)


class MultiphotonResolution(TheoreticalResolution):
    """Class for calculating the theoretical resolution of a multiphoton microscope based on its parameters.
    This class inherits from the TheoreticalResolution base class and implements the method to calculate the theoretical resolution in the XY and Z dimensions using the appropriate formulas for multiphoton microscopy.
    """

    name = "multiphoton"

    def __init__(self):
        super(MultiphotonResolution, self).__init__()

    def getTheoreticalResolution(self):
        """Calculates the theoretical resolution of a multiphoton microscope in the XY and Z dimensions based on the emission wavelength, numerical aperture, and refractive index.
        Returns:
            list: A list containing the theoretical resolution in the Z dimension followed by the resolution in the XY dimensions (resZ, resXY, resXY).
        """
        if self._numericalAperture < 0.7:
            resXY = (0.377 * self._emissionWavelength) / self._numericalAperture
        else:
            resXY = (0.383 * self._emissionWavelength) / (self._numericalAperture**0.91)
        resZ = (0.626 * self._emissionWavelength) / (
            self._refractiveIndex
            - math.sqrt(self._refractiveIndex**2 - self._numericalAperture**2)
        )
        return [resZ, resXY, resXY]
    
    def getSamplingDistance(self,k=2):
        """Calculates the recommended sampling distance for a multiphoton microscope based on the theoretical resolution in the XY and Z dimensions.
        Returns:
            list: A list containing the recommended sampling distance in the Z dimension followed by the distance in the XY dimensions (distZ, distXY, distXY).
        """
        res = self.angularAperture()
        distXY = self._emissionWavelength / (4*k*self._numericalAperture)
        distZ = self._emissionWavelength / (2*k*self._refractiveIndex * (1 - math.cos(res)))
        return [distZ, distXY, distXY]
