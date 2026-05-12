import math

from microscopy_metrics.resolutionTools.theoretical_resolution import (
    TheoreticalResolution,
)


class ConfocalResolution(TheoreticalResolution):
    """Class for calculating the theoretical resolution of a confocal microscope based on its parameters.
    This class inherits from the TheoreticalResolution base class and implements the method to calculate the theoretical resolution in the XY and Z dimensions using the appropriate formulas for confocal microscopy.
    """

    name = "confocal"

    def __init__(self):
        super(ConfocalResolution, self).__init__()

    def getTheoreticalResolution(self):
        """Calculates the theoretical resolution of a confocal microscope in the XY and Z dimensions based on the emission wavelength, numerical aperture, and refractive index.
        Returns:
            list: A list containing the theoretical resolution in the Z dimension followed by the resolution in the XY dimensions (resZ, resXY, resXY).
        """
        resXY = (0.51 * self._excitationWavelength) / self._numericalAperture
        resZ = (0.88 * self._excitationWavelength) / (
            self._refractiveIndex
            - math.sqrt(self._refractiveIndex**2 - self._numericalAperture**2)
        )
        return [resZ, resXY, resXY]

    def getSamplingDistance(self):
        """Calculates the recommended sampling distance for a confocal microscope based on the theoretical resolution in the XY and Z dimensions.
        Returns:
            list: A list containing the recommended sampling distance in the Z dimension followed by the distance in the XY dimensions (distZ, distXY, distXY).
        """
        res = self.angularAperture()
        distXY = self._excitationWavelength / (8*self._numericalAperture)
        distZ = self._excitationWavelength / (4*self._refractiveIndex * (1 - math.cos(res)))
        return [distZ, distXY, distXY]