from microscopy_metrics.resolutionTools.theoretical_resolution import TheoreticalResolution


class WidefieldResolution(TheoreticalResolution):
    """Class for calculating the theoretical resolution of a widefield microscope based on its parameters.
    This class inherits from the TheoreticalResolution base class and implements the method to calculate the theoretical resolution in the XY and Z dimensions using the appropriate formulas for widefield microscopy.
    """

    name = "widefield"

    def __init__(self):
        super(WidefieldResolution, self).__init__()

    def getTheoreticalResolution(self):
        """Calculates the theoretical resolution of a widefield microscope in the XY and Z dimensions based on the emission wavelength, numerical aperture, and refractive index.
        Returns:
            list: A list containing the theoretical resolution in the Z dimension followed by the resolution in the XY dimensions (resZ, resXY, resXY).
        """
        resXY = (0.51 * self._emissionWavelength) / self._numericalAperture
        resZ = (1.77 * self._refractiveIndex * self._emissionWavelength) / (
            self._numericalAperture**2
        )
        return [resZ, resXY, resXY]
