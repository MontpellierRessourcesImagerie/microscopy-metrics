import math
from microscopy_metrics.resolutionTools.theoretical_resolution import TheoreticalResolution

class MultiphotonResolution(TheoreticalResolution):
    name="multiphoton"
    def __init__(self):
        super(MultiphotonResolution, self).__init__()

    def getTheoreticalResolution(self):
        if self._numericalAperture < 0.7:
            resXY = (0.377 * self._emissionWavelength) / self._numericalAperture
        else:
            resXY = (0.383 * self._emissionWavelength) / (
                    self._numericalAperture ** 0.91
            )
        resZ = (0.626 * self._emissionWavelength) / (
            self._refractiveIndex
            - math.sqrt(self._refractiveIndex ** 2 - self._numericalAperture ** 2)
        )
        return [resZ, resXY, resXY]
