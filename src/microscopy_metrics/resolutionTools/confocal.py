import math
from microscopy_metrics.resolutionTools.theoretical_resolution import TheoreticalResolution

class ConfocalResolution(TheoreticalResolution):
    name="confocal"
    def __init__(self):
        super(ConfocalResolution, self).__init__()

    def getTheoreticalResolution(self):
        resXY = (0.51 * self._emissionWavelength) / self._numericalAperture
        resZ = (0.88 * self._emissionWavelength) / (
            self._refractiveIndex
            - math.sqrt(self._refractiveIndex ** 2 - self._numericalAperture ** 2)
        )
        return [resZ, resXY, resXY]