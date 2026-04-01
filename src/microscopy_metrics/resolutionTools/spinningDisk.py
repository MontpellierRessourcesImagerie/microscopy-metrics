import math
from microscopy_metrics.resolutionTools.theoretical_resolution import TheoreticalResolution

class SpinningDiskResolution(TheoreticalResolution):
    name="spinning disk"
    def __init__(self):
        super(SpinningDiskResolution, self).__init__()

    def getTheoreticalResolution(self):
        resXY = (0.51 * self._emissionWavelength) / self._numericalAperture
        resZ = self._emissionWavelength / (
            self._refractiveIndex
            - math.sqrt(self._refractiveIndex ** 2 - self._numericalAperture ** 2)
        )
        return [resZ, resXY, resXY]