from microscopy_metrics.resolutionTools.theoretical_resolution import TheoreticalResolution

class WidefieldResolution(TheoreticalResolution):
    name = "widefield"
    def __init__(self):
        super(WidefieldResolution, self).__init__()

    def getTheoreticalResolution(self):
        resXY = (0.51 * self._emissionWavelength) / self._numericalAperture
        resZ = (1.77 * self._refractiveIndex * self._emissionWavelength) / (
                self._numericalAperture ** 2
        )
        return [resZ, resXY, resXY]