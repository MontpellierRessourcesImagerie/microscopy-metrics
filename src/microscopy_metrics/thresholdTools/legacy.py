import numpy as np
from microscopy_metrics.thresholdTools.threshold_tool import Threshold

class ThresholdLegacy(Threshold):
    name = "legacy"

    def __init__(self, nb_iteration=100):
        super(ThresholdLegacy, self).__init__()
        self.nbIteration = nb_iteration

    def getThreshold(self, image):
        """Apply the Metroloj_Qc's 'legacy threshold'

        Returns:
            float: Value of the threshold
        """
        imgMin = np.min(image)
        imgMax = np.max(image)
        midpoint = (imgMax - imgMin) / 2
        image[image < 0] = 0
        for i in range(self.nbIteration):
            background = image[image <= midpoint]
            signal = image[image > midpoint]
            meanBackground = np.mean(background) if len(background) > 0 else imgMin
            meanSignal = np.mean(signal) if len(signal) > 0 else imgMax
            nMidpoint = (meanBackground + meanSignal) / 2
            if abs(nMidpoint - midpoint) < 1e-6:
                break
            midpoint = nMidpoint
        return midpoint