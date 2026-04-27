import numpy as np

from microscopy_metrics.thresholdTools.threshold_tool import Threshold


class ThresholdLegacy(Threshold):
    """Class for computing the legacy thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute the legacy threshold value based on the provided image.
    The legacy method iteratively computes the threshold value by separating the pixel intensities into background and signal classes and calculating their means until convergence.
    """

    name = "legacy"

    def __init__(self, nb_iteration=100):
        super(ThresholdLegacy, self).__init__()
        self.nbIteration = nb_iteration

    def getThreshold(self, image):
        """Computes the legacy threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed legacy threshold value.
        """
        imgMin = float(np.min(image))
        imgMax = float(np.max(image))
        midpoint = (imgMax + imgMin) / 2
        image[image < 0] = 0
        for _ in range(self.nbIteration):
            background = image[image <= midpoint]
            signal = image[image > midpoint]
            meanBackground = np.mean(background) if len(background) > 0 else imgMin
            meanSignal = np.mean(signal) if len(signal) > 0 else imgMax
            nMidpoint = (meanBackground + meanSignal) / 2
            if abs(nMidpoint - midpoint) < 1e-6:
                break
            midpoint = nMidpoint
        return midpoint
