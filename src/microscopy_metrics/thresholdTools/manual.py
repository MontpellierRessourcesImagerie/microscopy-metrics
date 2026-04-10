import numpy as np

from microscopy_metrics.thresholdTools.threshold_tool import Threshold


class ThresholdManual(Threshold):
    """Class for computing a manual thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute a manual threshold value based on the provided image and a user-defined relative threshold parameter.
    The manual method calculates the threshold value as a fraction of the maximum pixel intensity in the image, allowing users to specify a relative threshold level for segmentation.
    """

    name = "manual"

    def __init__(self, rel_threshold=0.5):
        super(ThresholdManual, self).__init__()
        self._relThreshold = rel_threshold

    def getThreshold(self, image):
        """Computes the manual threshold value based on the provided image and the user-defined relative threshold parameter.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed manual threshold value, calculated as a fraction of the maximum pixel intensity in the image.
        """
        return self._relThreshold * np.max(image)
