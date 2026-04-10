from skimage.filters import threshold_minimum

from microscopy_metrics.thresholdTools.threshold_tool import Threshold


class ThresholdMinimum(Threshold):
    """Class for computing the minimum thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute the minimum threshold value based on the provided image.
    The minimum method computes the threshold value by finding the minimum between the two peaks in the histogram of pixel intensities in the image, which is often used for images with bimodal histograms.
    """

    name = "minimum"

    def __init__(self):
        super(ThresholdMinimum, self).__init__()

    def getThreshold(self, image):
        """Computes the minimum threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed minimum threshold value.
        """
        return threshold_minimum(image)
