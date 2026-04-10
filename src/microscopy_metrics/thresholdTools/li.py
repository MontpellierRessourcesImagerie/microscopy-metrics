from skimage.filters import threshold_li

from microscopy_metrics.thresholdTools.threshold_tool import Threshold


class ThresholdLi(Threshold):
    """Class for computing the Li thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute the Li threshold value based on the provided image.
    The Li method iteratively computes the threshold value by minimizing the cross-entropy between the foreground and background pixel intensity distributions in the image.
    """

    name = "li"

    def __init__(self):
        super(ThresholdLi, self).__init__()

    def getThreshold(self, image):
        """Computes the Li threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed Li threshold value.
        """
        return threshold_li(image)
