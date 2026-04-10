from skimage.filters import threshold_isodata

from microscopy_metrics.thresholdTools.threshold_tool import Threshold


class ThresholdIsodata(Threshold):
    """Class for computing the isodata thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute the isodata threshold value based on the provided image.
    The isodata method iteratively computes the threshold value by minimizing the intra-class variance of the pixel intensities in the image.
    """

    name = "isodata"

    def __init__(self):
        super(ThresholdIsodata, self).__init__()

    def getThreshold(self, image):
        """Computes the isodata threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed isodata threshold value.
        """
        return threshold_isodata(image)
