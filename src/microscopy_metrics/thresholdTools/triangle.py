from skimage.filters import threshold_triangle

from microscopy_metrics.thresholdTools.threshold_tool import Threshold


class ThresholdTriangle(Threshold):
    """Class for computing the triangle thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute the triangle threshold value based on the provided image.
    The triangle method computes the threshold value by finding the point on the histogram of pixel intensities in the image that forms a triangle with the maximum and minimum pixel intensity values, which is often used for images with skewed histograms.
    """

    name = "triangle"

    def __init__(self):
        super(ThresholdTriangle, self).__init__()

    def getThreshold(self, image):
        """Computes the triangle threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed triangle threshold value.
        """
        return threshold_triangle(image)
