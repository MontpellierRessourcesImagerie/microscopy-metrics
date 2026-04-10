from skimage.filters import threshold_otsu

from microscopy_metrics.thresholdTools.threshold_tool import Threshold


class ThresholdOtsu(Threshold):
    """Class for computing the Otsu thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute the Otsu threshold value based on the provided image.
    The Otsu method computes the threshold value by maximizing the between-class variance of pixel intensities in the image, which is often used for images with bimodal histograms.
    """

    name = "otsu"

    def __init__(self):
        super(ThresholdOtsu, self).__init__()

    def getThreshold(self, image):
        """Computes the Otsu threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed Otsu threshold value.
        """
        return threshold_otsu(image)
