from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from microscopy_metrics.detectionTools.detection_tool import DetectionTool


class PeakLocalMaxDetector(DetectionTool):
    """Class for detecting local maxima using the peak_local_max function from scikit-image.
    This class inherits from the DetectionTool base class and implements the detect method to identify local maxima in the input image.
    """

    name = "peak local maxima"

    def __init__(self):
        super().__init__()
        self._minDistance = 1

    def detect(self):
        """Detects local maxima in the input image using the peak_local_max function from scikit-image.
        The method applies a Gaussian filter to smooth the image, performs a high-pass filtering, and then uses the peak_local_max function to find local maxima based on the specified parameters (e.g., min_distance, threshold_abs).
        The detected local maxima are stored in the _centroids attribute for further processing.
        """
        self.setNormalizedImage()
        ndi.gaussian_filter(
            self._normalizedImage, sigma=2.0, output=self._normalizedImage
        )
        self.gaussianHighPass()
        self._centroids = peak_local_max(
            self._highPassedImage,
            min_distance=self._minDistance,
            threshold_abs=self._thresholdTool.getThreshold(self._highPassedImage),
        )
