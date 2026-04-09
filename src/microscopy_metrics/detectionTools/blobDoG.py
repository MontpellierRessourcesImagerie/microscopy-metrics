from skimage.feature import blob_dog
import numpy as np
from microscopy_metrics.detectionTools.detection_tool import DetectionTool


class BlobDogDetector(DetectionTool):
    """Class for detecting blobs in microscopy images using the Difference of Gaussian (DoG) method.
    This class inherits from the DetectionTool base class and implements the detect method to identify blobs in the input image using the DoG algorithm.
    The detected blobs are stored as centroids for further processing
    """

    name = "Difference of Gaussian"

    def __init__(self):
        super().__init__()

    def detect(self):
        """Detects blobs in the input image using the Difference of Gaussian (DoG) method.
        The method applies the DoG algorithm to the normalized image and identifies blob centroids based on the specified parameters (e.g., max_sigma, threshold). The detected centroids are stored in the _centroids attribute for further processing.
        """
        self.setNormalizedImage()
        blobs = blob_dog(
            self._normalizedImage,
            max_sigma=self._sigma,
            threshold=self._thresholdTool.getThreshold(self._normalizedImage),
        )
        self._centroids = np.array([[blob[0], blob[1], blob[2]] for blob in blobs])
