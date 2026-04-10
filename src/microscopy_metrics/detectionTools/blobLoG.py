import numpy as np

from skimage.feature import blob_log

from microscopy_metrics.detectionTools.detection_tool import DetectionTool


class BlobLogDetector(DetectionTool):
    """Class for detecting blobs using the Laplacian of Gaussian (LoG) method.
    This class inherits from the DetectionTool base class and implements the detect method to identify blobs in the input image using the LoG algorithm.
    """

    name = "Laplacian of Gaussian"

    def __init__(self):
        super().__init__()

    def detect(self):
        """Detects blobs in the input image using the Laplacian of Gaussian (LoG) method.
        The method applies the LoG algorithm to the normalized image and identifies blob centroids based on the specified parameters (e.g., max_sigma, threshold).
        The detected centroids are stored in the _centroids attribute for further processing.
        """
        self.setNormalizedImage()
        blobs = blob_log(
            self._normalizedImage,
            max_sigma=self._sigma,
            threshold=self._thresholdTool.getThreshold(self._normalizedImage),
        )
        self._centroids = np.array([[blob[0], blob[1], blob[2]] for blob in blobs])
