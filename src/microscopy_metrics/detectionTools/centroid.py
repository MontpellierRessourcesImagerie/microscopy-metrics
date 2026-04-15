import numpy as np

from scipy import ndimage as ndi
from skimage.measure import regionprops, label

from microscopy_metrics.detectionTools.detection_tool import DetectionTool


class CentroidDetector(DetectionTool):
    """Class for detecting blobs using a centroid method.
    This class inherits from the DetectionTool base class and implements the detect method to identify blobs in the input image using a centroid algorithm.
    """

    name = "Centroids"

    def __init__(self):
        super().__init__()

    def detect(self):
        """Detects centroids in the input image using a simple thresholding and region properties approach.
        The method applies a Gaussian filter to smooth the image, performs a high-pass filtering, and then uses a threshold to create a binary image.
        The connected components in the binary image are labeled, and the centroids of these components are calculated and stored in the _centroids attribute for further processing.
        """
        self.setNormalizedImage()
        self._normalizedImage = ndi.gaussian_filter(self._normalizedImage, sigma=2.0)
        self.gaussianHighPass()
        binaryImage = self._highPassedImage > self._thresholdTool.getThreshold(
            self._highPassedImage
        )

        labeledImage = label(binaryImage)
        regionProps = regionprops(labeledImage)
        tmpCentroids = []
        for prop in regionProps:
            tmpCentroids.append(prop.centroid)
        self._centroids = np.array(tmpCentroids)
