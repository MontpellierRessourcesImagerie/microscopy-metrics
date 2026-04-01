from skimage.feature import blob_log
import numpy as np
from microscopy_metrics.thresholdTools.threshold_tool import Threshold
from microscopy_metrics.detectionTools.detection_tool import DetectionTool

class BlobLogDetector(DetectionTool):
    name = "Laplacian of Gaussian"
    
    def __init__(self):
        super(BlobLogDetector, self).__init__()
    
    def detect(self):
        self.setNormalizedImage()
        blobs = blob_log(
            self._normalizedImage,
            max_sigma=self._sigma,
            threshold=self._thresholdTool.getThreshold(self._normalizedImage),
        )
        self._centroids = np.array([[blob[0], blob[1], blob[2]] for blob in blobs])