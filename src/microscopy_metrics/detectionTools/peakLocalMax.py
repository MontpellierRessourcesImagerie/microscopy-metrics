from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import numpy as np
from microscopy_metrics.thresholdTools.threshold_tool import Threshold
from microscopy_metrics.detectionTools.detection_tool import DetectionTool

class PeakLocalMaxDetector(DetectionTool):
    name = "peak local maxima"
    
    def __init__(self):
        super(PeakLocalMaxDetector, self).__init__()
        self._minDistance = 1
    
    def detect(self):
        self.setNormalizedImage()
        self._normalizedImage = ndi.gaussian_filter(self._normalizedImage, sigma=2.0)
        self.gaussianHighPass()
        self._centroids = peak_local_max(
            self._highPassedImage,
            min_distance=self._minDistance,
            threshold_abs=self._thresholdTool.getThreshold(self._highPassedImage),
        )