from skimage.measure import regionprops, label
from scipy import ndimage as ndi
import numpy as np
from microscopy_metrics.thresholdTools.threshold_tool import Threshold
from microscopy_metrics.detectionTools.detection_tool import DetectionTool

class CentroidDetector(DetectionTool):
    name = "Centroids"

    def __init__(self):
        super(CentroidDetector, self).__init__()
    
    def detect(self):
        self.setNormalizedImage()
        self._normalizedImage = ndi.gaussian_filter(self._normalizedImage, sigma=2.0)
        self.gaussianHighPass()
        binaryImage = self._highPassedImage > self._thresholdTool.getThreshold(self._highPassedImage)
        labeledImage = label(binaryImage)
        regionProps = regionprops(labeledImage)
        tmpCentroids = []
        for prop in regionProps:
            tmpCentroids.append(prop.centroid)

        self._centroids = np.array(tmpCentroids)