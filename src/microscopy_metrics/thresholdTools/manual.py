import numpy as np
from microscopy_metrics.thresholdTools.threshold_tool import Threshold

class ThresholdManual(Threshold):
    name = "manual"

    def __init__(self, rel_threshold=0.5):
        super(ThresholdManual, self).__init__()
        self._relThreshold = rel_threshold

    def getThreshold(self, image):
        return self._relThreshold * np.max(image)