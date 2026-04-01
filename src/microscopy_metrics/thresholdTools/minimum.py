from skimage.filters import threshold_minimum
from microscopy_metrics.thresholdTools.threshold_tool import Threshold

class ThresholdMinimum(Threshold):
    name = "minimum"

    def __init__(self):
        super(ThresholdMinimum, self).__init__()

    def getThreshold(self, image):
        return threshold_minimum(image)