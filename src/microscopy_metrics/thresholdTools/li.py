from skimage.filters import threshold_li
from microscopy_metrics.thresholdTools.threshold_tool import Threshold

class ThresholdLi(Threshold):
    name = "li"

    def __init__(self):
        super(ThresholdLi, self).__init__()

    def getThreshold(self, image):
        return threshold_li(image)