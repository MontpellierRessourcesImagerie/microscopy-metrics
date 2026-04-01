from skimage.filters import threshold_isodata
from microscopy_metrics.thresholdTools.threshold_tool import Threshold

class ThresholdIsodata(Threshold):
    name = "isodata"

    def __init__(self):
        super(ThresholdIsodata, self).__init__()

    def getThreshold(self, image):
        return threshold_isodata(image)