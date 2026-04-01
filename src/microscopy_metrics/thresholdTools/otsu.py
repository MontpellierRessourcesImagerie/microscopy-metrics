from skimage.filters import threshold_otsu
from microscopy_metrics.thresholdTools.threshold_tool import Threshold

class ThresholdOtsu(Threshold):
    name = "otsu"

    def __init__(self):
        super(ThresholdOtsu, self).__init__()

    def getThreshold(self, image):
        return threshold_otsu(image)