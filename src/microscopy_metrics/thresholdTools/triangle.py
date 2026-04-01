from skimage.filters import threshold_triangle
from microscopy_metrics.thresholdTools.threshold_tool import Threshold

class ThresholdTriangle(Threshold):
    name = "triangle"

    def __init__(self):
        super(ThresholdTriangle, self).__init__()

    def getThreshold(self, image):
        return threshold_triangle(image)