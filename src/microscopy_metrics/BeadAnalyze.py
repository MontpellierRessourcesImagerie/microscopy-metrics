from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.metricTool.metricTool import MetricTool


class BeadAnalyze(object):
    def __init__(self, id = 0, image = None, roi = None, centroid = None):
        self._id = id
        self._image = image
        self._roi = roi
        self._centroid = centroid
        self._rejected = False
        self._rejectionDesc = ""
        self._metricTool = None
        self._fitTool = None


    def runFitting(self, fittingType = "1D", spacing = [1,1,1], outputDir = None, prominenceRel = None):
        self._fitTool = FittingTool.getInstance(fittingType)
        self._fitTool._image = self._image
        self._fitTool._centroid = self._centroid
        self._fitTool._roi = self._roi
        self._fitTool._spacing = spacing
        self._fitTool._outputDir = outputDir
        if hasattr(self._fitTool, "_prominenceRel") and prominenceRel is not None:
            self._fitTool._prominenceRel = prominenceRel
        self._fitTool.processSingleFit(self._id)

    def runSBRMetric(self, spacing = [1,1,1], ringInnerDistance = 1.0, ringThickness = 2.0):
        self._metricTool = MetricTool()
        self._metricTool._image = self._image
        print(self._image.shape)
        self._metricTool._pixelSize = spacing
        self._metricTool._ringInnerDistance = ringInnerDistance
        self._metricTool._ringThickness = ringThickness
        self._metricTool.processSingleSBRRing()
        
