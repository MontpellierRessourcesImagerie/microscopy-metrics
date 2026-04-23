from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.metricTool.metricTool import MetricTool


class BeadAnalyzer(object):
    """Class to manage the analysis of individual beads, including storing the bead's image, region of interest (ROI), centroid, and results of fitting and metric calculations.
    It provides methods for running fitting and calculating metrics for the bead, which are used in the overall analysis of microscopy images.
    """

    def __init__(self, id=0, image=None, roi=None, centroid=None):
        self._id = id
        self._image = image
        self._roi = roi
        self._centroid = centroid
        self._rejected = False
        self._rejectionDesc = ""
        self._metricTool = None
        self._fitTool = None

    def toDict(self):
        """Converts the bead analysis results into a dictionary format for easier access and manipulation of the bead's data, including its ID, image, ROI, centroid, rejection status, and any rejection descriptions.
        Returns:
            dict: A dictionary containing the bead's analysis results.
        """
        return {
            "id": self._id,
            "image": self._image,
            "roi": self._roi,
            "centroid": self._centroid,
            "rejected": self._rejected,
            "rejectionDesc": self._rejectionDesc,
        }

    def runFitting(
        self, fittingType="1D", spacing=[1, 1, 1], outputDir=None, prominenceRel=None
    ):
        """Runs the fitting process for the bead using the specified fitting type, spacing, output directory, and prominence relative value.
        Args:
            fittingType (str, optional): The type of fitting to perform. Defaults to "1D".
            spacing (list, optional): The spacing of the image pixels. Defaults to [1,1,1].
            outputDir (_type_, optional): The directory to save the fitting results. Defaults to None.
            prominenceRel (_type_, optional): The relative prominence value for the fitting. Defaults to None.
        """
        self._fitTool = FittingTool.getInstance(fittingType)
        self._fitTool._image = self._image
        self._fitTool._centroid = self._centroid
        self._fitTool._roi = self._roi
        self._fitTool._spacing = spacing
        self._fitTool._outputDir = outputDir
        if hasattr(self._fitTool, "_prominenceRel") and prominenceRel is not None:
            self._fitTool._prominenceRel = prominenceRel
        self._fitTool.processSingleFit(self._id)

    def runSBRMetric(self, spacing=[1, 1, 1], ringInnerDistance=1.0, ringThickness=2.0):
        """Runs the signal-to-background ratio (SBR) metric calculation for the bead using the specified spacing, ring inner distance, and ring thickness.
        Args:
            spacing (list, optional): The spacing of the image pixels. Defaults to [1,1,1].
            ringInnerDistance (float, optional): The inner distance of the ring for SBR calculation. Defaults to 1.0.
            ringThickness (float, optional): The thickness of the ring for SBR calculation. Defaults to 2.0.
        """
        self._metricTool = MetricTool()
        self._metricTool._image = self._image
        self._metricTool._pixelSize = spacing
        self._metricTool._ringInnerDistance = ringInnerDistance
        self._metricTool._ringThickness = ringThickness
        self._metricTool.processSingleSBRRing()
