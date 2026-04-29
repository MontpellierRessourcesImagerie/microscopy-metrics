from concurrent.futures import ThreadPoolExecutor, as_completed


class Metrics(object):
    """Class for calculating various metrics related to microscopy images."""

    def __init__(self):
        self._imageAnalyzer = None
        self._ringInnerDistance = None
        self._ringThickness = None

        self._TheoreticalResolutionTool = None

    @property
    def ringInnerDistance(self):
        return self._ringInnerDistance

    @ringInnerDistance.setter
    def ringInnerDistance(self, value):
        if not isinstance(value, float):
            raise ValueError("Please, enter a float value as ringInnerDistance")
        self._ringInnerDistance = value

    @property
    def ringThickness(self):
        return self._ringThickness

    @ringThickness.setter
    def ringThickness(self, value):
        if not isinstance(value, float):
            raise ValueError("Please, enter a float value as ringInnerDistance")
        self._ringThickness = value

    @property
    def theoreticalResolutionTool(self):
        return self._TheoreticalResolutionTool

    @theoreticalResolutionTool.setter
    def theoreticalResolutionTool(self, value):
        self._TheoreticalResolutionTool = value

    def signalToBackgroundRatioRing(self):
        """Calculates the signal-to-background ratio (SBR) for a set of microscopy images using a ring-based method.
        It uses a ThreadPoolExecutor to parallelize the processing of multiple images for improved performance.
        Raises:
            ValueError: If there are no images in the input list or if any of the images have an incorrect format.
        """
        self._imageAnalyzer._meanSBR = 0.0

        if len(self._imageAnalyzer._beadAnalyzer) == 0:
            raise ValueError("You must have at least one PSF")
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    bead.runSBRMetric,
                    self._imageAnalyzer._pixelSize,
                    self._ringInnerDistance,
                    self._ringThickness,
                ): i
                for i, bead in enumerate(self._imageAnalyzer._beadAnalyzer)
                if bead._rejected == False and bead._roi is not None
            }
            for future in as_completed(futures):
                _ = future.result()
        total = 0
        for bead in self._imageAnalyzer._beadAnalyzer:
            if bead._rejected == False and bead._roi is not None:
                self._imageAnalyzer._meanSBR += bead._metricTool._SBR
                total += 1
        self._imageAnalyzer._meanSBR = self._imageAnalyzer._meanSBR / total

    def runPrefittingMetrics(self):
        """Runs the pre-fitting metrics calculations, including signal-to-background ratio (SBR) calculation and theoretical resolution estimation."""
        self.SBR = []
        yield {"desc": "SBR calculation..."}
        self.signalToBackgroundRatioRing()
        yield {"desc": "Estimating theoretical resolution..."}
        self._imageAnalyzer._theoreticalResolution = (
            self._TheoreticalResolutionTool.getTheoreticalResolution()
        )
        yield {"desc": "Mesh-based metrics calculation..."}
        self.runMeshMetrics()

    def runMeshMetrics(self):
        """Runs the mesh-based metrics calculations, including mesh building, concavity, curvature, and sphericity calculations."""
        if len(self._imageAnalyzer._beadAnalyzer) == 0:
            raise ValueError("You must have at least one PSF")
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    bead._metricTool.meshMetrics,
                ): i
                for i, bead in enumerate(self._imageAnalyzer._beadAnalyzer)
                if bead._rejected == False and bead._roi is not None
            }
            for future in as_completed(futures):
                _ = future.result()