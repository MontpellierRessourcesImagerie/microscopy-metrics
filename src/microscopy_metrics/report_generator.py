from abc import abstractmethod


class ReportGenerator(object):
    """Abstract base class for generating reports based on microscopy image analysis results."""

    _generatorClasses = {}

    def __init__(self):
        self._inputDir = None
        self._imageAnalyze = None
        self._detectionDatas = {}
        self._fittingDatas = {}
        self._microscopeDatas = {}
        self._roiDatas = {}
        self._thresholdDatas = {}

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._generatorClasses:
            raise ValueError("Class was already registered")
        cls._generatorClasses[name] = cls

    @classmethod
    def getInstance(cls, methodName: str):
        """Factory method to create an instance of a report class based on the provided method name.
        Args:
            methodName (str): Name of the report method (e.g., "PDF", "CSV", "HTML").
        Returns:
            ReportGenerator: An instance of the report class corresponding to the method name.
        """
        if not cls._generatorClasses:
            import microscopy_metrics.reportTools

        generatorClass = cls._generatorClasses[methodName]
        return generatorClass()

    @abstractmethod
    def generateReport(self, outputPath=None):
        pass
