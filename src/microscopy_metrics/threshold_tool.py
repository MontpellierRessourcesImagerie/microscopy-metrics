from skimage.filters import (
    threshold_otsu,
    threshold_isodata,
    threshold_li,
    threshold_minimum,
    threshold_triangle,
)
import numpy as np
from abc import ABC, abstractmethod


class Threshold(object):
    name = "middle"
    _thresholdClasses = {}

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._thresholdClasses:
            raise ValueError("Class was already registered")
        cls._thresholdClasses[name] = cls

    @classmethod
    def getInstance(cls, methodName):
        thresholdClass = cls._thresholdClasses[methodName]
        return thresholdClass()

    def getThreshold(self, image):
        return max(image) / 2

    @property
    @abstractmethod
    def name(self):
        pass


class ThresholdOtsu(Threshold):
    name = "otsu"

    def __init__(self):
        super(ThresholdOtsu, self).__init__()

    def getThreshold(self, image):
        return threshold_otsu(image)


class ThresholdIsodata(Threshold):
    name = "isodata"

    def __init__(self):
        super(ThresholdIsodata, self).__init__()

    def getThreshold(self, image):
        return threshold_isodata(image)


class ThresholdLi(Threshold):
    name = "li"

    def __init__(self):
        super(ThresholdLi, self).__init__()

    def getThreshold(self, image):
        return threshold_li(image)


class ThresholdMinimum(Threshold):
    name = "minimum"

    def __init__(self):
        super(ThresholdMinimum, self).__init__()

    def getThreshold(self, image):
        return threshold_minimum(image)


class ThresholdTriangle(Threshold):
    name = "triangle"

    def __init__(self):
        super(ThresholdTriangle, self).__init__()

    def getThreshold(self, image):
        return threshold_triangle(image)


class ThresholdLegacy(Threshold):
    name = "legacy"

    def __init__(self, nb_iteration=100):
        super(ThresholdLegacy, self).__init__()
        self.nbIteration = nb_iteration

    def getThreshold(self, image):
        """Apply the Metroloj_Qc's 'legacy threshold'

        Returns:
            float: Value of the threshold
        """
        imgMin = np.min(image)
        imgMax = np.max(image)
        midpoint = (imgMax - imgMin) / 2
        image[image < 0] = 0
        for i in range(self.nbIteration):
            background = image[image <= midpoint]
            signal = image[image > midpoint]
            meanBackground = np.mean(background) if len(background) > 0 else imgMin
            meanSignal = np.mean(signal) if len(signal) > 0 else imgMax
            nMidpoint = (meanBackground + meanSignal) / 2
            if abs(nMidpoint - midpoint) < 1e-6:
                break
            midpoint = nMidpoint
        return midpoint


class ThresholdManual(Threshold):
    name = "manual"

    def __init__(self, rel_threshold=0.5):
        super(ThresholdManual, self).__init__()
        self._relThreshold = rel_threshold

    def getThreshold(self, image):
        return self._relThreshold * np.max(image)
