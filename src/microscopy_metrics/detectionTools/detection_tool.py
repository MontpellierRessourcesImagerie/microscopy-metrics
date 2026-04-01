from scipy import ndimage as ndi
import numpy as np
from abc import ABC, abstractmethod


class DetectionTool(object):
    _detectionClasses = {}

    def __init__(self):
        self._image = None
        self._sigma = 2.0
        self._thresholdTool = None
        self._normalizedImage = None
        self._highPassedImage = None
        self._centroids = []

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._detectionClasses:
            raise ValueError("Class was already registered")
        cls._detectionClasses[name] = cls

    @classmethod
    def getInstance(cls, methodName):
        detectionClass = cls._detectionClasses[methodName]
        return detectionClass()

    def setNormalizedImage(self):
        """Method to normalize a 2D or 3D image and erase negative values

        Raises:
            ValueError: This function only operate on 2D or 3D images
        """
        if self._image.ndim not in (2, 3):
            raise ValueError("Image have to be in 2D or 3D.")
        self._normalizedImage = self._image.astype(np.float32)
        self._normalizedImage = (
                                        self._normalizedImage - np.min(self._normalizedImage)
        ) / (np.max(self._normalizedImage) - np.min(self._normalizedImage) + 1e-6)
        self._normalizedImage[self._normalizedImage < 0] = 0

    def gaussianHighPass(self):
        lowPass = ndi.gaussian_filter(self._normalizedImage, self._sigma)
        self._highPassedImage = self._normalizedImage - lowPass

    @abstractmethod
    def detect(self):
        pass


    @property
    @abstractmethod
    def name(self):
        pass