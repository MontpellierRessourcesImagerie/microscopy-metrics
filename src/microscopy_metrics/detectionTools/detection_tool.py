import numpy as np

from abc import abstractmethod
from scipy import ndimage as ndi


class DetectionTool(object):
    """Abstract base class for bead detection tools in microscopy images.
    This class provides a common interface and shared functionality for various detection algorithms.
    It includes methods for normalizing images, applying Gaussian high-pass filtering, and an abstract method for detecting features in the image.
    Subclasses must implement the detect method to specify the detection algorithm they use.
    The class also maintains a registry of detection classes for easy instantiation based on method names.
    """

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
    def getInstance(cls, methodName: str):
        """Factory method to create an instance of a detection class based on the provided method name.
        Args:
            methodName (str): Name of the detection method (e.g., "Laplacian of Gaussian", "Difference of Gaussian", "Centroids").
        Returns:
            DetectionTool: An instance of the detection class corresponding to the method name.
        """
        detectionClass = cls._detectionClasses[methodName]
        return detectionClass()

    @property
    @abstractmethod
    def name(self):
        pass

    def setNormalizedImage(self):
        """Normalizes the input image for processing by the detection algorithms.
        This method checks if the input image is 2D or 3D, converts it to a float32 type, and normalizes its pixel values to the range [0, 1].
        It also ensures that any negative pixel values are set to zero.
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
        """Applies a Gaussian high-pass filter to the normalized image.
        This method applies a Gaussian filter to the normalized image to create a low-pass filtered version, and then subtracts this low-pass image from the original normalized image to obtain the high-pass filtered image.
        The resulting high-pass image emphasizes features in the original image that are smaller than the specified sigma value.
        """
        lowPass = ndi.gaussian_filter(self._normalizedImage, self._sigma)
        self._highPassedImage = self._normalizedImage - lowPass

    @abstractmethod
    def detect(self):
        pass
