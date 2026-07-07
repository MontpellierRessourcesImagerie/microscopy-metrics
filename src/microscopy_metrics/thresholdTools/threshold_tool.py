import numpy as np

from abc import abstractmethod
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_minimum,
    threshold_otsu,
    threshold_triangle
)

class Threshold(object):
    """Base class for thresholding methods.
    This class defines the interface for thresholding algorithms and provides a mechanism for registering and retrieving specific thresholding implementations.
    Subclasses must implement the `getThreshold` method to compute the threshold value based on the
    """

    name = "middle"
    _thresholdClasses = {}

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._thresholdClasses:
            raise ValueError("Class was already registered")
        cls._thresholdClasses[name] = cls

    @classmethod
    def getInstance(cls, methodName):
        """Factory method to create an instance of a thresholding class based on the provided method name.
        Args:
            methodName (str): Name of the thresholding method (e.g., "otsu", "li", "minimum").
        Returns:
            Threshold: An instance of the requested thresholding class.
        """
        thresholdClass = cls._thresholdClasses[methodName]
        return thresholdClass()

    def getThreshold(self, image):
        """Abstract method to compute the threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed threshold value.
        """
        return np.max(image) / 2

    @property
    @abstractmethod
    def name(self):
        pass


class ThresholdIsodata(Threshold):
    """Class for computing the isodata thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute the isodata threshold value based on the provided image.
    The isodata method iteratively computes the threshold value by minimizing the intra-class variance of the pixel intensities in the image.
    """

    name = "isodata"

    def __init__(self):
        super(ThresholdIsodata, self).__init__()

    def getThreshold(self, image):
        """Computes the isodata threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed isodata threshold value.
        """
        return threshold_isodata(image)


class ThresholdLegacy(Threshold):
    """Class for computing the legacy thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute the legacy threshold value based on the provided image.
    The legacy method iteratively computes the threshold value by separating the pixel intensities into background and signal classes and calculating their means until convergence.
    """

    name = "legacy"

    def __init__(self, nb_iteration=100):
        super(ThresholdLegacy, self).__init__()
        self.nbIteration = nb_iteration

    def getThreshold(self, image):
        """Computes the legacy threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed legacy threshold value.
        """
        imgMin = float(np.min(image))
        imgMax = float(np.max(image))
        midpoint = (imgMax + imgMin) / 2
        image[image < 0] = 0
        for _ in range(self.nbIteration):
            background = image[image <= midpoint]
            signal = image[image > midpoint]
            meanBackground = np.mean(background) if len(background) > 0 else imgMin
            meanSignal = np.mean(signal) if len(signal) > 0 else imgMax
            nMidpoint = (meanBackground + meanSignal) / 2
            if abs(nMidpoint - midpoint) < 1e-6:
                break
            midpoint = nMidpoint
        return midpoint


class ThresholdLi(Threshold):
    """Class for computing the Li thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute the Li threshold value based on the provided image.
    The Li method iteratively computes the threshold value by minimizing the cross-entropy between the foreground and background pixel intensity distributions in the image.
    """

    name = "li"

    def __init__(self):
        super(ThresholdLi, self).__init__()

    def getThreshold(self, image):
        """Computes the Li threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed Li threshold value.
        """
        return threshold_li(image)
    

class ThresholdManual(Threshold):
    """Class for computing a manual thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute a manual threshold value based on the provided image and a user-defined relative threshold parameter.
    The manual method calculates the threshold value as a fraction of the maximum pixel intensity in the image, allowing users to specify a relative threshold level for segmentation.
    """

    name = "manual"

    def __init__(self, rel_threshold=0.5):
        super(ThresholdManual, self).__init__()
        self._relThreshold = rel_threshold

    def getThreshold(self, image):
        """Computes the manual threshold value based on the provided image and the user-defined relative threshold parameter.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed manual threshold value, calculated as a fraction of the maximum pixel intensity in the image.
        """
        return self._relThreshold * np.max(image)
    

class ThresholdMinimum(Threshold):
    """Class for computing the minimum thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute the minimum threshold value based on the provided image.
    The minimum method computes the threshold value by finding the minimum between the two peaks in the histogram of pixel intensities in the image, which is often used for images with bimodal histograms.
    """

    name = "minimum"

    def __init__(self):
        super(ThresholdMinimum, self).__init__()

    def getThreshold(self, image):
        """Computes the minimum threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed minimum threshold value.
        """
        return threshold_minimum(image)
    

class ThresholdOtsu(Threshold):
    """Class for computing the Otsu thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute the Otsu threshold value based on the provided image.
    The Otsu method computes the threshold value by maximizing the between-class variance of pixel intensities in the image, which is often used for images with bimodal histograms.
    """

    name = "otsu"

    def __init__(self):
        super(ThresholdOtsu, self).__init__()

    def getThreshold(self, image):
        """Computes the Otsu threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed Otsu threshold value.
        """
        return threshold_otsu(image)


class ThresholdTriangle(Threshold):
    """Class for computing the triangle thresholding method.
    This class inherits from the Threshold base class and implements the getThreshold method to compute the triangle threshold value based on the provided image.
    The triangle method computes the threshold value by finding the point on the histogram of pixel intensities in the image that forms a triangle with the maximum and minimum pixel intensity values, which is often used for images with skewed histograms.
    """

    name = "triangle"

    def __init__(self):
        super(ThresholdTriangle, self).__init__()

    def getThreshold(self, image):
        """Computes the triangle threshold value based on the provided image.
        Args:
            image (np.ndarray): The input image for which the threshold value is to be computed.
        Returns:
            float: The computed triangle threshold value.
        """
        return threshold_triangle(image)