from abc import abstractmethod


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
        return max(image) / 2

    @property
    @abstractmethod
    def name(self):
        pass
