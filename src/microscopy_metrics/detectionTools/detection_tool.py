import numpy as np

from abc import abstractmethod
from scipy import ndimage as ndi
from skimage.feature import blob_log
from skimage.feature import blob_dog
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label


class DetectionTool(object):
    """Abstract base class for bead detection tools in microscopy images.
    It includes methods for normalizing images, applying Gaussian high-pass filtering, and an abstract method for detecting features in the image.
    The class also maintains a registry of detection classes for easy instantiation based on method names.
    Attributes:
        _detectionClasses (dict): A dictionary to store registered detection classes.
        _image (np.ndarray): The input image for detection.
        _sigma (float): The standard deviation for Gaussian filtering.
        _thresholdTool (ThresholdTool): An instance of a thresholding tool used for image processing.
        _normalizedImage (np.ndarray): The normalized version of the input image.
        _highPassedImage (np.ndarray): The high-pass filtered version of the normalized image.
        _centroids (list): A list to store detected centroids in the image.
    """

    _detectionClasses = {}

    def __init__(self):
        self._image : np.ndarray = None
        self._sigma : float = 2.0
        self._thresholdTool = None
        self._normalizedImage : np.ndarray = None
        self._highPassedImage : np.ndarray = None
        self._centroids : list = []

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._detectionClasses:
            raise ValueError("Class was already registered")
        cls._detectionClasses[name] = cls

    @classmethod
    def getInstance(cls, methodName: str):
        """Factory method to create an instance of a detection class based on the provided method name.
        Args:
            methodName (str): Name of the detection method (e.g., 'peak local maxima', 'Laplacian of Gaussian', 'Difference of Gaussian', 'Centroids').
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
        """Normalize the input image for processing by the detection algorithms.
        Converts the original image to a float64 type and normalizes pixel values to the range [0, 1].
        Negative pixel values are set to zero.
        Raises:
            ValueError: If the image is not 2D or 3D
        Note:
            Call this method before detection to ensure the image is normalized.
        """
        if self._image.ndim not in (2, 3):
            raise ValueError("Image must be in 2D or 3D.")
        self._normalizedImage = self._image.astype(np.float64)
        self._normalizedImage = (
            self._normalizedImage - np.min(self._normalizedImage)
        ) / (np.max(self._normalizedImage) - np.min(self._normalizedImage) + 1e-6)
        self._normalizedImage[self._normalizedImage < 0] = 0

    def gaussianHighPass(self):
        """Applies a Gaussian high-pass filter to the normalized image.
        Raises:
            ValueError: If the normalized image is not set before calling this method.
        Note:
            This method should be called after setNormalizedImage()"""
        if self._normalizedImage is None:
            raise ValueError(
                "Normalized image is not set. Call setNormalizedImage() first."
            )
        lowPass = ndi.gaussian_filter(self._normalizedImage, self._sigma)
        self._highPassedImage = self._normalizedImage - lowPass

    @abstractmethod
    def detect(self):
        """Detect features in the image.
        Subclasses must implement this method to specify the detection algorithm they use.
        """
        pass


class PeakLocalMaxDetector(DetectionTool):
    """Inherits from the DetectionTool base class and implements the detect method to identify local maxima in the input image.
    Attributes:
        name (str): The name of the detection method, set to 'peak local maxima'.
        _minDistance (int): The minimum distance between detected peaks, used to filter out closely spaced maxima.
    """

    name = "peak local maxima"

    def __init__(self):
        super().__init__()
        self._minDistance : int = 1

    def detect(self):
        """Detects local maxima in the input image.
        Uses a difference of Gaussian to enhance the image and identifies local maxima based on the specified minimum distance and threshold.
        Detected maxima are stored in 'self._centroids'.
        """
        self.setNormalizedImage()
        ndi.gaussian_filter(
            self._normalizedImage, sigma=2.0, output=self._normalizedImage
        )
        self.gaussianHighPass()
        self._centroids = peak_local_max(
            self._highPassedImage,
            min_distance=self._minDistance,
            threshold_abs=self._thresholdTool.getThreshold(self._highPassedImage),
        )


class CentroidDetector(DetectionTool):
    """Inherits from the DetectionTool base class and implements the detect method to identify blobs in the input image using a centroid algorithm.
    Attributes:
        name (str): The name of the detection method, set to 'Centroids'.
    """

    name = "Centroids"

    def __init__(self):
        super().__init__()

    def detect(self):
        """Detects centroids in the input image.
        Applies a difference of Gaussian to enhance the image, thresholds the high-passed image to create a binary image, and then labels the connected components in the binary image.
        Centroids are stored in 'self._centroids'.
        """
        self.setNormalizedImage()
        self._normalizedImage = ndi.gaussian_filter(self._normalizedImage, sigma=2.0)
        self.gaussianHighPass()
        self.binaryImage = self._highPassedImage > self._thresholdTool.getThreshold(
            self._highPassedImage
        )
        self.labeledImage = label(self.binaryImage)
        regionProps = regionprops(self.labeledImage)
        tmpCentroids = []
        for prop in regionProps:
            tmpCentroids.append(prop.centroid)
        self._centroids = np.array(tmpCentroids)


class BlobLogDetector(DetectionTool):
    """Inherits from the DetectionTool base class and implements the detect method to identify blobs in the input image using the LoG algorithm.
    Attributes:
        name (str): The name of the detection method, set to 'Laplacian of Gaussian'.
    """

    name = "Laplacian of Gaussian"

    def __init__(self):
        super().__init__()

    def detect(self):
        """Detects blobs in the input image using the Laplacian of Gaussian (LoG) method.
        Applies the LoG algorithm to the normalized image and identifies blob centroids based on the specified parameters (e.g., max_sigma, threshold).
        Detected centroids are stored in 'self._centroids'.
        """
        self.setNormalizedImage()
        blobs = blob_log(
            self._normalizedImage,
            max_sigma=self._sigma,
            threshold=self._thresholdTool.getThreshold(self._normalizedImage),
        )
        self._centroids = np.array([[blob[0], blob[1], blob[2]] for blob in blobs])


class BlobDogDetector(DetectionTool):
    """Inherits from the DetectionTool base class and implements the detect method to identify blobs in the input image using the DoG algorithm.
    Attributes:
        name (str): The name of the detection method, set to 'Difference of Gaussian'.
    """

    name = "Difference of Gaussian"

    def __init__(self):
        super().__init__()

    def detect(self):
        """Detects blobs in the input image using the Difference of Gaussian (DoG) method.
        Applies the DoG algorithm to the normalized image and identifies blob centroids based on the specified parameters (e.g., max_sigma, threshold).
        Detected centroids are stored in 'self._centroids' for further processing.
        """
        self.setNormalizedImage()
        blobs = blob_dog(
            self._normalizedImage,
            max_sigma=self._sigma,
            threshold=self._thresholdTool.getThreshold(self._normalizedImage),
        )
        self._centroids = np.array([[blob[0], blob[1], blob[2]] for blob in blobs])
