import numpy as np

from abc import abstractmethod
from scipy import ndimage as ndi
from skimage.feature import blob_log
from skimage.feature import blob_dog
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label


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
        This method checks if the input image is 2D or 3D, converts it to a float64 type, and normalizes its pixel values to the range [0, 1].
        It also ensures that any negative pixel values are set to zero.
        Raises:
            ValueError: This function only operate on 2D or 3D images
        """
        if self._image.ndim not in (2, 3):
            raise ValueError("Image have to be in 2D or 3D.")
        self._normalizedImage = self._image.astype(np.float64)
        self._normalizedImage = (
            self._normalizedImage - np.min(self._normalizedImage)
        ) / (np.max(self._normalizedImage) - np.min(self._normalizedImage) + 1e-6)
        self._normalizedImage[self._normalizedImage < 0] = 0

    def gaussianHighPass(self):
        """Applies a Gaussian high-pass filter to the normalized image.
        This method applies a Gaussian filter to the normalized image to create a low-pass filtered version, and then subtracts this low-pass image from the original normalized image to obtain the high-pass filtered image.
        The resulting high-pass image emphasizes features in the original image that are smaller than the specified sigma value.
        """
        if self._normalizedImage is None:
            raise ValueError("Normalized image is not set. Call setNormalizedImage() first.")
        lowPass = ndi.gaussian_filter(self._normalizedImage, self._sigma)
        self._highPassedImage = self._normalizedImage - lowPass

    @abstractmethod
    def detect(self):
        pass


class PeakLocalMaxDetector(DetectionTool):
    """Class for detecting local maxima using the peak_local_max function from scikit-image.
    This class inherits from the DetectionTool base class and implements the detect method to identify local maxima in the input image.
    """

    name = "peak local maxima"

    def __init__(self):
        super().__init__()
        self._minDistance = 1

    def detect(self):
        """Detects local maxima in the input image using the peak_local_max function from scikit-image.
        The method applies a Gaussian filter to smooth the image, performs a high-pass filtering, and then uses the peak_local_max function to find local maxima based on the specified parameters (e.g., min_distance, threshold_abs).
        The detected local maxima are stored in the _centroids attribute for further processing.
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
    """Class for detecting blobs using a centroid method.
    This class inherits from the DetectionTool base class and implements the detect method to identify blobs in the input image using a centroid algorithm.
    """

    name = "Centroids"

    def __init__(self):
        super().__init__()

    def detect(self):
        """Detects centroids in the input image using a simple thresholding and region properties approach.
        The method applies a Gaussian filter to smooth the image, performs a high-pass filtering, and then uses a threshold to create a binary image.
        The connected components in the binary image are labeled, and the centroids of these components are calculated and stored in the _centroids attribute for further processing.
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
    """Class for detecting blobs using the Laplacian of Gaussian (LoG) method.
    This class inherits from the DetectionTool base class and implements the detect method to identify blobs in the input image using the LoG algorithm.
    """

    name = "Laplacian of Gaussian"

    def __init__(self):
        super().__init__()

    def detect(self):
        """Detects blobs in the input image using the Laplacian of Gaussian (LoG) method.
        The method applies the LoG algorithm to the normalized image and identifies blob centroids based on the specified parameters (e.g., max_sigma, threshold).
        The detected centroids are stored in the _centroids attribute for further processing.
        """
        self.setNormalizedImage()
        blobs = blob_log(
            self._normalizedImage,
            max_sigma=self._sigma,
            threshold=self._thresholdTool.getThreshold(self._normalizedImage),
        )
        self._centroids = np.array([[blob[0], blob[1], blob[2]] for blob in blobs])


class BlobDogDetector(DetectionTool):
    """Class for detecting blobs using the Difference of Gaussian (DoG) method.
    This class inherits from the DetectionTool base class and implements the detect method to identify blobs in the input image using the DoG algorithm.
    """

    name = "Difference of Gaussian"

    def __init__(self):
        super().__init__()

    def detect(self):
        """Detects blobs in the input image using the Difference of Gaussian (DoG) method.
        The method applies the DoG algorithm to the normalized image and identifies blob centroids based on the specified parameters (e.g., max_sigma, threshold).
        The detected centroids are stored in the _centroids attribute for further processing.
        """
        self.setNormalizedImage()
        blobs = blob_dog(
            self._normalizedImage,
            max_sigma=self._sigma,
            threshold=self._thresholdTool.getThreshold(self._normalizedImage),
        )
        self._centroids = np.array([[blob[0], blob[1], blob[2]] for blob in blobs])
