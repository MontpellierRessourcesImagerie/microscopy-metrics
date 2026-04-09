import os
from abc import abstractmethod

import numpy as np


class FittingTool(object):
    """Base class for fitting tools used in microscopy metrics.
    This class provides common functionalities and structure for different types of fitting methods (e.g., 1D, 2D, 3D Gaussian fitting).
    It includes methods for setting the image, centroid, spacing, region of interest (ROI), output directory, and results.
    It also defines abstract methods that must be implemented by subclasses for specific fitting techniques.
    """

    _fittingClasses = {}

    def __init__(self):
        self._image: np.ndarray = None
        self._centroid: list = []
        self._spacing: list = [1, 1, 1]
        self._roi: list = []
        self._outputDir: str = ""
        self._results: list = []
        self._show: bool = True
        self._amp: float = 1.0
        self._coords: list = []

        self.thetas = [0.0, 0.0, 0.0]
        self.fwhms = [0.0, 0.0, 0.0]
        self.uncertainties = [[0.0] * 4 for _ in range(3)]
        self.determinations = [0.0, 0.0, 0.0]
        self.parameters = [0.0] * 8
        self.pcovs = [[], [], []]
        self.params1D = [0.0] * 8
        self.axes = ["Z", "Y", "X"]

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._fittingClasses:
            raise ValueError("Class was already registered")
        cls._fittingClasses[name] = cls

    @classmethod
    def getInstance(cls, methodName: str):
        """Factory method to create an instance of a fitting class based on the provided method name.
        Args:
            methodName (str): Name of the fitting method (e.g., "1D", "2D", "3D").
        Returns:
            FittingTool: An instance of the fitting class corresponding to the method name.
        """
        fitClass = cls._fittingClasses[methodName]
        return fitClass()

    def fwhm(self, sigma: float) -> float:
        """Calculates the full width at half maximum (FWHM) for a Gaussian function based on the provided sigma value.
        Args:
            sigma (float): The standard deviation of the Gaussian function.

        Returns:
            float: The calculated FWHM value.
        """
        return 2 * np.sqrt(2 * np.log(2)) * sigma

    @abstractmethod
    def gauss(self, amp: float, bg: float, mu: list, sigma: list):
        pass

    @abstractmethod
    def evalFun(
        self, x: np.ndarray, amp: float, bg: float, mu: list, sigma: list
    ) -> float:
        pass

    @abstractmethod
    def fitCurve(
        self,
        amp: float,
        bg: float,
        mu: list,
        sigma: list,
        coords: np.ndarray,
        psf: np.ndarray,
    ) -> tuple:
        pass

    @abstractmethod
    def processSingleFit(self, index: int):
        pass

    def setNormalizedImage(self) -> np.ndarray:
        """Normalizes the input image to a range of [0, 1] and ensures that all values are non-negative.

        Raises:
            ValueError: If the input image is not 2D or 3D.

        Returns:
            np.ndarray: The normalized image with values in the range [0, 1].
        """
        if self._image.ndim not in (2, 3):
            raise ValueError("Image has to be in 2D or 3D.")

        imageFloat = self._image.astype(np.float64)
        img_min = np.min(imageFloat)
        img_max = np.max(imageFloat)

        imageFloat = (imageFloat - img_min) / (img_max - img_min + 1e-6)
        imageFloat[imageFloat < 0.0] = 0.0
        return imageFloat

    def getActivePath(self, index: int):
        """Provides the path to the folder corresponding to the selected bead, creating it if it does not exist.
        Args:
            index (int): The index of the bead for which to get the active path.

        Returns:
            Path: The path to the folder corresponding to the selected bead.
        """
        activePath = os.path.join(self._outputDir, f"bead_{index}")
        if not os.path.exists(activePath):
            os.makedirs(activePath)
        return activePath

    def uncertainty(self, pcov: np.ndarray) -> np.ndarray:
        """Calculates the uncertainties of the fitted parameters based on the provided covariance matrix.

        Args:
            pcov (np.ndarray): The covariance matrix between parameters obtained from the fitting process.

        Returns:
            np.ndarray: The uncertainties of the fitted parameters.
        """
        perr = np.sqrt(np.diag(pcov))
        return perr

    @abstractmethod
    def determination(self, params: list, coords: np.ndarray, psf: np.ndarray) -> float:
        pass

    def getLocalCentroid(self):
        """Calculates the local centroid of the PSF within the region of interest (ROI) based on the provided image and ROI.
        Returns:
            List(int): The coordinates of the local centroid within the ROI.
        """
        return [
            int(self._centroid[0]),
            int(self._centroid[1] - self._roi[0][1]),
            int(self._centroid[2] - self._roi[0][2]),
        ]

    @staticmethod
    def mip3d(image: np.ndarray, axis: int = 0) -> np.ndarray:
        """Calculates the maximum intensity projection (MIP) of a 3D image along a specified axis.
        Args:
            image (np.ndarray): The input 3D image.
            axis (int): The axis along which to compute the MIP (0 for z, 1 for y, 2 for x).
        Returns:
            np.ndarray: The maximum intensity projection of the input image along the specified axis.
        Raises:
            ValueError: If the input image is not 3D or if the specified axis is not valid.
        """
        if image.ndim != 3:
            raise ValueError("Image has to be in 3 dimensions")
        if axis not in {0, 1, 2}:
            raise ValueError("Axis must be 0 (z), 1 (y) or 2 (x).")

        return np.max(image, axis=axis)

    def compute1DParams(self):
        """Computes the initial parameters for the 1D Gaussian fit based on the PSF data and the center coordinates.
        Returns:
            List(float): Initial parameters from the 1D Gaussian fit
        """
        fitTool1D = FittingTool.getInstance("1D")
        fitTool1D._show = self._show
        fitTool1D._image = self._image
        fitTool1D._roi = self._roi
        fitTool1D._spacing = self._spacing
        fitTool1D._outputDir = self._outputDir
        fitTool1D._centroid = self._centroid
        fitTool1D.processSingleFit(0)
        self.params1D = fitTool1D.parameters
