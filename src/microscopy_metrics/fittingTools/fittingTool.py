import numpy as np
from scipy import ndimage as ndi
import os
from abc import abstractmethod

class FittingTool(object):
    _fittingClasses={}
    def __init__(self):
        self._image = None
        self._centroid = []
        self._spacing = [1,1,1]
        self._roi = []
        self._outputDir = ""
        self._results = []
        self._show = True
        self._amp = 1.0

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._fittingClasses:
            raise ValueError("Class was already registered")
        cls._fittingClasses[name] = cls

    @classmethod
    def getInstance(cls, methodName):
        fitClass = cls._fittingClasses[methodName]
        return fitClass()

    def getCovMatrix(self, image, centroid):
        """Function to get covariance matrix of a 1D,2D or 3D image

        Args:
            image (np.ndarray): Input image
            centroid (List(float)): coordinates of the centroid
        """

        def cov(x, y, i):
            return np.sum(x * y * i) / np.sum(i)

        extends = [np.arange(l) for l in image.shape]
        grids = np.meshgrid(*extends, indexing="ij")

        if image.ndim == 1:
            x = grids[0].ravel() - centroid[0]
            return np.sqrt(cov(x, x, image.ravel()))
        elif image.ndim == 2:
            y = grids[0].ravel() - centroid[0]
            x = grids[1].ravel() - centroid[1]
            cxx = cov(x, x, image.ravel())
            cyy = cov(y, y, image.ravel())
            cxy = cov(x, y, image.ravel())
            return np.array([[cxx, cxy], [cxy, cyy]])
        elif image.ndim == 3:
            z = grids[0].ravel() - centroid[0]
            y = grids[1].ravel() - centroid[1]
            x = grids[2].ravel() - centroid[2]
            cxx = cov(x, x, image.ravel())
            cyy = cov(y, y, image.ravel())
            czz = cov(z, z, image.ravel())
            cxy = cov(x, y, image.ravel())
            cxz = cov(x, z, image.ravel())
            cyz = cov(y, z, image.ravel())
            return np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]])
        else:
            NotImplementedError()

    def fwhm(self, sigma):
        """
        Args:
            sigma (float): Gaussian width parameter

        Returns:
            float: Full width half maximum
        """
        return 2 * np.sqrt(2 * np.log(2)) * sigma

    @abstractmethod
    def gauss(self, amp, bg, mu, sigma):
        pass

    @abstractmethod
    def evalFun(self, x, amp, bg, mu, sigma):
        pass

    @abstractmethod
    def fitCurve(self, amp, bg, mu, sigma, coords, psf):
        pass

    @abstractmethod
    def processSingleFit(self, index):
        pass

    def setNormalizedImage(self):
        """Method to normalize a 2D or 3D image and erase negative values

        Raises:
            ValueError: This function only operate on 2D or 3D images

        Returns:
            np.ndarray: Image normalized
        """
        if self._image.ndim not in (2, 3):
            raise ValueError("Image have to be in 2D or 3D.")
        imageFloat = self._image.astype(np.float64)
        imageFloat = (imageFloat - np.min(imageFloat)) / (
                np.max(imageFloat) - np.min(imageFloat) + 1e-6
        )
        imageFloat[imageFloat < 0] = 0
        return self._image

    def getActivePath(self, index):
        """
        Args:
            index (int): Bead ID corresping to it's position in the list

        Returns:
            Path: Folder's path found (or created) for the selected bead
        """
        activePath = os.path.join(self._outputDir, f"bead_{index}")
        if not os.path.exists(activePath):
            os.makedirs(activePath)
        return activePath

    def uncertainty(self, pcov):
        """Measure uncertainty of parameters returned by Gaussian fit.

        Parameters
        ----------
        pcov : list
            The covariance matrix between parameters
        Returns
        -------
        perr : np.array
            The list of uncertainty factor for each parameter
        """
        perr = np.sqrt(np.diag(pcov))
        return perr

    @abstractmethod
    def determination(self, params, coords, psf):
        pass

    def getLocalCentroid(self):
        return [
            int(self._centroid[0]),
            int(self._centroid[1] - self._roi[0][1]),
            int(self._centroid[2] - self._roi[0][2]),
        ]

    def mip3d(self,image, axis=0):
        if image.ndim != 3:
            raise ValueError("Image have to be in 3 dimensions")
        if axis not in {0, 1, 2}:
            raise ValueError("Axis must be 0 (z), 1 (y) or 2 (x).")

        return np.max(image, axis=axis)