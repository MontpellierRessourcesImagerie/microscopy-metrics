import math
from matplotlib import image
import numpy as np

from scipy.ndimage import median_filter
from skimage.measure import regionprops, label

from microscopy_metrics.utils import umToPx
from microscopy_metrics.thresholdTools.legacy import ThresholdLegacy


class MetricTool(object):
    """Class for calculating various metrics related to microscopy images, such as signal-to-background ratio (SBR), lateral asymmetry ratio (LAR), and sphericity ratio."""

    def __init__(self):
        self._image = None
        self._ringInnerDistance = 1.0
        self._ringThickness = 2.0
        self._pixelSize = [1, 1, 1]

        self._SBR = 0
        self._LAR = 0
        self._sphericity = 0

    def setNormalizedImage(self, image: np.ndarray) -> np.ndarray:
        """Normalizes the input image to a range of [0, 1] and ensures that all values are non-negative.
        Args:
            image (np.ndarray): The input image to be normalized, which should be a 2D or 3D array representing the microscopy image data.
        Raises:
            ValueError: If the input image is not 2D or 3D.
        Returns:
            np.ndarray: The normalized image with values in the range [0, 1].
        """
        if image.ndim not in (2, 3):
            raise ValueError("Image have to be in 2D or 3D.")
        imageFloat = image.astype(np.float64)
        imageFloat = (imageFloat - np.min(imageFloat)) / (
            np.max(imageFloat) - np.min(imageFloat) + 1e-6
        )
        imageFloat[imageFloat < 0] = 0
        return imageFloat

    def processSingleSBRRing(self):
        """Calculates the signal-to-background ratio (SBR) for a single microscopy image using a ring-based method.
        The calculated SBR values are stored in the class attributes for further analysis and evaluation.
        Raises:
            ValueError: If the input image is not 2D or 3D, if there are no background pixels detected, or if there are no signal pixels detected in the image.
        """
        if self._image.ndim not in (2, 3):
            print("Incorrect picture format")
            return -1
        if self._image.size == 0:
            print("Image is empty")
            return -1
        imageFloat = self.setNormalizedImage(self._image)
        imageFloat = median_filter(imageFloat, size=5)
        thresholdAbs = ThresholdLegacy(nb_iteration=1000).getThreshold(imageFloat)
        binaryImage = imageFloat > thresholdAbs
        labeledImage = label(binaryImage)
        regions = regionprops(labeledImage)
        if not regions:
            return -1
        largestRegion = max(regions, key=lambda r: r.area)
        minZ, minY, minX, maxZ, maxY, maxX = largestRegion.bbox
        center = largestRegion.centroid
        diameterZ = maxZ - minZ
        diameterY = maxY - minY
        diameterX = maxX - minX
        diameterBead = max(diameterZ, diameterY, diameterX)
        innerDistance = (
            umToPx(self._ringInnerDistance, self._pixelSize[2]) + diameterBead / 2
        )
        outerDistance = umToPx(self._ringThickness, self._pixelSize[2]) + innerDistance
        signal = 0.0
        nSignal = 0
        background = 0.0
        nBackground = 0
        for z in range(binaryImage.shape[0]):
            for y in range(binaryImage.shape[1]):
                for x in range(binaryImage.shape[2]):
                    distance = np.sqrt(
                        (z - center[0]) ** 2
                        + (y - center[1]) ** 2
                        + (x - center[2]) ** 2
                    )
                    if binaryImage[z, y, x] == 1:
                        nSignal += 1
                        signal += self._image[z, y, x]
                    else:
                        if innerDistance <= distance <= outerDistance:
                            nBackground += 1
                            background += self._image[z, y, x]
        if nBackground == 0:
            raise ValueError("There are no background pixel detected")
        meanBackground = background / nBackground
        if nSignal == 0:
            raise ValueError("There are no signal pixel detected")
        meanSignal = signal / nSignal
        self._SBR = float(meanSignal / meanBackground)

    def lateralAsymmetryRatio(self, FWHM: list):
        """Calculates the lateral asymmetry ratio (LAR) for a PSF based on the full width at half maximum (FWHM) values.
        Args:
            FWHM (list): A list of FWHM values for the PSF.
        Raises:
           ValueError: If the FWHM values are not available or if there are not enough FWHM values to calculate the LAR.
        """
        if FWHM == []:
            raise ValueError(
                "FWHM values are not available or insufficient to calculate LAR."
            )
        tmp = np.array([FWHM[1], FWHM[2]])
        self._LAR = tmp.min() / tmp.max()

    def sphericityRatio(self, FWHM: list):
        """Calculates the sphericity ratio for a PSF based on the full width at half maximum (FWHM) values.
        Args:
            FWHM (list): A list of FWHM values for the PSF.
        Raises:
            ValueError: If the FWHM values are not available or if there are not enough FWHM values to calculate the sphericity ratio.
        """
        if FWHM == []:
            raise ValueError(
                "FWHM values are not available or insufficient to calculate sphericity."
            )
        FWHMxy = math.sqrt(FWHM[2] * FWHM[1])
        sphericity = FWHMxy / FWHM[0]
        self._sphericity = sphericity
