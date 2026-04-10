import math
import numpy as np

from scipy.ndimage import median_filter
from skimage.measure import regionprops, label
from concurrent.futures import ThreadPoolExecutor, as_completed

from microscopy_metrics.utils import umToPx
from microscopy_metrics.thresholdTools.legacy import ThresholdLegacy


class Metrics(object):
    """Class for calculating various metrics related to microscopy images, including signal-to-background ratio (SBR), lateral asymmetry ratio (LAR), sphericity, and theoretical resolution.
    This class provides methods for processing microscopy images, calculating metrics based on the image data, and storing the results for further analysis and evaluation.
    """

    def __init__(self, image=None):
        self._image = image
        self._images = []
        self._ringInnerDistance = 1.0
        self._ringThickness = 2.0
        self._pixelSize = [1, 1, 1]
        self._FWHM = []
        self._TheoreticalResolutionTool = None

        self.LAR = 0
        self._sphericity = 0
        self.SBR = []
        self.meanSBR = 0.0
        self.theoreticalResolution = []

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
            raise ValueError("Please, select an Image with 2 or 3 dimensions.")
        self._image = image

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        if len(images) == 0 or images is None:
            raise ValueError("Please, send at list one image")
        self._images = images

    @property
    def ringInnerDistance(self):
        return self._ringInnerDistance

    @ringInnerDistance.setter
    def ringInnerDistance(self, value):
        if not isinstance(value, float):
            raise ValueError("Please, enter a float value as ringInnerDistance")
        self._ringInnerDistance = value

    @property
    def ringThickness(self):
        return self._ringThickness

    @ringThickness.setter
    def ringThickness(self, value):
        if not isinstance(value, float):
            raise ValueError("Please, enter a float value as ringInnerDistance")
        self._ringThickness = value

    @property
    def pixelSize(self):
        return self._pixelSize

    @pixelSize.setter
    def pixelSize(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("Shape format not compatible with current image")
        self._pixelSize = value

    @property
    def FWHM(self):
        return self._FWHM

    @FWHM.setter
    def FWHM(self, value):
        if not isinstance(value, list):
            raise ValueError("FWHM must be a list")
        self._FWHM = value

    @property
    def sphericity(self):
        return self._sphericity

    @sphericity.setter
    def sphericity(self, value):
        if not isinstance(value, float):
            raise ValueError("Sphericity muse be a float")
        self._sphericity = value

    @property
    def theoreticalResolutionTool(self):
        return self._TheoreticalResolutionTool

    @theoreticalResolutionTool.setter
    def theoreticalResolutionTool(self, value):
        self._TheoreticalResolutionTool = value

    def setNormalizedImage(self, image):
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

    def processSingleSBRRing(self, index, image):
        """Calculates the signal-to-background ratio (SBR) for a single microscopy image using a ring-based method.
        The method processes the input image to identify the signal and background regions based on a ring-shaped area around the detected bead.
        It calculates the mean signal and background intensities and computes the SBR for the image.
        The calculated SBR values are stored in the class attributes for further analysis and evaluation.
        Args:
            index (int): The index of the image being processed, used for storing results in the SBR attribute.
            image (np.ndarray): The input microscopy image for which to calculate the SBR, which should be a 2D or 3D array representing the image data.
        Raises:
            ValueError: If the input image is not 2D or 3D, if there are no background pixels detected, or if there are no signal pixels detected in the image.
        Returns:
            float: The calculated signal-to-background ratio (SBR) for the input image, or -1 if the image format is incorrect or if no signal/background pixels are detected.
        """
        if image.ndim not in (2, 3):
            print("Incorrect picture format")
            return -1
        imageFloat = self.setNormalizedImage(image)
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
                        signal += image[z, y, x]
                    else:
                        if innerDistance <= distance <= outerDistance:
                            nBackground += 1
                            background += image[z, y, x]

        if nBackground == 0:
            raise ValueError("There are no background pixel detected")
        meanBackground = background / nBackground
        if nSignal == 0:
            raise ValueError("There are no signal pixel detected")
        meanSignal = signal / nSignal
        ratio = float(meanSignal / meanBackground)
        return ratio

    def signalToBackgroundRatioRing(self):
        """Calculates the signal-to-background ratio (SBR) for a set of microscopy images using a ring-based method.
        The method iterates through the list of input images, applying the processSingleSBRRing method to each image to calculate the SBR.
        It uses a ThreadPoolExecutor to parallelize the processing of multiple images for improved performance.
        The calculated SBR values for each image are stored in the SBR attribute, and the mean SBR across all images is calculated and stored in the meanSBR attribute for further analysis and evaluation.
        Raises:
            ValueError: If there are no images in the input list or if any of the images have an incorrect format.
        Returns:
            None: The method does not return a value, but it updates the SBR and meanSBR attributes of the class with the calculated values for each image and the mean SBR across all images, respectively.
        """
        meanSBR = 0.0
        SBR = []

        if len(self._images) == 0:
            raise ValueError("You must have at least one PSF")
        total = len(self._images)
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    lambda i, img: self.processSingleSBRRing(i, img), i, image
                ): i
                for i, image in enumerate(self._images)
            }
            for future in as_completed(futures):
                result = future.result()
                if result == -1:
                    total += result
                else:
                    SBR.append(result)
                    meanSBR += result
        self.meanSBR = meanSBR / total
        self.SBR = SBR

    def runPrefittingMetrics(self):
        """Runs the pre-fitting metrics calculations, including signal-to-background ratio (SBR) calculation and theoretical resolution estimation.
        The method first calculates the SBR for the input images using the signalToBackgroundRatioRing method, and then estimates the theoretical resolution using the theoreticalResolutionTool.
        The calculated SBR values and theoretical resolution are stored in the class attributes for further analysis and evaluation.
        Raises:
            ValueError: If there are no images in the input list or if the theoreticalResolutionTool is not set.
        """
        self.SBR = []
        self.meanSBR = 0.0
        yield {"desc": "SBR calculation..."}
        self.signalToBackgroundRatioRing()
        yield {"desc": "Estimating theoretical resolution..."}
        self.theoreticalResolution = (
            self._TheoreticalResolutionTool.getTheoreticalResolution()
        )

    def lateralAsymmetryRatio(self):
        """Calculates the lateral asymmetry ratio (LAR) for the detected point spread function (PSF) based on the calculated full width at half maximum (FWHM) values.
        Raises:
           ValueError: If the FWHM values are not available or if there are not enough FWHM values to calculate the LAR.
        """
        if self._FWHM == []:
            raise ValueError(
                "FWHM values are not available or insufficient to calculate LAR."
            )
        tmp = np.array([self._FWHM[1], self._FWHM[2]])
        self.LAR = tmp.min() / tmp.max()

    def sphericityRatio(self):
        """Calculates the sphericity ratio for the detected point spread function (PSF) based on the calculated full width at half maximum (FWHM) values.
        Raises:
            ValueError: If the FWHM values are not available or if there are not enough FWHM values to calculate the sphericity ratio.
        """
        if self._FWHM == []:
            raise ValueError(
                "FWHM values are not available or insufficient to calculate sphericity."
            )
        FWHMxy = math.sqrt(self._FWHM[2] * self._FWHM[1])
        sphericity = FWHMxy / self._FWHM[0]
        self._sphericity = sphericity
