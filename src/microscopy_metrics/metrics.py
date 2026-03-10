import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from skimage.exposure import adjust_sigmoid
from sklearn.metrics import r2_score
from scipy.ndimage import median_filter
import math
from .utils import *
from matplotlib import pyplot as plt
from .fitting import *
from .theoretical_resolution import *
from .threshold_tool import ThresholdLegacy
        

class Metrics(object):
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
        """Method to normalize a 2D or 3D image and erase negative values

        Args:
            image (np.ndarray): Image to be normalized

        Raises:
            ValueError: This function only operate on 2D or 3D images

        Returns:
            np.ndarray: Image normalized
        """
        if image.ndim not in (2, 3):
            raise ValueError("Image have to be in 2D or 3D.")
        imageFloat = image.astype(np.float32)
        imageFloat = (imageFloat - np.min(imageFloat)) / (
                np.max(imageFloat) - np.min(imageFloat) + 1e-6
        )
        imageFloat[imageFloat < 0] = 0
        return imageFloat

    def processSingleSBRRing(self, index, image):
        """Function to calculate Signa to background ratio using a ring for a specific bead

        Args:
            index (int): Bead ID corresping to it's position in the list
            image (np.ndarray): Image to use for the calculation

        Raises:
            ValueError: There have to be at least one background pixel in the image
            ValueError: There have to be at least one signal pixel in the image

        Returns:
            float: Signal to background ratio of the image
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
        outerDistance = (
                umToPx(self._ringThickness, self._pixelSize[2]) + innerDistance
        )
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
        self.SBR = []
        self.meanSBR = 0.0
        yield {"desc": "SBR calculation..."}
        self.signalToBackgroundRatioRing()
        yield {"desc": "Estimating theoretical resolution..."}
        self.theoreticalResolution = (
            self._TheoreticalResolutionTool.getTheoreticalResolution()
        )

    def lateralAsymmetryRatio(self):
        if self._FWHM == []:
            return
        tmp = np.array([self._FWHM[1], self._FWHM[2]])
        self.LAR = tmp.min() / tmp.max()

    def sphericityRatio(self):
        if self._FWHM == []:
            return
        FWHMxy = math.sqrt(self._FWHM[2] * self._FWHM[1])
        sphericity = FWHMxy / self._FWHM[0]
        self._sphericity = sphericity
