import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed


class Metrics(object):
    """Class for calculating various metrics related to microscopy images, including signal-to-background ratio (SBR), lateral asymmetry ratio (LAR), sphericity, and theoretical resolution.
    This class provides methods for processing microscopy images, calculating metrics based on the image data, and storing the results for further analysis and evaluation.
    """

    def __init__(self):
        self._imageAnalyze = None
        self._ringInnerDistance = None
        self._ringThickness = None

        self._TheoreticalResolutionTool = None
        self.meanSBR = 0.0

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
        self.meanSBR = 0.0

        if len(self._imageAnalyze._beadAnalyze) == 0:
            raise ValueError("You must have at least one PSF")
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(bead.runSBRMetric, self._imageAnalyze._pixelSize, self._ringInnerDistance, self._ringThickness): i
                for i, bead in enumerate(self._imageAnalyze._beadAnalyze) if bead._rejected == False and bead._roi is not None
            }
            for future in as_completed(futures):
                result = future.result()
        total = 0
        for bead in self._imageAnalyze._beadAnalyze:
            if bead._rejected == False and bead._roi is not None:
                self.meanSBR += bead._metricTool._SBR
                total += 1
        self.meanSBR = self.meanSBR / total

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
