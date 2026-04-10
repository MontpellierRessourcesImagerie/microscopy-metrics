import os

from concurrent.futures import ThreadPoolExecutor, as_completed

from microscopy_metrics.fittingTools.fittingTool import FittingTool


class Fitting(object):
    """Class for performing fitting of microscopy images using various fitting tools.
    This class manages the fitting process, including setting images, centroids, spacing, regions of interest (ROIs), and output directory.
    It also handles the execution of fitting using the specified fitting tool and stores the results.
    """

    def __init__(self):
        self._images = []
        self._centroids = []
        self._spacing = [1, 1, 1]
        self._rois = []
        self._outputDir = ""
        self.results = []
        self.fitType = "1D"
        self._thresholdRSquared = 0.95
        self.retainedId = []
        self._prominenceRel = None

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        if len(images) == 0 or images is None:
            raise ValueError("Please, send at list one image")
        self._images = images

    @property
    def centroids(self):
        return self._centroids

    @centroids.setter
    def centroids(self, centroids):
        if len(centroids) == 0 or centroids is None:
            raise ValueError("Please, send at list one centroid")
        self._centroids = centroids

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, value):
        if value is None or len(value) == 0:
            raise ValueError("Shape format not compatible with current image")
        self._spacing = value

    @property
    def rois(self):
        return self._rois

    @rois.setter
    def rois(self, rois):
        if len(rois) == 0 or rois is None:
            raise ValueError("Please, send at list one ROI")
        self._rois = rois

    @property
    def outputDir(self):
        return self._outputDir

    @outputDir.setter
    def outputDir(self, value):
        if value is None or not os.path.exists(value):
            raise ValueError("The outputDir is wrong")
        self._outputDir = value

    def runFitting(self, index):
        """Runs the fitting process for a single index, using the specified fitting tool and storing the results.
        The method retrieves the local centroid of the image, extracts the corresponding image, spacing, and ROI based on the provided index, and then initializes the fitting tool with these parameters.
        The fitting process is executed, and the results are returned in a structured format for further analysis and evaluation.
        Args:
            index (int): The index of the fit being processed, used for retrieving the corresponding image, centroid, spacing, and ROI for the fitting process.
        Returns:
            list: A list containing the index, calculated FWHM values, uncertainties, coefficient of determination, parameters for the fit, and covariance matrix for the fit.
        """
        fitTool = FittingTool.getInstance(self.fitType)
        fitTool._image = self._images[index]
        fitTool._centroid = self._centroids[index]
        fitTool._spacing = self.spacing
        fitTool._roi = self._rois[index]
        fitTool._outputDir = self._outputDir
        if hasattr(fitTool, "_prominenceRel") and self._prominenceRel is not None:
            fitTool._prominenceRel = self._prominenceRel
        fitTool.processSingleFit(index)
        return [
            index,
            fitTool.fwhms,
            fitTool.uncertainties,
            fitTool.determinations,
            fitTool.parameters,
            fitTool.pcovs,
        ]

    def computeFitting(self):
        """Computes the fitting for all provided images, centroids, spacing, and ROIs using the specified fitting tool.
        The method initializes a thread pool executor to run the fitting process concurrently for each index, improving performance when processing multiple images.
        The results of the fitting process are collected and stored in the class attributes for further analysis and evaluation.
        The method also applies a threshold on the coefficient of determination (R²) to filter out fits that do not meet the specified quality criteria, retaining only those that exceed the threshold for further analysis.
        """
        self.results = []
        workers = int(os.cpu_count() * 0.75)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.runFitting, i): i
                for i, roi in enumerate(self._rois)
            }
            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
        tmp = []
        self.retainedId = []
        for i, result in enumerate(self.results):
            if result is None:
                print(f"Bead {i} is None")
                continue
            meanDetermination = (result[3][0] + result[3][1] + result[3][2]) / 3.0
            if meanDetermination >= self._thresholdRSquared:
                tmp.append(result)
                self.retainedId.append(result[0])
        if len(tmp) > 0:
            self.results = tmp
