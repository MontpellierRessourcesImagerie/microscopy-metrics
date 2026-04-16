import os

from concurrent.futures import ThreadPoolExecutor, as_completed

from microscopy_metrics.fittingTools.fittingTool import FittingTool


class Fitting(object):
    """Class for performing fitting of microscopy images using various fitting tools.
    This class manages the fitting process, including setting images, centroids, spacing, regions of interest (ROIs), and output directory.
    It also handles the execution of fitting using the specified fitting tool and stores the results.
    """

    def __init__(self):
        self.fitType = "1D"
        self._thresholdRSquared = 0.95
        self._prominenceRel = None
        self._imageAnalyze = None


    def runFitting(self, index):
        """Runs the fitting process for a single index, using the specified fitting tool and storing the results.
        The method retrieves the local centroid of the image, extracts the corresponding image, spacing, and ROI based on the provided index, and then initializes the fitting tool with these parameters.
        The fitting process is executed, and the results are returned in a structured format for further analysis and evaluation.
        Args:
            index (int): The index of the fit being processed, used for retrieving the corresponding image, centroid, spacing, and ROI for the fitting process.
        Returns:
            list: A list containing the index, calculated FWHM values, uncertainties, coefficient of determination, parameters for the fit, and covariance matrix for the fit.
        """
        bead = self._imageAnalyze._beadAnalyze[index]
        bead.runFitting(fittingType=self.fitType, spacing=self._imageAnalyze._pixelSize, outputDir=self._imageAnalyze._path, prominenceRel=self._prominenceRel)

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
                for i, bead in enumerate(self._imageAnalyze._beadAnalyze) 
                if bead._rejected == False and bead._roi is not None
            }
            for future in as_completed(futures):
                future.result()
        for bead in self._imageAnalyze._beadAnalyze:
            if bead._rejected == False and bead._roi is not None:
                if bead._fitTool is None:
                    bead._rejected = True
                    bead._rejectionDesc = "Fitting failed"
                meanDetermination = (bead._fitTool.determinations[0] + bead._fitTool.determinations[1] + bead._fitTool.determinations[2]) / 3.0
                if meanDetermination < self._thresholdRSquared:
                    bead._rejected = True
                    bead._rejectionDesc = "R² below threshold"
