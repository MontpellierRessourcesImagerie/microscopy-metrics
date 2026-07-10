import os

from concurrent.futures import ThreadPoolExecutor, as_completed

class Fitting(object):
    """Class to manage the fitting process for microscopy image analysis, including running the fitting for individual beads and computing the fitting results for all beads in the analysis.
    
    Attributes:
        fitType (str): The type of fitting to be performed (e.g., "1D", "2D", "3D").
        _thresholdRSquared (float): The threshold for the coefficient of determination (R²) used to filter out poor fits.
        _prominenceRel (float): The relative prominence used for peak detection during fitting.
        _imageAnalyzer (ImageAnalyzer): An instance of the ImageAnalyzer class used for analyzing the microscopy images and managing the fitting process.
    """

    def __init__(self):
        self.fitType = "1D"
        self._thresholdRSquared = 0.95
        self._prominenceRel = None
        self._imageAnalyzer = None

    def runFitting(self, index):
        """Runs the fitting process for a single index, using the specified fitting tool and storing the results.
        
        Args:
            index (int): The index of the fit being processed, used for retrieving the corresponding image, centroid, spacing, and ROI for the fitting process.
        """
        bead = self._imageAnalyzer._beadAnalyzer[index]
        bead.runFitting(
            fittingType=self.fitType,
            spacing=self._imageAnalyzer._pixelSize,
            outputDir=self._imageAnalyzer._path,
            prominenceRel=self._prominenceRel,
        )

    def computeFitting(self):
        """Computes the fitting for all provided images, centroids, spacing, and ROIs using the specified fitting tool.
        The method initializes a thread pool executor to run the fitting process concurrently for each index, improving performance when processing multiple images.
        The method also applies a threshold on the coefficient of determination (R²) to filter out fits that do not meet the specified quality criteria, retaining only those that exceed the threshold for further analysis.
        """
        self.results = []
        workers = int(os.cpu_count() * 0.75)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.runFitting, i): i
                for i, bead in enumerate(self._imageAnalyzer._beadAnalyzer)
                if bead._rejected == False and bead._roi is not None
            }
            for future in as_completed(futures):
                future.result()
        for bead in self._imageAnalyzer._beadAnalyzer:
            if bead._rejected == False and bead._roi is not None:
                if bead._fitTool is None:
                    bead._rejected = True
                    bead._rejectionDesc = "Fitting failed"
                meanDetermination = (
                    bead._fitTool.determinations[0]
                    + bead._fitTool.determinations[1]
                    + bead._fitTool.determinations[2]
                ) / 3.0
                if meanDetermination < self._thresholdRSquared:
                    bead._rejected = True
                    bead._rejectionDesc = "R² below threshold"
        beadsKept = len([b for b in self._imageAnalyzer._beadAnalyzer if b._rejected == False and b._roi is not None])
        if beadsKept == 0:
            print("No beads kept after fitting. Please check the fitting results and adjust the threshold if necessary.")
        for bead in self._imageAnalyzer._beadAnalyzer:
            if bead._rejected == False and bead._roi is not None:
                self._imageAnalyzer._meanDetermination[0] += bead._fitTool.determinations[0] / beadsKept
                self._imageAnalyzer._meanDetermination[1] += bead._fitTool.determinations[1] / beadsKept
                self._imageAnalyzer._meanDetermination[2] += bead._fitTool.determinations[2] / beadsKept
                self._imageAnalyzer._meanFWHM[0] += bead._fitTool.fwhms[0] / beadsKept
                self._imageAnalyzer._meanFWHM[1] += bead._fitTool.fwhms[1] / beadsKept
                self._imageAnalyzer._meanFWHM[2] += bead._fitTool.fwhms[2] / beadsKept
                self._imageAnalyzer._meanUncertainty[0] += bead._fitTool.uncertainties[0] / beadsKept
                self._imageAnalyzer._meanUncertainty[1] += bead._fitTool.uncertainties[1] / beadsKept
                self._imageAnalyzer._meanUncertainty[2] += bead._fitTool.uncertainties[2] / beadsKept

    def displayFitting(self, outputDir):
        """Displays the fitting results for all beads that were not rejected during the fitting process.
        
        Args:
            outputDir (str): The directory where the fitting results will be saved.
        """
        if self._imageAnalyzer is None:
            return None
        if len(self._imageAnalyzer._beadAnalyzer) == 0:
            return None
        for bead in self._imageAnalyzer._beadAnalyzer:
            if bead._rejected == False and bead._roi is not None:
                bead._fitTool.plotFit(os.path.join(outputDir, f"bead_{bead._id}"))
                bead._fitTool.showFit(os.path.join(outputDir, f"bead_{bead._id}"))