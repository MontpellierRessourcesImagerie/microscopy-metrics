import os
import csv

from microscopy_metrics.report_generator import ReportGenerator


class ReportCSV(ReportGenerator):
    """Class based on the ReportGenerator abstract base class that generates a CSV report based on microscopy image analysis results."""

    name = "CSV"

    def __init__(self):
        super().__init__()

    def generateReport(self, outputPath=None):
        """Generates a CSV report containing the results of the microscopy image analysis, including bead information, fitting results, and calculated metrics.
        Args:
            outputPath (str, optional): Path to the directory where the CSV report will be saved. Defaults to None.
        """
        if outputPath is None:
            outputPath = os.path.dirname(self._imageAnalyzer._path)
        CSVPath = os.path.join(outputPath, "PSF_analysis_result.csv")
        with open(CSVPath, mode="w", newline="") as file:
            writer = csv.writer(file)
            metrics =(
                "Bead ID",
                "Centroid coordinates Z",
                "Centroid coordinates Y",
                "Centroid coordinates X",
                "Theoretical resolution Z",
                "Theoretical resolution Y",
                "Theoretical resolution X",
                "FWHM Z",
                "FWHM Y",
                "FWHM X",
                "Uncertainty Z",
                "Uncertainty Y",
                "Uncertainty X",
                "Determination (R²) Z",
                "Determination (R²) Y",
                "Determination (R²) X",
                "Signal to background ratio",
                "Contrast",
                "Ellipticity ratio",
                "Sphericity",
                "Lateral asymmetry ratio",
                "Orientation",
                "Comaticity",
                "Skeleton to extremities ratio",
                "Concavity",
                "Astigmatism",
                "Spherical aberration",
            )
            beadsData = [metrics]
            for i, bead in enumerate(
                bead
                for bead in self._imageAnalyzer._beadAnalyzer
                if bead._rejected == False and bead._roi is not None
            ):
                dataBead = (
                    f"{bead._id if bead._id is not None else 'N/A'}",
                    f"{bead._centroid[0] if bead._centroid[0] is not None else 'N/A'}",
                    f"{bead._centroid[1] if bead._centroid[1] is not None else 'N/A'}",
                    f"{bead._centroid[2] if bead._centroid[2] is not None else 'N/A'}",
                    f"{self._imageAnalyzer._theoreticalResolution[0] if self._imageAnalyzer._theoreticalResolution[0] is not None else 'N/A'}",
                    f"{self._imageAnalyzer._theoreticalResolution[1] if self._imageAnalyzer._theoreticalResolution[1] is not None else 'N/A'}",
                    f"{self._imageAnalyzer._theoreticalResolution[2] if self._imageAnalyzer._theoreticalResolution[2] is not None else 'N/A'}",
                    f"{bead._fitTool.fwhms[0] if bead._fitTool.fwhms[0] is not None else 'N/A'}",
                    f"{bead._fitTool.fwhms[1] if bead._fitTool.fwhms[1] is not None else 'N/A'}",
                    f"{bead._fitTool.fwhms[2] if bead._fitTool.fwhms[2] is not None else 'N/A'}",
                    f"{bead._fitTool.uncertainties[0][3] if bead._fitTool.uncertainties[0][3] is not None else 'N/A'}",
                    f"{bead._fitTool.uncertainties[1][3] if bead._fitTool.uncertainties[1][3] is not None else 'N/A'}",
                    f"{bead._fitTool.uncertainties[2][3] if bead._fitTool.uncertainties[2][3] is not None else 'N/A'}",
                    f"{bead._fitTool.determinations[0] if bead._fitTool.determinations[0] is not None else 'N/A'}",
                    f"{bead._fitTool.determinations[1] if bead._fitTool.determinations[1] is not None else 'N/A'}",
                    f"{bead._fitTool.determinations[2] if bead._fitTool.determinations[2] is not None else 'N/A'}",
                    f"{bead._metricTool._SBR if bead._metricTool._SBR is not None else 'N/A'}",
                    f"{bead._fitTool.contrast if bead._fitTool.contrast is not None else 'N/A'}",
                    f"{bead._metricTool._ellipsRatio if bead._metricTool._ellipsRatio is not None else 'N/A'}",
                    f"{bead._metricTool._sphericity if bead._metricTool._sphericity is not None else 'N/A'}",
                    f"{bead._metricTool._LAR if bead._metricTool._LAR is not None else 'N/A'}",
                    f"{bead._metricTool._orientation if bead._metricTool._orientation is not None else 'N/A'}",
                    f"{bead._metricTool._comaticity if bead._metricTool._comaticity is not None else 'N/A'}",
                    f"{bead._metricTool._skeleton2Extremities if bead._metricTool._skeleton2Extremities is not None else 'N/A'}",
                    f"{bead._metricTool._RMin if bead._metricTool._RMin is not None else 'N/A'}",
                    f"{bead._metricTool.meshBuilder._concavity if bead._metricTool.meshBuilder._concavity is not None else 'N/A'}",
                    f"{bead._metricTool._astigmatism if bead._metricTool._astigmatism is not None else 'N/A'}",
                    f"{bead._metricTool._sphericalAberration if bead._metricTool._sphericalAberration is not None else 'N/A'}",
                )   
                beadsData.append(dataBead)
            generalData = (
                "Average",
                "",
                "",
                "",
                f"{self._imageAnalyzer._theoreticalResolution[0] if self._imageAnalyzer._theoreticalResolution[0] is not None else 'N/A'}",
                f"{self._imageAnalyzer._theoreticalResolution[1] if self._imageAnalyzer._theoreticalResolution[1] is not None else 'N/A'}",
                f"{self._imageAnalyzer._theoreticalResolution[2] if self._imageAnalyzer._theoreticalResolution[2] is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanFWHM[0] if self._imageAnalyzer._meanFWHM[0] is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanFWHM[1] if self._imageAnalyzer._meanFWHM[1] is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanFWHM[2] if self._imageAnalyzer._meanFWHM[2] is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanUncertainty[0][3] if self._imageAnalyzer._meanUncertainty[0][3] is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanUncertainty[1][3] if self._imageAnalyzer._meanUncertainty[1][3] is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanUncertainty[2][3] if self._imageAnalyzer._meanUncertainty[2][3] is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanDetermination[0] if self._imageAnalyzer._meanDetermination[0] is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanDetermination[1] if self._imageAnalyzer._meanDetermination[1] is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanDetermination[2] if self._imageAnalyzer._meanDetermination[2] is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanSBR if self._imageAnalyzer._meanSBR is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanContrast if self._imageAnalyzer._meanContrast is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanEllipsRatio if self._imageAnalyzer._meanEllipsRatio is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanSphericity if self._imageAnalyzer._meanSphericity is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanLAR if self._imageAnalyzer._meanLAR is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanOrientation if self._imageAnalyzer._meanOrientation is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanComaticity if self._imageAnalyzer._meanComaticity is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanSkeleton2Extremities if self._imageAnalyzer._meanSkeleton2Extremities is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanRMin if self._imageAnalyzer._meanRMin is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanAstigmatism if self._imageAnalyzer._meanAstigmatism is not None else 'N/A'}",
                f"{self._imageAnalyzer._meanSphericalAberration if self._imageAnalyzer._meanSphericalAberration is not None else 'N/A'}",
            )
            beadsData.append(generalData)
            formated = zip(*beadsData)
            for row in formated:
                writer.writerow(row)
