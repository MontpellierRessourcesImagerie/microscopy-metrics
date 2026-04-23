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
            for i, bead in enumerate(
                bead
                for bead in self._imageAnalyzer._beadAnalyzer
                if bead._rejected == False and bead._roi is not None
            ):
                writer.writerow([f"Bead {i}"])
                dataBead = [
                    ["", "Z", "Y", "X"],
                    [
                        "Centroid coordinates",
                        f"{bead._centroid[0]}",
                        f"{bead._centroid[1]}",
                        f"{bead._centroid[2]}",
                    ],
                ]
                writer.writerow(["Bead's datas"])
                writer.writerows(dataBead)
                writer.writerow([])
                dataFitting = [
                    ["", "Z", "Y", "X"],
                    [
                        "Theoretical resolution",
                        f"{self._imageAnalyzer._theoreticalResolution[0]}",
                        f"{self._imageAnalyzer._theoreticalResolution[1]}",
                        f"{self._imageAnalyzer._theoreticalResolution[2]}",
                    ],
                    [
                        "FWHM",
                        f"{bead._fitTool.fwhms[0]}",
                        f"{bead._fitTool.fwhms[1]}",
                        f"{bead._fitTool.fwhms[2]}",
                    ],
                    [
                        "Uncertainty",
                        f"{bead._fitTool.uncertainties[0][3]:.4f}",
                        f"{bead._fitTool.uncertainties[1][3]:.4f}",
                        f"{bead._fitTool.uncertainties[2][3]:.4f}",
                    ],
                    [
                        "Determination",
                        f"{bead._fitTool.determinations[0]:.4f}",
                        f"{bead._fitTool.determinations[1]:.4f}",
                        f"{bead._fitTool.determinations[2]:.4f}",
                    ],
                ]
                writer.writerow(["Fitting's datas"])
                writer.writerows(dataFitting)
                writer.writerow([])
                dataMetrics = [
                    ["Metric", "Value"],
                    ["Signal to background ratio", f"{bead._metricTool._SBR:.4f}"],
                    ["Lateral asymmetry ratio", f"{bead._metricTool._LAR:.4f}"],
                    ["Sphericity", f"{bead._metricTool._sphericity:.4f}"],
                    ["Banana magnitude", f"{bead._metricTool._banana:.4f}"]
                ]
                writer.writerow(["Metric's results"])
                writer.writerows(dataMetrics)
                writer.writerow([])
                writer.writerow([])
