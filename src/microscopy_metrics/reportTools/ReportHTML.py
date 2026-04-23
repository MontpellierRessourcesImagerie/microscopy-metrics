import os
from jinja2 import Environment, FileSystemLoader

from microscopy_metrics.report_generator import ReportGenerator


class ReportHTML(ReportGenerator):
    """Class based on the ReportGenerator abstract base class that generates an HTML report based on microscopy image analysis results."""

    name = "HTML"

    def __init__(self):
        super().__init__()

    def generateReport(self, outputPath=None):
        """Generates an HTML report containing the results of the microscopy image analysis, including bead information, fitting results, and calculated metrics.
        Args:
            outputPath (str, optional): Path to the directory where the HTML report will be saved. Defaults to None.
        """
        if outputPath is None:
            outputPath = os.path.dirname(self._imageAnalyzer._path)
        templateDir = os.path.join(os.path.dirname(__file__), "res", "template")
        env = Environment(loader=FileSystemLoader(templateDir))
        template = env.get_template("report_template.html")
        for bead in self._imageAnalyzer._beadAnalyzer:
            if bead._rejected == False and bead._roi is not None:
                beadPath = os.path.join(outputPath, f"bead_{bead._id}")
                activePath = os.path.join(beadPath, "report.html")
                data = {
                    "title": f"Bead {bead._id}",
                    "bead": bead,
                    "theoretical_resolution": self._imageAnalyzer._theoreticalResolution,
                    "path": beadPath,
                }
                htmlContent = template.render(data)
                with open(activePath, "w") as f:
                    f.write(htmlContent)
        mainTemplate = env.get_template("main_report_template.html")
        data = {
            "title": f"Microscopy Metrics Report - {os.path.basename(self._imageAnalyzer._path)}",
            "beads": self._imageAnalyzer._beadAnalyzer,
        }
        mainHtmlContent = mainTemplate.render(data)
        mainReportPath = os.path.join(outputPath, "index.html")
        with open(mainReportPath, "w") as f:
            f.write(mainHtmlContent)
