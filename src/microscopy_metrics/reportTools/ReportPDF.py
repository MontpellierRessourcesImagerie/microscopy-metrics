import os

from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, Table, TableStyle

from microscopy_metrics.report_generator import ReportGenerator


class ReportPDF(ReportGenerator):
    """Class based on the ReportGenerator abstract base class that generates a PDF report based on microscopy image analysis results."""

    name = "PDF"

    def __init__(self):
        super().__init__()
        self._pdf = None
        self.yRejected = 750

    def drawParagaphOnPDF(self, textLines, x, y):
        """Helper to quickly write a paragraph on the pdf
        Args:
            textLines (List(String)): The list of lines we want to write on the pdf
            x (int): x coordinate of the paragraph
            y (int): y coordinate of the paragraph
        """
        stylesheet = getSampleStyleSheet()
        normalStyle = stylesheet["Normal"]
        fullText = "<br/>".join(textLines)
        p = Paragraph(fullText, normalStyle)
        p.wrapOn(self.pdf, 500, 100)
        p.drawOn(self.pdf, x, y)

    def drawTableOnPDF(self, data):
        """Helper to quickly add a table on the pdf
        Args:
            data (Matrix(String)): The table to write on the pdf
        """
        s = getSampleStyleSheet()
        s = s["BodyText"]
        s.wordWrap = "CJK"
        data2 = [[Paragraph(cell, s) for cell in row] for row in data]
        t = Table(data=data2, colWidths=[80, 50, 50, 50])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        t.wrapOn(self.pdf, 0, 0)
        t.drawOn(self.pdf, 40, 590)

    def drawSingleBeadRejectedReportPDF(self, bead):
        """Helper to write the report of a rejected bead on the pdf
        Args:
            bead (BeadAnalyze): The bead for which we want to write the rejected report
        """
        stylesheet = getSampleStyleSheet()
        normalStyle = stylesheet["Normal"]
        textLines = [
            f"Bead {bead._id if bead._id is not None else 'Unknown'} analysis : REJECTED",
            f"centroid: {', '.join(f'{c:.2f}' for c in bead._centroid) if bead._centroid is not None else 'Unknown'}",
            f"Rejection reason: {bead._rejectionDesc if bead._rejectionDesc != '' else 'Unknown'}",
        ]
        text = self.pdf.beginText(40, self.yRejected)
        text.setFont(normalStyle.fontName, normalStyle.fontSize)
        for line in textLines:
            text.textLine(line)
        self.pdf.drawText(text)

    def drawSingleBeadReportPDF(self, bead):
        """Helper to write the report of a single bead on the pdf
        Args:
            bead (BeadAnalyze): The bead for which we want to write the report
        """
        beadPath = os.path.join(self._inputDir, f"bead_{bead._id}")
        self.pdf.setFont("Helvetica-Bold", 36)
        self.pdf.drawCentredString(
            150, 770, f"Bead {bead._id if bead._id is not None else 'Unknown'}"
        )
        if os.path.exists(os.path.join(beadPath, "Localisation.png")):
            self.pdf.drawImage(
                os.path.join(beadPath, "Localisation.png"),
                350,
                600,
                width=200,
                height=200,
                preserveAspectRatio=True,
            )
        textLines = [
            f"centroid: {', '.join(f'{c:.2f}' for c in bead._centroid) if bead._centroid is not None else 'Unknown'}",
            f"Signal to Background ratio: {f'{bead._metricTool._SBR:.2f}' if bead._metricTool._SBR is not None else 'Unknown'}",
            f"Lateral asymmetry ratio: {f'{bead._metricTool._LAR:.2f}' if bead._metricTool._LAR is not None else 'Unknown'}",
            f"Sphericity: {f'{bead._metricTool._sphericity:.2f}' if bead._metricTool._sphericity is not None else 'Unknown'}",
        ]
        self.drawParagaphOnPDF(textLines, 40, 700)
        data = [
            ["", "Z", "Y", "X"],
            [
                "Theoretical resolution",
                f"{self._imageAnalyze._theoreticalResolution[0]:.4f}",
                f"{self._imageAnalyze._theoreticalResolution[1]:.4f}",
                f"{self._imageAnalyze._theoreticalResolution[2]:.4f}",
            ],
            [
                "FWHM (µm)",
                f"{bead._fitTool.fwhms[0]:.4f}",
                f"{bead._fitTool.fwhms[1]:.4f}",
                f"{bead._fitTool.fwhms[2]:.4f}",
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
        self.drawTableOnPDF(data)
        if os.path.exists(os.path.join(beadPath, "YZ_view.png")):
            self.pdf.drawImage(
                os.path.join(beadPath, "YZ_view.png"),
                50,
                425,
                width=150,
                height=150,
                preserveAspectRatio=True,
            )
        if os.path.exists(os.path.join(beadPath, "fit_curve_1D_X.png")):
            self.pdf.drawImage(
                os.path.join(beadPath, "fit_curve_1D_X.png"),
                300,
                375,
                width=250,
                height=250,
                preserveAspectRatio=True,
            )
        if os.path.exists(os.path.join(beadPath, "XZ_view.png")):
            self.pdf.drawImage(
                os.path.join(beadPath, "XZ_view.png"),
                50,
                225,
                width=150,
                height=150,
                preserveAspectRatio=True,
            )
        if os.path.exists(os.path.join(beadPath, "fit_curve_1D_Y.png")):
            self.pdf.drawImage(
                os.path.join(beadPath, "fit_curve_1D_Y.png"),
                300,
                175,
                width=250,
                height=250,
                preserveAspectRatio=True,
            )
        if os.path.exists(os.path.join(beadPath, "XY_view.png")):
            self.pdf.drawImage(
                os.path.join(beadPath, "XY_view.png"),
                50,
                25,
                width=150,
                height=150,
                preserveAspectRatio=True,
            )
        if os.path.exists(os.path.join(beadPath, "fit_curve_1D_Z.png")):
            self.pdf.drawImage(
                os.path.join(beadPath, "fit_curve_1D_Z.png"),
                300,
                -25,
                width=250,
                height=250,
                preserveAspectRatio=True,
            )
        self.pdf.showPage()

    def generateReport(self, outputPath=None):
        """Generates a PDF report containing the results of the microscopy image analysis, including bead information, fitting results, and calculated metrics.
        Args:
            outputPath (str, optional): Path to the directory where the PDF report will be saved. Defaults to None.
        """
        if outputPath is None:
            outputPath = os.path.dirname(self._imageAnalyze._path)
        pdfPath = os.path.join(outputPath, f"PSF_analysis_result.pdf")
        self.pdf = canvas.Canvas(pdfPath, pagesize=A4)
        self.pdf.setTitle("PSF analysis results")
        self.pdf.setFont("Helvetica-Bold", 36)
        self.pdf.drawCentredString(300, 770, "Analysis Results")
        textLines = [
            f"Date and time of analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Image location: {self._imageAnalyze._path}",
            f"Identified beads: {len(self._imageAnalyze._beadAnalyze)}",
            f"Signal to background ratio: {self._imageAnalyze._meanSBR:.2f}",
        ]
        self.drawParagaphOnPDF(textLines, 40, 680)
        self.pdf.setFont("Helvetica-Bold", 18)
        self.pdf.drawCentredString(300, 650, "Acquisition parameters")
        textLines = [
            f"Pixel size: [{self._imageAnalyze._pixelSize[0]},{self._imageAnalyze._pixelSize[1]},{self._imageAnalyze._pixelSize[2]}]",
            f"Image shape: [{self._imageAnalyze._image.shape[0]},{self._imageAnalyze._image.shape[1]},{self._imageAnalyze._image.shape[2]}]",
            f"Microscope type: {self._microscopeDatas['microscopeType'] if 'microscopeType' in self._microscopeDatas else 'N/A'}",
            f"Emission wavelength: {self._microscopeDatas['emissionWavelength'] if 'emissionWavelength' in self._microscopeDatas else 'N/A'}nm",
            f"Refractive index: {self._microscopeDatas['refractiveIndex'] if 'refractiveIndex' in self._microscopeDatas else 'N/A'}",
            f"Numerical aperture: {self._microscopeDatas['numericalAperture'] if 'numericalAperture' in self._microscopeDatas else 'N/A'}",
        ]
        self.drawParagaphOnPDF(textLines, 40, 600)
        self.pdf.setFont("Helvetica-Bold", 18)
        self.pdf.drawCentredString(300, 500, "Detection parameters")
        textLines = [
            f"Detection method: {self._detectionDatas['detectionTool'] if 'detectionTool' in self._detectionDatas else 'N/A'}",
            f"Minimal distance: {self._detectionDatas['minDist'] if 'minDist' in self._detectionDatas else 'N/A'}",
            f"Sigma: {self._detectionDatas['sigma'] if 'sigma' in self._detectionDatas else 'N/A'}",
            f"Threshold tool: {self._thresholdDatas['thresholdTool'] if 'thresholdTool' in self._thresholdDatas else 'N/A'}",
            f"Threshold relative: {self._thresholdDatas['thresholdRel'] if 'thresholdRel' in self._thresholdDatas and self._thresholdDatas['thresholdTool'] == 'manual' else 'N/A'}",
        ]
        self.drawParagaphOnPDF(textLines, 40, 450)
        self.pdf.setFont("Helvetica-Bold", 18)
        self.pdf.drawCentredString(300, 350, "Extraction parameters")
        textLines = [
            f"Bead size: {self._roiDatas['beadSize'] if 'beadSize' in self._roiDatas else 'N/A'}",
            f"Crop factor: {self._roiDatas['cropFactor'] if 'cropFactor' in self._roiDatas else 'N/A'}",
            f"Distance ring-bead: {self._roiDatas['ringInnerDistance'] if 'ringInnerDistance' in self._roiDatas else 'N/A'}",
            f"Ring thickness: {self._roiDatas['ringThickness'] if 'ringThickness' in self._roiDatas else 'N/A'}",
            f"Rejection distance: {self._roiDatas['rejectionDistance'] if 'rejectionDistance' in self._roiDatas else 'N/A'}",
            f"Intensity threshold: {self._roiDatas['thresholdIntensity'] if 'thresholdIntensity' in self._roiDatas else 'N/A'}",
        ]
        self.drawParagaphOnPDF(textLines, 40, 300)
        self.pdf.setFont("Helvetica-Bold", 18)
        self.pdf.drawCentredString(300, 150, "Fitting parameters")
        textLines = [
            f"Fitting type: {self._fittingDatas['fitType'] if 'fitType' in self._fittingDatas else 'N/A'}",
            f"R² threshold: {self._fittingDatas['thresholdRSquared'] if 'thresholdRSquared' in self._fittingDatas else 'N/A'}",
            f"Prominence relative: {self._fittingDatas['prominenceRel'] if 'prominenceRel' in self._fittingDatas else 'N/A'}",
        ]
        self.drawParagaphOnPDF(textLines, 40, 100)
        if any(bead._rejected == True for bead in self._imageAnalyze._beadAnalyze):
            self.pdf.showPage()
            self.pdf.setFont("Helvetica-Bold", 36)
            self.pdf.drawCentredString(150, 770, "REJECTED")
        for bead in self._imageAnalyze._beadAnalyze:
            if bead._rejected == True:
                if self.yRejected < 50:
                    self.pdf.showPage()
                    self.pdf.setFont("Helvetica-Bold", 36)
                    self.pdf.drawCentredString(150, 770, "REJECTED")
                    self.yRejected = 750
                self.drawSingleBeadRejectedReportPDF(bead)
                self.yRejected -= 50
        self.pdf.showPage()
        for bead in self._imageAnalyze._beadAnalyze:
            if bead._rejected == False and bead._roi is not None:
                self.drawSingleBeadReportPDF(bead)
        self.pdf.save()
