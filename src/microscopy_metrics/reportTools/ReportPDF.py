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
        t.drawOn(self.pdf, 40, 530)

    def drawSingleBeadRejectedReportPDF(self, bead):
        """Helper to write the report of a rejected bead on the pdf
        Args:
            bead (BeadAnalyzer): The bead for which we want to write the rejected report
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
            bead (BeadAnalyzer): The bead for which we want to write the report
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
            f"Comaticity: {f'{bead._metricTool._comaticity:.2f}' if bead._metricTool._comaticity is not None else 'Unknown'}",
            f"Spherical aberration: {f'{bead._metricTool._sphericalAberration:.2f}' if bead._metricTool._sphericalAberration is not None else 'Unknown'}",
            f"Astigmatism: {f'{bead._metricTool._astigmatism:.2f}' if bead._metricTool._astigmatism is not None else 'Unknown'}",
            f"Contrast: {f'{bead._fitTool.contrast:.2f}' if bead._fitTool.contrast is not None else 'Unknown'}",
            f"Ellipticity ratio: {f'{bead._metricTool._ellipsRatio:.2f}' if bead._metricTool._ellipsRatio is not None else 'Unknown'}",
            f"Orientation: {f'{bead._metricTool._orientation:.2f}' if bead._metricTool._orientation is not None else 'Unknown'}",

        ]
        self.drawParagaphOnPDF(textLines, 40, 640)
        data = [
            ["", "Z", "Y", "X"],
            [
                "Theoretical resolution",
                f"{self._imageAnalyzer._theoreticalResolution[0]:.4f}",
                f"{self._imageAnalyzer._theoreticalResolution[1]:.4f}",
                f"{self._imageAnalyzer._theoreticalResolution[2]:.4f}",
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
                375,
                width=150,
                height=150,
                preserveAspectRatio=True,
            )
        if os.path.exists(os.path.join(beadPath, "fit_curve_1D_X.png")):
            self.pdf.drawImage(
                os.path.join(beadPath, "fit_curve_1D_X.png"),
                300,
                300,
                width=250,
                height=250,
                preserveAspectRatio=True,
            )
        if os.path.exists(os.path.join(beadPath, "XZ_view.png")):
            self.pdf.drawImage(
                os.path.join(beadPath, "XZ_view.png"),
                50,
                175,
                width=150,
                height=150,
                preserveAspectRatio=True,
            )
        if os.path.exists(os.path.join(beadPath, "fit_curve_1D_Y.png")):
            self.pdf.drawImage(
                os.path.join(beadPath, "fit_curve_1D_Y.png"),
                300,
                100,
                width=250,
                height=250,
                preserveAspectRatio=True,
            )
        if os.path.exists(os.path.join(beadPath, "XY_view.png")):
            self.pdf.drawImage(
                os.path.join(beadPath, "XY_view.png"),
                50,
                0,
                width=150,
                height=150,
                preserveAspectRatio=True,
            )
        if os.path.exists(os.path.join(beadPath, "fit_curve_1D_Z.png")):
            self.pdf.drawImage(
                os.path.join(beadPath, "fit_curve_1D_Z.png"),
                300,
                -75,
                width=250,
                height=250,
                preserveAspectRatio=True,
            )
        self.pdf.showPage()

    def drawParameterTableOnPDF(self, title, data, y):
        """Helper to draw a styled parameter table with a title
        Args:
            title (str): The section title
            data (List[List[str]]): The key-value pairs for the table
            y (int): The y-coordinate to start drawing (top to bottom)
        Returns:
            int: The new y-coordinate after drawing the table
        """
        self.pdf.setFont("Helvetica-Bold", 18)
        self.pdf.drawCentredString(300, y, title)
        t = Table(data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        w, h = t.wrapOn(self.pdf, 0, 0)
        t.drawOn(self.pdf, 100, y - h - 10)
        return y - h - 50

    def generateReport(self, outputPath=None):
        """Generates a PDF report containing the results of the microscopy image analysis, including bead information, fitting results, and calculated metrics.
        Args:
            outputPath (str, optional): Path to the directory where the PDF report will be saved. Defaults to None.
        """
        if outputPath is None:
            outputPath = os.path.dirname(self._imageAnalyzer._path)
        pdfPath = os.path.join(outputPath, f"PSF_analysis_result.pdf")
        self.pdf = canvas.Canvas(pdfPath, pagesize=A4)
        self.pdf.setTitle("PSF analysis results")
        self.pdf.setFont("Helvetica-Bold", 28)
        self.pdf.drawCentredString(300, 770, "Analysis Parameters Report")
        self.pdf.setFont("Helvetica", 10)
        self.pdf.drawCentredString(300, 750, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        current_y = 700
        microscope_data = [
            ["Microscope type", f"{self._microscopeDatas.get('microscopeType', 'N/A')}"],
            ["Emission wavelength", f"{self._microscopeDatas.get('emissionWavelength', 'N/A')} nm"],
            ["Excitation wavelength", f"{self._microscopeDatas.get('excitationWavelength', 'N/A')} nm"],
            ["Refractive index", f"{self._microscopeDatas.get('refractiveIndex', 'N/A')}"],
            ["Numerical aperture", f"{self._microscopeDatas.get('numericalAperture', 'N/A')}"],
        ]
        current_y = self.drawParameterTableOnPDF("Microscope parameters", microscope_data, current_y)
        detection_data = [
            ["Detection method", f"{self._detectionDatas.get('detectionTool', 'N/A')}"],
            ["Minimal distance", f"{self._detectionDatas.get('minDist', 'N/A')}"],
            ["Sigma", f"{self._detectionDatas.get('sigma', 'N/A')}"],
            ["Threshold tool", f"{self._thresholdDatas.get('thresholdTool', 'N/A')}"],
            ["Threshold relative", f"{self._thresholdDatas.get('thresholdRel', 'N/A')}"],
        ]
        if self._thresholdDatas.get('thresholdTool') == 'manual':
            detection_data.append(["Threshold relative", f"{self._thresholdDatas.get('thresholdRel', 'N/A')}"])
        
        current_y = self.drawParameterTableOnPDF("Detection parameters", detection_data, current_y)
        extraction_data = [
            ["Bead size", f"{self._roiDatas.get('beadSize', 'N/A')}"],
            ["Crop factor", f"{self._roiDatas.get('cropFactor', 'N/A')}"],
            ["Distance ring-bead", f"{self._roiDatas.get('ringInnerDistance', 'N/A')}"],
            ["Ring thickness", f"{self._roiDatas.get('ringThickness', 'N/A')}"],
            ["Rejection distance", f"{self._roiDatas.get('rejectionDistance', 'N/A')}"],
            ["Intensity threshold", f"{self._roiDatas.get('thresholdIntensity', 'N/A')}"],
        ]
        current_y = self.drawParameterTableOnPDF("Extraction parameters", extraction_data, current_y)
        if current_y < 150:
            self.pdf.showPage()
            current_y = 750
        fitting_data = [
            ["Fitting type", f"{self._fittingDatas.get('fitType', 'N/A')}"],
            ["R² threshold", f"{self._fittingDatas.get('thresholdRSquared', 'N/A')}"],
            ["Prominence relative", f"{self._fittingDatas.get('prominenceRel', 'N/A')}"],
        ]
        self.drawParameterTableOnPDF("Fitting parameters", fitting_data, current_y)
        self.pdf.showPage()
        if self._imageAnalyzer is not None :
            self._imageAnalyzer.generatePDFReport(self.pdf)
        self.pdf.save()