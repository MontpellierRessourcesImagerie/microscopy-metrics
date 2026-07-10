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
        self.generateSimplifiedReport(outputPath)

    def generateSimplifiedReport(self, outputPath=None):
        """Generates a simplified PDF report containing the results of the microscopy image analysis, including bead information, fitting results, and calculated metrics.
        
        Args:
            outputPath (str, optional): Path to the directory where the PDF report will be saved. Defaults to None.
        """
        if outputPath is None:
            outputPath = os.path.dirname(self._imageAnalyzer._path)
        pdfPath = os.path.join(outputPath, f"PSF_analysis_result_simplified.pdf")
        self.pdf = canvas.Canvas(pdfPath, pagesize=A4)
        self.pdf.setTitle("PSF analysis results - Simplified")
        self.pdf.setFont("Helvetica-Bold", 28)
        self.pdf.drawCentredString(300, 770, "Analysis Parameters Report - Simplified")
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
            self._imageAnalyzer.generateSimplifiedPDFReport(self.pdf)
        self.pdf.save()