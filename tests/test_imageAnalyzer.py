import os
import pytest
import numpy as np
from skimage.draw import disk
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from microscopy_metrics.BeadAnalyzer import BeadAnalyzer
from microscopy_metrics.ImageAnalyzer import ImageAnalyzer

def test_image_analyzer_toDict():
    """Unit test for ImageAnalyzer class."""
    image_analyzer = ImageAnalyzer()
    assert (image_analyzer.toDict()['path'] == "~/")


def test_image_analyzer_generateReport(tmp_path):
    """Unit test for ImageAnalyzer class."""
    image_analyzer = ImageAnalyzer()
    report_path = tmp_path / "image_report.txt"
    pdf = canvas.Canvas(report_path, pagesize=A4)
    pdf.setTitle("PSF analysis results - Simplified")
    pdf.setFont("Helvetica-Bold", 28)
    pdf.drawCentredString(300, 770, "Analysis Parameters Report - Simplified")
    pdf.setFont("Helvetica", 10)
    image_analyzer.generatePDFReport(pdf)

def test_image_analyzer_generateSimplifiedPDFReport(tmp_path):
    """Unit test for ImageAnalyzer class."""
    image_analyzer = ImageAnalyzer()
    report_path = tmp_path / "image_report.txt"
    pdf = canvas.Canvas(report_path, pagesize=A4)
    pdf.setTitle("PSF analysis results - Simplified")
    pdf.setFont("Helvetica-Bold", 28)
    pdf.drawCentredString(300, 770, "Analysis Parameters Report - Simplified")
    pdf.setFont("Helvetica", 10)
    image_analyzer.generateSimplifiedPDFReport(pdf)