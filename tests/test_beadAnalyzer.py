import os
import pytest
import numpy as np
from skimage.draw import disk
from microscopy_metrics.BeadAnalyzer import BeadAnalyzer
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from microscopy_metrics.fittingTools.fitting3D import Fitting3D

PSF_SIZE = 100

@pytest.fixture
def image():
    shape = (50, 100, 100)
    bead_positions = [(25, 50, 50)]
    image = np.zeros(shape, dtype=np.uint8)
    for z, y, x in bead_positions:
        rr, cc = disk((x, y), 3, shape=shape[1:])
        image[z, cc, rr] = 255
    yield image

@pytest.fixture
def psf():
    fitTool = Fitting3D()
    fitTool._show = False
    params = [
        255,
        0,
        PSF_SIZE / 2,
        PSF_SIZE / 2,
        PSF_SIZE / 2,
        PSF_SIZE / 10,
        PSF_SIZE / 10,
        PSF_SIZE / 10,
    ]
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    psf = fitTool.gauss(*params)(coords)
    FWHM = [fitTool.fwhm(params[5]), fitTool.fwhm(params[6]), fitTool.fwhm(params[7])]
    yield psf, FWHM

def test_bead_analyzer_toDict(image):
    """Unit test for BeadAnalyzer class."""
    bead_analyzer = BeadAnalyzer()
    bead_analyzer._image = image
    bead_analyzer._centroid = np.array([25, 50, 50])
    bead_analyzer._roi = [[50,0,0], [50,100,0], [50,100,100], [50,0,100]]
    assert (bead_analyzer.toDict()['centroid'].tolist() == [25, 50, 50])

def test_bead_analyzer_runSBRMetric(image):
    """Unit test for BeadAnalyzer class."""
    bead_analyzer = BeadAnalyzer()
    bead_analyzer._image = image
    bead_analyzer._centroid = np.array([25, 50, 50])
    bead_analyzer._roi = [[50,0,0], [50,100,0], [50,100,100], [50,0,100]]
    bead_analyzer.runSBRMetric()
    assert bead_analyzer._metricTool._SBR is not None

def test_bead_analyzer_runFitting(psf, tmp_path):
    """Unit test for BeadAnalyzer class."""
    psf, _ = psf
    bead_analyzer = BeadAnalyzer()
    bead_analyzer._image = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    bead_analyzer._centroid = np.array([int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)])
    bead_analyzer._roi = [np.array([0, 0, 0])]
    bead_analyzer._outputDir = tmp_path
    bead_analyzer._spacing = [1, 1, 1]
    bead_analyzer.runFitting(outputDir=tmp_path)
    assert bead_analyzer._fitTool is not None

def test_bead_analyzer_generateReport(psf, tmp_path):
    """Unit test for BeadAnalyzer class."""
    psf, _ = psf
    bead_analyzer = BeadAnalyzer()
    bead_analyzer._image = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    bead_analyzer._centroid = np.array([int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)])
    bead_analyzer._roi = [np.array([0, 0, 0])]
    bead_analyzer._outputDir = tmp_path
    bead_analyzer._spacing = [1, 1, 1]
    bead_analyzer.runFitting(outputDir=tmp_path)
    bead_analyzer.runSBRMetric()
    report_path = tmp_path / "bead_report.txt"
    pdf = canvas.Canvas(report_path, pagesize=A4)
    pdf.setTitle("PSF analysis results - Simplified")
    pdf.setFont("Helvetica-Bold", 28)
    pdf.drawCentredString(300, 770, "Analysis Parameters Report - Simplified")
    pdf.setFont("Helvetica", 10)
    bead_analyzer.generatePDFReport(pdf, report_path, [0,0,0], [0,0,0])