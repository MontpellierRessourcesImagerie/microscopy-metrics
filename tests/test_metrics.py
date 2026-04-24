import pytest
import numpy as np
from skimage.draw import disk
from microscopy_metrics.metrics import Metrics
from microscopy_metrics.fittingTools.fitting3D import Fitting3D
from microscopy_metrics.metricTool.metricTool import MetricTool
PSF_SIZE = 100

@pytest.fixture
def psf():
    fitTool = Fitting3D()
    fitTool._show = False
    params = [255,0,PSF_SIZE/2,PSF_SIZE/2,PSF_SIZE/2,PSF_SIZE/10,PSF_SIZE/10,PSF_SIZE/10]
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    psf = fitTool.gauss(*params)(coords)
    FWHM = [fitTool.fwhm(params[5]), fitTool.fwhm(params[6]), fitTool.fwhm(params[7])]
    yield psf,FWHM

@pytest.fixture
def ellipsPsf():
    fitTool = Fitting3D()
    fitTool._show = False
    params = [255,0,PSF_SIZE/2,PSF_SIZE/2,PSF_SIZE/2,PSF_SIZE/3,PSF_SIZE/10,PSF_SIZE/10]
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    psf = fitTool.gauss(*params)(coords)
    FWHM = [fitTool.fwhm(params[5]), fitTool.fwhm(params[6]), fitTool.fwhm(params[7])]
    yield psf,FWHM

def test_signal_to_background_ratio(psf):
    SIZE = 50
    image = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)
    BACKGROUNVAL = 10
    image += BACKGROUNVAL
    SIGNALVAL = 100
    rr, cc = disk((SIZE // 2, SIZE // 2), SIZE // 4, shape=image.shape[1:])
    for z in range(SIZE):
        image[rr, cc, z] = SIGNALVAL
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 10.0
    metricTool._ringThickness = 5.0
    metricTool._pixelSize = [1, 1, 1]
    metricTool.processSingleSBRRing()
    assert metricTool._SBR == SIGNALVAL / BACKGROUNVAL

def test_signal_to_background_ratio_psf(psf):
    psfData,_ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 13.0
    metricTool._ringThickness = 2.0
    metricTool._pixelSize = [1, 1, 1]
    metricTool.processSingleSBRRing()
    assert metricTool._SBR > 20

def test_LAR_psf(psf):
    psfData,FWHM = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 13.0
    metricTool._ringThickness = 2.0
    metricTool._pixelSize = [1, 1, 1]
    metricTool.lateralAsymmetryRatio(FWHM)
    assert metricTool._LAR == 1

def test_sphericity(psf):
    psfData,_ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 13.0
    metricTool._ringThickness = 2.0
    metricTool._pixelSize = [1, 1, 1]
    metricTool.sphericity()
    assert np.isclose(metricTool._sphericity,1,rtol=0.05)

def test_sphericity_ellipsPsf(ellipsPsf):
    psfData,FWHM = ellipsPsf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 13.0
    metricTool._ringThickness = 2.0
    metricTool._pixelSize = [1, 1, 1]
    metricTool.sphericity()
    assert np.isclose(metricTool._sphericity,0.7,rtol=0.05)

def test_comaticity_perfect_psf(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    metricTool.comaticity()
    assert metricTool._comaticity == 0.0

def test_sphericalAberration_perfect_psf(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    metricTool.sphericalAberration()
    assert np.isclose(metricTool._sphericalAberration, 0.0, atol=0.01)

def test_astigmatism_perfect_psf(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]    
    mu = [PSF_SIZE/2, PSF_SIZE/2, PSF_SIZE/2]
    sigma = [PSF_SIZE/10, PSF_SIZE/10, PSF_SIZE/10]
    metricTool.astigmatism(mu, sigma)
    assert np.isclose(metricTool._astigmatism, 0.0, atol=0.05)