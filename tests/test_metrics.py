import pytest
import numpy as np
from skimage.draw import disk
from microscopy_metrics.metrics import Metrics
from microscopy_metrics.fittingTools.fitting3D import Fitting3D
from microscopy_metrics.metricTool.metricTool import MetricTool

def test_signal_to_background_ratio():
    """Unit test for signal to background ratio of a picture"""
    PSF_SIZE = 100
    fitTool = Fitting3D()
    fitTool._show = False
    params = [255,0,PSF_SIZE/2,PSF_SIZE/2,PSF_SIZE/2,PSF_SIZE/10,PSF_SIZE/10,PSF_SIZE/10]
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    psf = fitTool.gauss(*params)(coords)
    image = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 13.0
    metricTool._ringThickness = 2.0
    metricTool._pixelSize = [1, 1, 1]
    metricTool.processSingleSBRRing()
    
    assert metricTool._SBR > 10
