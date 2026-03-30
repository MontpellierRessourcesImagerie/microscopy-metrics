import pytest
import numpy as np
from skimage.draw import disk
from microscopy_metrics.metrics import *

def create_test_image(shape,bead_positions,radius=3,intensity=255):
    """Create a test image containing beads with known positions"""
    image = np.zeros(shape,dtype=np.uint8)
    for z,y,x in bead_positions:
        rr,cc = disk((x,y), radius,shape=shape[1:])
        image[z,cc,rr] = intensity
    return image

def test_signal_to_background_ratio():
    """Unit test for signal to background ratio of a picture"""
    shape = (50,100,100)
    bead_positions = [(25,50,50)]
    metrics = Metrics()
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
    metrics._images = image
    metrics._ringInnerDistance = 13.0
    SBR = metrics.processSingleSBRRing(0,image)
    assert SBR > 10
