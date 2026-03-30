import pytest
import numpy as np
from skimage.draw import disk
from microscopy_metrics.utils import *

def create_test_image(shape):
    """Create a test image containing beads with known positions"""
    image = np.zeros(shape,dtype=np.uint8)
    return image

def generate_psf_profil(length=100,amplitude=1.0,center=50.0,sigma=5.0):
    """Generate synthetic PSF profile"""
    coords = np.linspace(0,length - 1, length)
    psf = amplitude * np.exp(-0.5*((coords - center) / sigma) ** 2)
    return coords,psf

def test_um_to_px():
    """Unit test for the um to px converter"""
    assert umToPx(10,0.0) == 0.0
    axisPhysicalSize = 10.0
    size_um = 100.0
    assert umToPx(size_um,axisPhysicalSize) == 10.0

def test_px_to_um():
    """Unit test for the um to px converter"""
    axisPhysicalSize = 10.0
    size_px = 1.0
    assert pxToUm(size_px,axisPhysicalSize) == 10.0


