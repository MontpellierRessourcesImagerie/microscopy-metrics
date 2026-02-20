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

def generate_psf_profil(length=100,amplitude=1.0,center=50.0,sigma=5.0):
    """Generate synthetic PSF profile"""
    coords = np.linspace(0,length - 1, length)
    psf = amplitude * np.exp(-0.5*((coords - center) / sigma) ** 2)
    return coords,psf

def test_signal_to_background_ratio():
    """Unit test for signal to background ratio of a picture"""
    shape = (50,100,100)
    bead_positions = [(25,50,50)]
    image = create_test_image(shape=shape,bead_positions=bead_positions)
    mean_SBR,SBR = signal_to_background_ratio(images=[image])
    assert len(SBR) == 1 and mean_SBR == np.inf

def test_signal_to_background_ratio_annulus():
    """Unit test for signal to background ratio using an annulus of a picture"""
    shape = (50,100,100)
    bead_positions = [(25,50,50)]
    image = create_test_image(shape=shape,bead_positions=bead_positions)
    mean_SBR,SBR = signal_to_background_ratio_annulus(images=[image],inner_annulus_distance=2,annulus_thickness=5)
    assert len(SBR) == 1 and mean_SBR == np.inf

def test_uncertainty():
    """Unit test for uncertainty calculation"""
    shape = (4,4)
    pcov = np.ones(shape=shape)
    uncert = uncertainty(pcov)
    assert uncert[3] == 1

def test_determination_perfect():
    """Unit test for determination calculation"""
    coords,psf = generate_psf_profil()
    params = (1.0,0.0,50.0,5.0)
    r_squared = determination(params,coords,psf)
    assert np.isclose(r_squared,1.0)

def test_determination_bad():
    """Unit test for bad determination calculation"""
    coords,psf = generate_psf_profil()
    params = (1.0,0.0,30.0,10.0)
    r_squared = determination(params,coords,psf)
    assert r_squared < 0.9