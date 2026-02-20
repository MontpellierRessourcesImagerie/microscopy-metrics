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

def test_get_shape():
    """Unit test for the image shape getter"""
    shape = (50,100,100)
    image = create_test_image(shape=shape)
    shape_test = get_shape(image)
    assert list(shape) == list(shape_test)
    shape = (100,100)
    image = create_test_image(shape=shape)
    shape_test = get_shape(image)
    assert list(shape) == list(shape_test)
    shape = (100)
    image = create_test_image(shape=shape)
    shape_test = get_shape(image)
    assert len(shape_test) == 1 and shape == shape_test[0]

def test_um_to_px():
    """Unit test for the um to px converter"""
    assert um_to_px(10,0.0) == 0.0
    axisPhysicalSize = 10.0
    size_um = 100.0
    assert um_to_px(size_um,axisPhysicalSize) == 10.0

def test_px_to_um():
    """Unit test for the um to px converter"""
    axisPhysicalSize = 10.0
    size_px = 1.0
    assert px_to_um(size_px,axisPhysicalSize) == 10.0

def test_is_roi_not_in_rejection():
    """Unit test to know if ROI in rejection zone"""
    centroid = (50,50,50)
    shape = (100,100,100)
    rejection = 10
    assert is_roi_not_in_rejection(centroid,shape,rejection) == True
    centroid = (5,50,50)
    assert is_roi_not_in_rejection(centroid,shape,rejection) == False

def test_is_roi_in_image():
    """Unit test for the roi in image detection"""
    roi = [
        [10,10,10],
        [10,10,20],
        [10,20,10],
        [10,20,20]
    ]
    shape = (20,21,21)
    assert is_roi_in_image(roi,shape) == True
    shape = (20,20,21)
    assert is_roi_in_image(roi,shape) == False
    shape = (20,21,20)
    assert is_roi_in_image(roi,shape) == False

def test_is_roi_overlapped():
    """Unit test for the roi overlapped"""
    roi = np.array([[0, 50, 50], [0, 50, 60], [0, 60, 60], [0, 60, 50]])
    rois = [
        np.array([[0, 10, 10], [0, 10, 20], [0, 20, 20], [0, 20, 10]]),
        np.array([[0, 30, 30], [0, 30, 40], [0, 40, 40], [0, 40, 30]])
    ]

    assert is_roi_overlapped(rois, roi) == False

def test_is_roi_overlapped_with_overlap():
    """Unit test for the roi overlapped"""
    rois = [
        np.array([[0, 10, 10], [0, 10, 20], [0, 20, 20], [0, 20, 10]])
    ]
    roi = np.array([[0, 15, 15], [0, 15, 25], [0, 25, 25], [0, 25, 15]])

    assert is_roi_overlapped(rois, roi)

def create_uniform_image(shape=(10, 10), low=0, high=1):
    """Create an Image with a uniform distribution"""
    return np.random.uniform(low, high, shape)

def create_bimodal_image(shape=(10, 10), low1=0, high1=0.5, low2=0.5, high2=1):
    """Create a bimodale image"""
    image = np.zeros(shape)
    half = shape[0] * shape[1] // 2
    image.flat[:half] = np.random.uniform(low1, high1, half)
    image.flat[half:] = np.random.uniform(low2, high2, half)
    return image

def test_legacy_threshold_uniform():
    """Unit test of legacy threshold on uniform image"""
    image = create_uniform_image(low=0, high=1)
    threshold = legacy_threshold(image)
    assert np.isclose(threshold, 0.5, rtol=0.1)

def test_legacy_threshold_bimodal():
    """Unit test of legacy threshold on bimodal image"""
    image = create_bimodal_image(low1=0, high1=0.4, low2=0.6, high2=1)
    threshold = legacy_threshold(image)
    assert 0.4 < threshold < 0.6
