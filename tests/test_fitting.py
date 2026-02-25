import pytest
import numpy as np
from skimage.draw import disk
from microscopy_metrics.fitting import *

def generate_psf_profil(length=100,amplitude=1.0,center=50.0,sigma=5.0):
    coords = np.linspace(0,length - 1, length)
    psf = amplitude * np.exp(-0.5*((coords - center) / sigma) ** 2)
    return coords,psf

def create_test_image_1d(length=10, centroid=5):
    image = np.zeros(length)
    image[centroid] = 1.0
    return image

def create_test_image_2d(shape=(10, 10), centroid=(5, 5)):
    image = np.zeros(shape)
    image[centroid] = 1.0
    return image

def create_test_image_3d(shape=(10, 10, 10), centroid=(5, 5, 5)):
    image = np.zeros(shape)
    image[centroid] = 1.0
    return image

def test_fwhm_calculation():
    FWHM = round(fwhm(1),2)
    assert FWHM == 2.35

def test_fit_curve_1D():
    """Unit test for the 1D fit curve function with a perfect adjustment"""
    coords_x,psf_x = generate_psf_profil()
    y_lim = (0,1.2)
    params = (1.0,0.0,50.0,5.0)
    popt,pcov,_ = fit_curve_1D(*params,coords_x,psf_x,y_lim)
    assert np.allclose(popt,params,rtol=0.1)

def test_get_cov_matrix_1D():
    image = create_test_image_1d()
    spacing = [1.0]
    centroid = [5]
    cov_matrix = get_cov_matrix(image,spacing,centroid)
    assert cov_matrix.shape == ()
    assert np.isclose(cov_matrix,0.0)

def test_get_cov_matrix_2D():
    image = create_test_image_2d()
    spacing = [1.0,1.0]
    centroid = [5,5]
    cov_matrix = get_cov_matrix(image,spacing,centroid)
    assert cov_matrix.shape == (2,2)
    expected_cov = np.zeros((2,2))
    assert np.allclose(cov_matrix,expected_cov)

def test_get_cov_matrix_3D():
    image = create_test_image_3d()
    spacing = [1.0,1.0,1.0]
    centroid = [5,5,5]
    cov_matrix = get_cov_matrix(image,spacing,centroid)
    assert cov_matrix.shape == (3,3)
    expected_cov = np.zeros((3,3))
    assert np.allclose(cov_matrix,expected_cov)
