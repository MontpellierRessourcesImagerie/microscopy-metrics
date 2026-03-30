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
    FWHM = round(FittingTool().fwhm(1),2)
    assert FWHM == 2.35

def test_1D_Fitting():
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
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    psfReshapeTest = psfReshape
    FWHM = [fitTool.fwhm(params[5]), fitTool.fwhm(params[6]), fitTool.fwhm(params[7])]
    fitTool1D = Fitting1D()
    fitTool1D._image = psfReshape
    fitTool1D._show = False
    fitTool1D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool1D._roi = [np.array([0, 0, 0])]
    result = fitTool1D.processSingleFit(0)
    amp = (result[4][0][0] + result[4][1][0] + result[4][2][0]) / 3.0
    assert np.isclose(amp, 255, rtol=20)
    bg = (result[4][0][1] + result[4][1][1] + result[4][2][1]) / 3.0
    assert np.isclose(bg,0, rtol=20)
    mu = [result[4][0][2], result[4][1][2], result[4][2][2]]
    assert np.isclose(mu[0],PSF_SIZE/2,rtol=10) and np.isclose(mu[1],PSF_SIZE/2,rtol=10) and np.isclose(mu[2],PSF_SIZE/2,rtol=10)
    sigma = [result[4][0][3], result[4][1][3], result[4][2][3]]
    assert np.isclose(sigma[0],PSF_SIZE/10,rtol=10) and np.isclose(sigma[1],PSF_SIZE/10,rtol=10) and np.isclose(sigma[2],PSF_SIZE/10,rtol=10)
    fwhms = [fitTool1D.fwhm(sigma[0]), fitTool1D.fwhm(sigma[1]), fitTool1D.fwhm(sigma[2])]
    assert np.isclose(FWHM[0],fwhms[0],rtol=1) and np.isclose(FWHM[1],fwhms[1],rtol=1) and np.isclose(FWHM[2],fwhms[2],rtol=1)

def test_2D_Fitting():
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
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    psfReshapeTest = psfReshape
    FWHM = [fitTool.fwhm(params[5]), fitTool.fwhm(params[6]), fitTool.fwhm(params[7])]
    fitTool2D = Fitting2D()
    fitTool2D._image = psfReshape
    fitTool2D._show = False
    fitTool2D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool2D._roi = [np.array([0, 0, 0])]
    result = fitTool2D.processSingleFit(0)
    amp = result[4][0]
    assert np.isclose(amp, 255, rtol=20)
    bg = result[4][1]
    assert np.isclose(bg,0, rtol=20)
    mu = [result[4][2], result[4][3], result[4][4]]
    assert np.isclose(mu[0],PSF_SIZE/2,rtol=10) and np.isclose(mu[1],PSF_SIZE/2,rtol=10) and np.isclose(mu[2],PSF_SIZE/2,rtol=10)
    sigma = [result[4][5], result[4][6], result[4][7]]
    assert np.isclose(sigma[0],PSF_SIZE/10,rtol=10) and np.isclose(sigma[1],PSF_SIZE/10,rtol=10) and np.isclose(sigma[2],PSF_SIZE/10,rtol=10)
    fwhms = [fitTool2D.fwhm(sigma[0]), fitTool2D.fwhm(sigma[1]), fitTool2D.fwhm(sigma[2])]
    assert np.isclose(FWHM[0],fwhms[0],rtol=1) and np.isclose(FWHM[1],fwhms[1],rtol=1) and np.isclose(FWHM[2],fwhms[2],rtol=1)

def test_3D_Fitting():
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
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    psfReshapeTest = psfReshape
    FWHM = [fitTool.fwhm(params[5]), fitTool.fwhm(params[6]), fitTool.fwhm(params[7])]
    fitTool3D = Fitting3D()
    fitTool3D._image = psfReshape
    fitTool3D._show = False
    fitTool3D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool3D._roi = [np.array([0, 0, 0])]
    result = fitTool3D.processSingleFit(0)
    amp = result[4][0]
    assert np.isclose(amp, 255, rtol=20)
    bg = result[4][1]
    assert np.isclose(bg,0, rtol=20)
    mu = [result[4][2], result[4][3], result[4][4]]
    assert np.isclose(mu[0],PSF_SIZE/2,rtol=10) and np.isclose(mu[1],PSF_SIZE/2,rtol=10) and np.isclose(mu[2],PSF_SIZE/2,rtol=10)
    sigma = [result[4][5], result[4][6], result[4][7]]
    assert np.isclose(sigma[0],PSF_SIZE/10,rtol=10) and np.isclose(sigma[1],PSF_SIZE/10,rtol=10) and np.isclose(sigma[2],PSF_SIZE/10,rtol=10)
    fwhms = [fitTool3D.fwhm(sigma[0]), fitTool3D.fwhm(sigma[1]), fitTool3D.fwhm(sigma[2])]
    assert np.isclose(FWHM[0],fwhms[0],rtol=1) and np.isclose(FWHM[1],fwhms[1],rtol=1) and np.isclose(FWHM[2],fwhms[2],rtol=1)