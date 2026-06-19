import pytest
import os
import numpy as np
from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.fittingTools.fitting1D import Fitting1D
from microscopy_metrics.fittingTools.fitting2D import Fitting2D
from microscopy_metrics.fittingTools.fitting3D import Fitting3D
from microscopy_metrics.fittingTools.fitting2DEllips import Fitting2DEllips
from microscopy_metrics.fittingTools.fitting2DRotation import Fitting2DRotation
from microscopy_metrics.fittingTools.fitting3DRotation import Fitting3DRotation
from microscopy_metrics.fittingTools.prominence import Prominence
from microscopy_metrics.scripts.PSFGenerator.PSF import PSFGenerator, PSFWithAstigmatismAberration, PSFWithSphericalAberration, PSFWithComaticAberration, PSFRandomParameter

PSF_SIZE = 100


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

@pytest.fixture
def comaticPsf():
    psf = PSFWithComaticAberration(PSF_SIZE, Intensity=0.08).psf
    return psf

def test_fwhm_calculation():
    FWHM = round(FittingTool().fwhm(1), 2)
    assert FWHM == 2.35

def test_contrast_calculation():
    fitTool = FittingTool()
    fitTool.parameters = [255, 0, PSF_SIZE / 2, PSF_SIZE / 2, PSF_SIZE / 2, PSF_SIZE / 10, PSF_SIZE / 10, PSF_SIZE / 10]
    fitTool.computeContrast()
    assert fitTool.contrast == 255.0

def test_computeContrast():
    fitTool = FittingTool()
    fitTool.parameters = [255, 1, PSF_SIZE / 2, PSF_SIZE / 2, PSF_SIZE / 2, PSF_SIZE / 10, PSF_SIZE / 10, PSF_SIZE / 10]
    fitTool.computeContrast()
    assert fitTool.contrast == 254.0

def test_getMu():
    fitTool = FittingTool()
    fitTool.parameters = [255, 0, PSF_SIZE / 2, PSF_SIZE / 2, PSF_SIZE / 2, PSF_SIZE / 10, PSF_SIZE / 10, PSF_SIZE / 10]
    mu = fitTool.getMu()
    assert mu[0] == PSF_SIZE / 2
    assert mu[1] == PSF_SIZE / 2
    assert mu[2] == PSF_SIZE / 2

def test_getSigma():
    fitTool = FittingTool()
    fitTool.parameters = [255, 0, PSF_SIZE / 2, PSF_SIZE / 2, PSF_SIZE / 2, PSF_SIZE / 10, PSF_SIZE / 10, PSF_SIZE / 10]
    sigma = fitTool.getSigma()
    assert sigma[0] == PSF_SIZE / 10
    assert sigma[1] == PSF_SIZE / 10
    assert sigma[2] == PSF_SIZE / 10

def test_mip_calculation(psf):
    psf, FWHM = psf
    fitTool = FittingTool()
    fitTool._image = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    mip_z = fitTool.mip3d(fitTool._image, axis=0)
    mip_y = fitTool.mip3d(fitTool._image, axis=1)
    mip_x = fitTool.mip3d(fitTool._image, axis=2)
    assert mip_z.shape == (PSF_SIZE, PSF_SIZE)
    assert mip_y.shape == (PSF_SIZE, PSF_SIZE)
    assert mip_x.shape == (PSF_SIZE, PSF_SIZE)

def test_normalization(psf):
    psf, FWHM = psf
    fitTool = FittingTool()
    fitTool._image = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    normalized_image = fitTool.setNormalizedImage()
    assert np.isclose(np.max(normalized_image), 1.0)
    assert np.isclose(np.min(normalized_image), 0.0)


def test_1D_Fitting(psf):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool1D = Fitting1D()
    fitTool1D._image = psfReshape
    fitTool1D._show = False
    fitTool1D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool1D._roi = [np.array([0, 0, 0])]
    fitTool1D.processSingleFit(0)
    assert np.isclose(fitTool1D.parameters[0], 255, rtol=20)
    assert np.isclose(fitTool1D.parameters[1], 0, rtol=20)
    mu = [fitTool1D.parameters[2], fitTool1D.parameters[3], fitTool1D.parameters[4]]
    assert (
        np.isclose(mu[0], PSF_SIZE / 2, rtol=10)
        and np.isclose(mu[1], PSF_SIZE / 2, rtol=10)
        and np.isclose(mu[2], PSF_SIZE / 2, rtol=10)
    )
    sigma = [fitTool1D.parameters[5], fitTool1D.parameters[6], fitTool1D.parameters[7]]
    assert (
        np.isclose(sigma[0], PSF_SIZE / 10, rtol=10)
        and np.isclose(sigma[1], PSF_SIZE / 10, rtol=10)
        and np.isclose(sigma[2], PSF_SIZE / 10, rtol=10)
    )
    fwhms = fitTool1D.fwhms
    assert (
        np.isclose(FWHM[0], fwhms[0], rtol=1)
        and np.isclose(FWHM[1], fwhms[1], rtol=1)
        and np.isclose(FWHM[2], fwhms[2], rtol=1)
    )

def test_1D_Plotting(psf, tmp_path):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool1D = Fitting1D()
    fitTool1D._image = psfReshape
    fitTool1D._show = True
    fitTool1D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool1D._roi = [np.array([0, 0, 0])]
    fitTool1D.processSingleFit(0)
    fitTool1D.plotFit(tmp_path )
    assert os.path.exists(tmp_path / "fit_curve_1D_Z.png")


def test_2D_Fitting(psf):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool2D = Fitting2D()
    fitTool2D._image = psfReshape
    fitTool2D._show = False
    fitTool2D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool2D._roi = [np.array([0, 0, 0])]
    fitTool2D.processSingleFit(0)
    assert np.isclose(fitTool2D.parameters[0], 255, rtol=20)
    assert np.isclose(fitTool2D.parameters[1], 0, rtol=20)
    mu = [fitTool2D.parameters[2], fitTool2D.parameters[3], fitTool2D.parameters[4]]
    assert (
        np.isclose(mu[0], PSF_SIZE / 2, rtol=10)
        and np.isclose(mu[1], PSF_SIZE / 2, rtol=10)
        and np.isclose(mu[2], PSF_SIZE / 2, rtol=10)
    )
    sigma = [fitTool2D.parameters[5], fitTool2D.parameters[6], fitTool2D.parameters[7]]
    assert (
        np.isclose(sigma[0], PSF_SIZE / 10, rtol=10)
        and np.isclose(sigma[1], PSF_SIZE / 10, rtol=10)
        and np.isclose(sigma[2], PSF_SIZE / 10, rtol=10)
    )
    fwhms = fitTool2D.fwhms
    assert (
        np.isclose(FWHM[0], fwhms[0], rtol=1)
        and np.isclose(FWHM[1], fwhms[1], rtol=1)
        and np.isclose(FWHM[2], fwhms[2], rtol=1)
    )

def test_2D_Plotting(psf, tmp_path):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool2D = Fitting2D()
    fitTool2D._image = psfReshape
    fitTool2D._show = False
    fitTool2D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool2D._roi = [np.array([0, 0, 0])]
    fitTool2D.processSingleFit(0)
    fitTool2D.plotFit(tmp_path)
    assert os.path.exists(tmp_path / "fit_curve_1D_Z.png")

def test_2D_showing(psf, tmp_path):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool2D = Fitting2D()
    fitTool2D._image = psfReshape
    fitTool2D._show = False
    fitTool2D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool2D._roi = [np.array([0, 0, 0])]
    fitTool2D.processSingleFit(0)
    fitTool2D.showFit(tmp_path)
    assert os.path.exists(tmp_path / "2D_Gaussian_Image_ZY.png")

def test_3D_Fitting(psf):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool3D = Fitting3D()
    fitTool3D._image = psfReshape
    fitTool3D._show = False
    fitTool3D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool3D._roi = [np.array([0, 0, 0])]
    fitTool3D.processSingleFit(0)
    assert np.isclose(fitTool3D.parameters[0], 255, rtol=20)
    assert np.isclose(fitTool3D.parameters[1], 0, rtol=20)
    mu = [fitTool3D.parameters[2], fitTool3D.parameters[3], fitTool3D.parameters[4]]
    assert (
        np.isclose(mu[0], PSF_SIZE / 2, rtol=10)
        and np.isclose(mu[1], PSF_SIZE / 2, rtol=10)
        and np.isclose(mu[2], PSF_SIZE / 2, rtol=10)
    )
    sigma = [fitTool3D.parameters[5], fitTool3D.parameters[6], fitTool3D.parameters[7]]
    assert (
        np.isclose(sigma[0], PSF_SIZE / 10, rtol=10)
        and np.isclose(sigma[1], PSF_SIZE / 10, rtol=10)
        and np.isclose(sigma[2], PSF_SIZE / 10, rtol=10)
    )
    fwhms = fitTool3D.fwhms
    assert (
        np.isclose(FWHM[0], fwhms[0], rtol=1)
        and np.isclose(FWHM[1], fwhms[1], rtol=1)
        and np.isclose(FWHM[2], fwhms[2], rtol=1)
    )

def test_3D_Plotting(psf, tmp_path):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool3D = Fitting3D()
    fitTool3D._image = psfReshape
    fitTool3D._show = False
    fitTool3D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool3D._roi = [np.array([0, 0, 0])]
    fitTool3D.processSingleFit(0)
    fitTool3D.plotFit(tmp_path)
    assert os.path.exists(tmp_path / "fit_curve_1D_Z.png")

def test_3D_showing(psf, tmp_path):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool3D = Fitting3D()
    fitTool3D._image = psfReshape
    fitTool3D._show = False
    fitTool3D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool3D._roi = [np.array([0, 0, 0])]
    fitTool3D.processSingleFit(0)
    fitTool3D.showFit(tmp_path)
    assert os.path.exists(tmp_path / "2D_Gaussian_Image_ZY.png")

def test_2D_Rotation_Fitting(psf):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool2DRotation = Fitting2DRotation()
    fitTool2DRotation._image = psfReshape
    fitTool2DRotation._show = False
    fitTool2DRotation._centroid = [
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
    ]
    fitTool2DRotation._roi = [np.array([0, 0, 0])]
    fitTool2DRotation.processSingleFit(0)
    assert np.isclose(fitTool2DRotation.parameters[0], 255, rtol=20)
    assert np.isclose(fitTool2DRotation.parameters[1], 0, rtol=20)
    mu = [
        fitTool2DRotation.parameters[2],
        fitTool2DRotation.parameters[3],
        fitTool2DRotation.parameters[4],
    ]
    assert (
        np.isclose(mu[0], PSF_SIZE / 2, rtol=10)
        and np.isclose(mu[1], PSF_SIZE / 2, rtol=10)
        and np.isclose(mu[2], PSF_SIZE / 2, rtol=10)
    )
    sigma = [
        fitTool2DRotation.parameters[5],
        fitTool2DRotation.parameters[6],
        fitTool2DRotation.parameters[7],
    ]
    assert (
        np.isclose(sigma[0], PSF_SIZE / 10, rtol=10)
        and np.isclose(sigma[1], PSF_SIZE / 10, rtol=10)
        and np.isclose(sigma[2], PSF_SIZE / 10, rtol=10)
    )
    fwhms = fitTool2DRotation.fwhms
    assert (
        np.isclose(FWHM[0], fwhms[0], rtol=1)
        and np.isclose(FWHM[1], fwhms[1], rtol=1)
        and np.isclose(FWHM[2], fwhms[2], rtol=1)
    )

def test_2D_Rotation_Plotting(psf, tmp_path):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool2DRotation = Fitting2DRotation()
    fitTool2DRotation._image = psfReshape
    fitTool2DRotation._show = False
    fitTool2DRotation._centroid = [
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
    ]
    fitTool2DRotation._roi = [np.array([0, 0, 0])]
    fitTool2DRotation.processSingleFit(0)
    fitTool2DRotation.plotFit(tmp_path)
    assert os.path.exists(tmp_path / "fit_curve_1D_Z.png")

def test_2D_Rotation_showing(psf, tmp_path):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool2DRotation = Fitting2DRotation()
    fitTool2DRotation._image = psfReshape
    fitTool2DRotation._show = False
    fitTool2DRotation._centroid = [
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
    ]
    fitTool2DRotation._roi = [np.array([0, 0, 0])]
    fitTool2DRotation.processSingleFit(0)
    fitTool2DRotation.showFit(tmp_path)
    assert os.path.exists(tmp_path / "2D_Gaussian_Image_ZY.png")


def test_3D_Rotation_Fitting(psf):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool3DRotation = Fitting3DRotation()
    fitTool3DRotation._image = psfReshape
    fitTool3DRotation._show = False
    fitTool3DRotation._centroid = [
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
    ]
    fitTool3DRotation._roi = [np.array([0, 0, 0])]
    fitTool3DRotation.processSingleFit(0)
    assert np.isclose(fitTool3DRotation.parameters[0], 255, rtol=20)
    assert np.isclose(fitTool3DRotation.parameters[1], 0, rtol=20)
    mu = [
        fitTool3DRotation.parameters[2],
        fitTool3DRotation.parameters[3],
        fitTool3DRotation.parameters[4],
    ]
    assert (
        np.isclose(mu[0], PSF_SIZE / 2, rtol=10)
        and np.isclose(mu[1], PSF_SIZE / 2, rtol=10)
        and np.isclose(mu[2], PSF_SIZE / 2, rtol=10)
    )
    sigma = [
        fitTool3DRotation.parameters[5],
        fitTool3DRotation.parameters[6],
        fitTool3DRotation.parameters[7],
    ]
    assert (
        np.isclose(sigma[0], PSF_SIZE / 10, rtol=10)
        and np.isclose(sigma[1], PSF_SIZE / 10, rtol=10)
        and np.isclose(sigma[2], PSF_SIZE / 10, rtol=10)
    )
    fwhms = fitTool3DRotation.fwhms
    assert (
        np.isclose(FWHM[0], fwhms[0], rtol=1)
        and np.isclose(FWHM[1], fwhms[1], rtol=1)
        and np.isclose(FWHM[2], fwhms[2], rtol=1)
    )

def test_3D_Rotation_Plotting(psf, tmp_path):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool3DRotation = Fitting3DRotation()
    fitTool3DRotation._image = psfReshape
    fitTool3DRotation._show = False
    fitTool3DRotation._centroid = [
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
    ]
    fitTool3DRotation._roi = [np.array([0, 0, 0])]
    fitTool3DRotation.processSingleFit(0)
    fitTool3DRotation.plotFit(tmp_path)
    assert os.path.exists(tmp_path / "fit_curve_1D_Z.png")

def test_3D_Rotation_showing(psf, tmp_path):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool3DRotation = Fitting3DRotation()
    fitTool3DRotation._image = psfReshape
    fitTool3DRotation._show = False
    fitTool3DRotation._centroid = [
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
    ]
    fitTool3DRotation._roi = [np.array([0, 0, 0])]
    fitTool3DRotation.processSingleFit(0)
    fitTool3DRotation.showFit(tmp_path)
    assert os.path.exists(tmp_path / "2D_Gaussian_Image_ZY.png")


def test_2D_Ellips_Fitting(psf):
    psf, FWHM = psf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool2DEllips = Fitting2DEllips()
    fitTool2DEllips._image = psfReshape
    fitTool2DEllips._show = False
    fitTool2DEllips._centroid = [
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
    ]
    fitTool2DEllips._roi = [np.array([0, 0, 0])]
    fitTool2DEllips.processSingleFit(0)
    assert np.isclose(fitTool2DEllips.parameters[0], 255, rtol=20)
    assert np.isclose(fitTool2DEllips.parameters[1], 0, rtol=20)
    mu = [
        fitTool2DEllips.parameters[2],
        fitTool2DEllips.parameters[3],
        fitTool2DEllips.parameters[4],
    ]
    assert (
        np.isclose(mu[0], PSF_SIZE / 2, rtol=10)
        and np.isclose(mu[1], PSF_SIZE / 2, rtol=10)
        and np.isclose(mu[2], PSF_SIZE / 2, rtol=10)
    )

def test_2D_Ellips_Plotting(comaticPsf, tmp_path):
    psf = comaticPsf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool2DEllips = Fitting2DEllips()
    fitTool2DEllips._image = psfReshape
    fitTool2DEllips._show = False
    fitTool2DEllips._centroid = [
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
    ]
    fitTool2DEllips._roi = [np.array([0, 0, 0])]
    fitTool2DEllips.processSingleFit(0)
    fitTool2DEllips.plotFit(tmp_path)
    assert os.path.exists(tmp_path / "fit_curve_1D_Z.png")

def test_2D_Ellips_showing(comaticPsf, tmp_path):
    psf = comaticPsf
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool2DEllips = Fitting2DEllips()
    fitTool2DEllips._image = psfReshape
    fitTool2DEllips._show = False
    fitTool2DEllips._centroid = [
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
        int(PSF_SIZE / 2),
    ]
    fitTool2DEllips._roi = [np.array([0, 0, 0])]
    fitTool2DEllips.processSingleFit(0)
    fitTool2DEllips.showFit(tmp_path)
    assert os.path.exists(tmp_path / "2D_Gaussian_Image_ZY.png")


def test_prominence_calculation(psf):
    psf, FWHM = psf
    fitTool = Prominence()
    fitTool._image = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool._roi = [np.array([0, 0, 0])]
    fitTool.processSingleFit(0)
    assert np.isclose(fitTool.parameters[0], 255, rtol=20)
    assert np.isclose(fitTool.parameters[1], 0, atol=1)
    assert np.isclose(fitTool.parameters[2], PSF_SIZE / 2, rtol=10)
    assert np.isclose(fitTool.parameters[3], PSF_SIZE / 2, rtol=10)
    assert np.isclose(fitTool.parameters[4], PSF_SIZE / 2, rtol=10)
    assert np.isclose(fitTool.parameters[5], PSF_SIZE / 10, rtol=10)
    assert np.isclose(fitTool.parameters[6], PSF_SIZE / 10, rtol=10)
    assert np.isclose(fitTool.parameters[7], PSF_SIZE / 10, rtol=10)

def test_prominence_plotting(psf, tmp_path):
    psf, FWHM = psf
    fitTool = Prominence()
    fitTool._image = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    fitTool._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool._roi = [np.array([0, 0, 0])]
    fitTool.processSingleFit(0)
    fitTool.plotFit(tmp_path /"fit_curve_1D_Z.png")
    assert os.path.exists(tmp_path / "fit_curve_1D_Z.png")


