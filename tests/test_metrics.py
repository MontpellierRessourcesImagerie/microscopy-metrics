import os

import pytest
import numpy as np
from skimage.draw import disk
from microscopy_metrics.fittingTools.fitting3D import Fitting3D
from microscopy_metrics.metricTool.metricTool import MetricTool
from microscopy_metrics.metricTool.meshTool import MeshBuilder
from microscopy_metrics.scripts.PSFGenerator.PSF import PSFGenerator, PSFWithAstigmatismAberration, PSFWithSphericalAberration, PSFWithComaticAberration


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
def ellipsPsf():
    fitTool = Fitting3D()
    fitTool._show = False
    params = [
        255,
        0,
        PSF_SIZE / 2,
        PSF_SIZE / 2,
        PSF_SIZE / 2,
        PSF_SIZE / 3,
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


@pytest.fixture
def sphericalAberrationPsf():
    psf = PSFWithSphericalAberration(PSF_SIZE).psf
    return psf


@pytest.fixture
def astigmatismPsf():
    psf = PSFWithAstigmatismAberration(PSF_SIZE).psf
    return psf

def test_normalizedImage():
    SIZE = 50
    image = np.random.rand(SIZE, SIZE, SIZE).astype(np.float32)
    metricTool = MetricTool()
    metricTool._image = image
    normalizedImage = metricTool.setNormalizedImage(image)
    assert np.isclose(np.min(normalizedImage), 0.0)
    assert np.isclose(np.max(normalizedImage), 1.0)

def test_normalizedImage_with_zero_image():
    SIZE = 50
    image = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)
    metricTool = MetricTool()
    metricTool._image = image
    normalizedImage = metricTool.setNormalizedImage(image)
    assert np.isclose(np.min(normalizedImage), 0.0)
    assert np.isclose(np.max(normalizedImage), 0.0)

def test_normalizedImage_with_negative_image():
    SIZE = 50
    image = np.random.rand(SIZE, SIZE, SIZE).astype(np.float32) - 0.5
    metricTool = MetricTool()
    metricTool._image = image
    normalizedImage = metricTool.setNormalizedImage(image)
    assert np.isclose(np.min(normalizedImage), 0.0)
    assert np.isclose(np.max(normalizedImage), 1.0)

def test_normalizedImage_with_1D():
    SIZE = 50
    image = np.random.rand(SIZE).astype(np.float32)
    metricTool = MetricTool()
    metricTool._image = image
    with pytest.raises(ValueError, match="Image have to be in 2D or 3D."):
        metricTool.setNormalizedImage(image)


def test_signal_to_background_ratio():
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
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 13.0
    metricTool._ringThickness = 2.0
    metricTool._pixelSize = [1, 1, 1]
    metricTool.processSingleSBRRing()
    assert metricTool._SBR > 20

def test_signal_to_background_ratio_1D():
    SIZE = 50
    image = np.random.rand(SIZE).astype(np.float32)
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 10.0
    metricTool._ringThickness = 5.0
    metricTool._pixelSize = [1, 1, 1]
    assert metricTool.processSingleSBRRing() == -1
    
def test_signal_to_background_ratio_None():
    metricTool = MetricTool()
    metricTool._image = None
    metricTool._ringInnerDistance = 10.0
    metricTool._ringThickness = 5.0
    metricTool._pixelSize = [1, 1, 1]
    assert metricTool.processSingleSBRRing() == -1

def test_signal_to_background_ratio_uniform():
    SIZE = 50
    image = np.ones((SIZE, SIZE, SIZE), dtype=np.float32) * 100
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 10.0
    metricTool._ringThickness = 5.0
    metricTool._pixelSize = [1, 1, 1]
    assert metricTool.processSingleSBRRing() == -1

def test_LAR_psf(psf):
    psfData, FWHM = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 13.0
    metricTool._ringThickness = 2.0
    metricTool._pixelSize = [1, 1, 1]
    metricTool.lateralAsymmetryRatio(FWHM)
    assert metricTool._LAR == 1

def test_LAR_no_FWHM(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 13.0
    metricTool._ringThickness = 2.0
    metricTool._pixelSize = [1, 1, 1]
    with pytest.raises(ValueError, match="FWHM values are not available or insufficient to calculate LAR."):
        metricTool.lateralAsymmetryRatio([])


def test_sphericity(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 13.0
    metricTool._ringThickness = 2.0
    metricTool._pixelSize = [1, 1, 1]
    metricTool.sphericity()
    assert np.isclose(metricTool._sphericity, 1, rtol=0.05)


def test_sphericity_ellipsPsf(ellipsPsf):
    psfData, _ = ellipsPsf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._ringInnerDistance = 13.0
    metricTool._ringThickness = 2.0
    metricTool._pixelSize = [1, 1, 1]
    metricTool.sphericity()
    assert np.isclose(metricTool._sphericity, 0.7, rtol=0.05)

def test_getContours(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    contours = metricTool.getContours(metricTool._image[:,:, PSF_SIZE // 2])
    assert len(contours) > 0

def test_comaticity_perfect_psf(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    metricTool.comaticity()
    assert metricTool._comaticity == 0.0


def test_comaticity_comaticPsf(comaticPsf):
    image = comaticPsf
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [0.05, 0.05, 0.05]
    metricTool.comaticity()
    assert metricTool._comaticity > 0.05

def test_comaticity_comaticPsfReversed(comaticPsf):
    image = comaticPsf
    image = np.flip(image, axis=0)
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [0.05, 0.05, 0.05]
    metricTool.comaticity()
    assert metricTool._comaticity > 0.05


def test_sphericalAberration_perfect_psf(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    metricTool.sphericalAberration()
    assert np.isclose(metricTool._sphericalAberration, 0.0, atol=0.01)


def test_sphericalAberration_sphericalAberrationPsf(sphericalAberrationPsf):
    image = sphericalAberrationPsf
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [0.05, 0.05, 0.05]
    metricTool.sphericalAberration()
    assert metricTool._sphericalAberration > 0.05


def test_astigmatism_perfect_psf(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    mu = [PSF_SIZE / 2, PSF_SIZE / 2, PSF_SIZE / 2]
    sigma = [PSF_SIZE / 10, PSF_SIZE / 10, PSF_SIZE / 10]
    metricTool.astigmatism(mu, sigma)
    assert np.isclose(metricTool._astigmatism, 0.0, atol=0.05)


def test_astigmatism_astigmatismPsf(astigmatismPsf):
    image = astigmatismPsf
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [0.05, 0.05, 0.05]
    mu = [PSF_SIZE / 2, PSF_SIZE / 2, PSF_SIZE / 2]
    sigma = [PSF_SIZE / 10, PSF_SIZE / 10, PSF_SIZE / 10]
    metricTool.astigmatism(mu, sigma)
    assert metricTool._astigmatism > 0.05


def test_ellips_perfect_psf(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    metricTool.ellipsRatio()
    assert np.isclose(metricTool._ellipsRatio, 1.0, atol=0.05)


def test_ellips_ellipsPsf(ellipsPsf):
    psfData, _ = ellipsPsf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1.0, 1.0, 1.0]
    metricTool.ellipsRatio()
    assert metricTool._ellipsRatio > 0.05
    assert metricTool._orientation > 0.0

def test_ellips_no_signal():
    SIZE = 50
    image = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1.0, 1.0, 1.0]
    assert metricTool.ellipsRatio() == 0.0


def test_MeshBuilder_raises_if_image_not_set():
    meshBuilder = MeshBuilder()
    with pytest.raises(ValueError, match="Image is not set"):
        meshBuilder.BuildMesh()

def test_MeshBuilder_raises_if_no_regions_found():
    metricTool = MetricTool()
    metricTool._image = np.zeros((PSF_SIZE, PSF_SIZE, PSF_SIZE), dtype=np.float32)
    with pytest.raises(ValueError, match="No regions found in the image"):
        metricTool.meshMetrics()

def test_concavity_perfect_psf(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    metricTool.meshMetrics()
    concavity = metricTool.meshBuilder._concavity
    assert np.isclose(concavity, 0.0, atol=0.15)

def test_saveMeshBuilder(psf, tmp_path):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    metricTool.meshMetrics()
    meshBuilder = metricTool.meshBuilder
    meshBuilder.saveMesh(tmp_path / "test_mesh.obj")
    assert os.path.exists(tmp_path / "test_mesh.obj")

def test_saveMesh_raises_if_mesh_not_built(psf, tmp_path):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    metricTool.meshMetrics()
    meshBuilder = metricTool.meshBuilder
    meshBuilder._verticesResized = None
    meshBuilder._faces = None

    with pytest.raises(ValueError, match="Mesh has not been built"):
        meshBuilder.saveMesh(tmp_path / "dummy.obj")

def test_concavity_raises_if_mesh_not_built(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    metricTool.meshMetrics()
    meshBuilder = metricTool.meshBuilder
    meshBuilder._verticesResized = None
    meshBuilder._faces = None

    with pytest.raises(ValueError, match="Mesh has not been built"):
        meshBuilder.concavity()

def test_curvature_raises_if_mesh_not_built(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    metricTool.meshMetrics()
    meshBuilder = metricTool.meshBuilder
    meshBuilder._verticesResized = None
    meshBuilder._faces = None

    with pytest.raises(ValueError, match="Mesh has not been built"):
        meshBuilder.curvature()


def test_concavity_comaticPsf(comaticPsf):
    image = comaticPsf
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [0.05, 0.05, 0.05]
    metricTool.meshMetrics()
    concavity = metricTool.meshBuilder._concavity
    assert concavity > 0.15


def test_skeletonizePath_perfect_psf(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    metricTool.meshMetrics()
    metricTool.skeletonizePath()
    assert len(metricTool._pathSkeleton.distances) > 0

def test_skeletonizePath_noMeshBuilder(psf):
    psfData, _ = psf
    image = psfData.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    metricTool.skeletonizePath()
    assert len(metricTool._pathSkeleton.distances) > 0

def test_skeletonizePath_noSkeleton():
    SIZE = 50
    image = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [1, 1, 1]
    assert metricTool.skeletonizePath() == None

def test_skeletonizePath_comaticPsf(comaticPsf):
    image = comaticPsf
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._pixelSize = [0.05, 0.05, 0.05]
    metricTool.meshMetrics()
    metricTool.skeletonizePath()
    assert len(metricTool._pathSkeleton.distances) > 0

def test_generateBeadOrientation(tmp_path):
    SIZE = 50
    image = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._orientation = 45.0
    metricTool._pixelSize = [1.0, 1.0, 1.0]
    metricTool.generateBeadOrientation(tmp_path)
    assert os.path.exists(tmp_path / "bead_orientation.png")

def test_generateBeadOrientation_noOrientation(tmp_path):
    SIZE = 50
    image = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)
    metricTool = MetricTool()
    metricTool._image = image
    metricTool._orientation = None
    metricTool._pixelSize = [1.0, 1.0, 1.0]
    assert metricTool.generateBeadOrientation(tmp_path / "bead_orientation.png") == None
