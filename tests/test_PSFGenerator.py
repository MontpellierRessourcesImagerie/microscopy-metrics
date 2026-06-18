import pytest
import numpy as np
from microscopy_metrics.scripts.PSFGenerator.PSF import PSFGenerator, PSFWithAstigmatismAberration, PSFWithSphericalAberration, PSFWithComaticAberration, PSFRandomParameter

PSF_SIZE = 100

def test_PSFGenerator():
    psf = PSFGenerator(PSF_SIZE)
    assert psf.size == PSF_SIZE

def test_PSFEquality():
    psf1 = PSFGenerator(PSF_SIZE)
    psf2 = PSFGenerator(PSF_SIZE)
    assert np.array_equal(psf1.psf, psf2.psf)

def test_AddNoise():
    psf = PSFGenerator(PSF_SIZE)
    psfNoise = PSFGenerator(PSF_SIZE)
    psfNoise.addNoise(0.1)
    assert not np.array_equal(psfNoise.psf, psf.psf)

def test_ShowPSF():
    psf = PSFGenerator(PSF_SIZE)
    psf.showPSF()

def test_PSFWithAstigmatismAberration():
    psfNoAberration = PSFGenerator(PSF_SIZE)
    psf = PSFWithAstigmatismAberration(PSF_SIZE)
    assert psf.size == PSF_SIZE and not np.array_equal(psf.psf, psfNoAberration.psf)

def test_PSFWithSphericalAberration():
    psfNoAberration = PSFGenerator(PSF_SIZE)
    psf = PSFWithSphericalAberration(PSF_SIZE)
    assert psf.size == PSF_SIZE and not np.array_equal(psf.psf, psfNoAberration.psf)

def test_PSFWithComaticAberration():
    psfNoAberration = PSFGenerator(PSF_SIZE)
    psf = PSFWithComaticAberration(PSF_SIZE, Intensity=0.08)
    assert psf.size == PSF_SIZE and not np.array_equal(psf.psf, psfNoAberration.psf)

def test_PSFRandomParameter():
    psf = PSFRandomParameter(PSF_SIZE)
    psfNoRandom = PSFGenerator(PSF_SIZE)
    assert psf.size == PSF_SIZE and not np.array_equal(psf.psf, psfNoRandom.psf)

def test_PSFRandomParameter_withAberration():
    psf = PSFRandomParameter(PSF_SIZE, aberrationType="astigmatism")
    psfNoRandom = PSFWithAstigmatismAberration(PSF_SIZE)
    assert psf.size == PSF_SIZE and not np.array_equal(psf.psf, psfNoRandom.psf)

def test_PSFRandomParameter_withSphericalAberration():
    psf = PSFRandomParameter(PSF_SIZE, aberrationType="spherical")
    psfNoRandom = PSFWithSphericalAberration(PSF_SIZE)
    assert psf.size == PSF_SIZE and not np.array_equal(psf.psf, psfNoRandom.psf)

def test_PSFRandomParameter_withComaticAberration():
    psf = PSFRandomParameter(PSF_SIZE, aberrationType="comatic")
    psfNoRandom = PSFWithComaticAberration(PSF_SIZE, Intensity=0.05)
    assert psf.size == PSF_SIZE and not np.array_equal(psf.psf, psfNoRandom.psf)
