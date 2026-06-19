from microscopy_metrics.resolutionTools.theoretical_resolution import (
    TheoreticalResolution,
)
import numpy as np


def test_WidefieldResolution():
    """Unit test of Widefield resolution calculation"""
    widefield = TheoreticalResolution.getInstance("widefield")
    results = widefield.getTheoreticalResolution()
    assert np.isclose(results[1], 255.0, rtol=0.000001)
    assert np.isclose(results[0], 1475.0, rtol=0.0001)


def test_WidefieldSamplingDistance():
    """Unit test of Widefield sampling distance calculation"""
    widefield = TheoreticalResolution.getInstance("widefield")
    results = widefield.getSamplingDistance()
    assert np.isclose(results[1], 125.0, rtol=0.000001)
    assert np.isclose(results[0], 750.0, rtol=0.0001)


def test_ConfocalResolution():
    """Unit test of Confocal resolution calculation"""
    confocal = TheoreticalResolution.getInstance("confocal")
    results = confocal.getTheoreticalResolution()
    assert np.isclose(results[1], 127.5, rtol=0.000001)
    assert np.isclose(results[0], 660.0, rtol=0.0001)


def test_ConfocalSamplingDistance():
    """Unit test of Confocal sampling distance calculation"""
    confocal = TheoreticalResolution.getInstance("confocal")
    results = confocal.getSamplingDistance()
    assert np.isclose(results[1], 31.25, rtol=0.000001)
    assert np.isclose(results[0], 187.5, rtol=0.0001)


def test_SpinningDiskResolution():
    """Unit test of Spinning Disk resolution calculation"""
    spinning = TheoreticalResolution.getInstance("spinning disk")
    results = spinning.getTheoreticalResolution()
    assert np.isclose(results[1], 255.0, rtol=0.000001)
    assert np.isclose(results[0], 1500.0, rtol=0.0001)


def test_SpinningDiskSamplingDistance():
    """Unit test of Spinning Disk sampling distance calculation"""
    spinning = TheoreticalResolution.getInstance("spinning disk")
    results = spinning.getSamplingDistance()
    assert np.isclose(results[1], 125.0, rtol=0.000001)
    assert np.isclose(results[0], 750.0, rtol=0.0001)


def test_MultiphotonResolution():
    """Unit test of Multiphoton resolution calculation"""
    multiphoton = TheoreticalResolution.getInstance("multiphoton")
    multiphoton._numericalAperture = 0.5
    results = multiphoton.getTheoreticalResolution()
    assert np.isclose(results[1], 169.65, rtol=0.01)
    assert np.isclose(results[0], 1641.867921041, rtol=0.0001)
    multiphoton._numericalAperture = 0.9
    results = multiphoton.getTheoreticalResolution()
    assert np.isclose(results[1], 94.846346948, rtol=0.000001)
    assert np.isclose(results[0], 469.5, rtol=0.0001)


def test_MultiphotonSamplingDistance():
    """Unit test of Multiphoton sampling distance calculation"""
    multiphoton = TheoreticalResolution.getInstance("multiphoton")
    results = multiphoton.getSamplingDistance()
    assert np.isclose(results[1], 31.25, rtol=0.000001)
    assert np.isclose(results[0], 187.5, rtol=0.0001)


def test_UnknownMicroscopyType():
    """Unit test of unknown microscopy type"""
    unknown = TheoreticalResolution()
    results = unknown.getTheoreticalResolution()
    assert results == [0,0,0]
    unknown.angularAperture()
    assert np.isclose(unknown._angularAperture, 0.64, atol=0.01)

def test_theoreticalResolution_GetterSetter():
    """Unit test of theoretical resolution getter and setter"""
    theoreticalResolution = TheoreticalResolution()
    theoreticalResolution.numericalAperture = 0.5
    assert np.isclose(theoreticalResolution.numericalAperture, 0.5, rtol=0.000001)
    theoreticalResolution.refractiveIndex = 1.33
    assert np.isclose(theoreticalResolution.refractiveIndex, 1.33, rtol=0.000001)
    theoreticalResolution.emissionWavelength = 500
    assert np.isclose(theoreticalResolution.emissionWavelength, 0.5, rtol=0.000001)
    theoreticalResolution.excitationWavelength = 400
    assert np.isclose(theoreticalResolution.excitationWavelength, 0.4, rtol=0.000001)
