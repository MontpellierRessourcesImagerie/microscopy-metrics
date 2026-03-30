from microscopy_metrics.theoretical_resolution import TheoreticalResolution
import numpy as np

def test_WidefieldResolution():
    """Unit test of Widefield resolution calculation"""
    widefield = TheoreticalResolution.getInstance("widefield")
    results = widefield.getTheoreticalResolution()
    assert np.isclose(results[1], 277.666667, rtol=0.000001)
    assert np.isclose(results[0], 1606.11111, rtol=0.0001)


def test_ConfocalResolution():
    """Unit test of Confocal resolution calculation"""
    confocal = TheoreticalResolution.getInstance("confocal")
    results = confocal.getTheoreticalResolution()
    assert np.isclose(results[1], 277.666667, rtol=0.000001)
    assert np.isclose(results[0], 1437.33333, rtol=0.0001)


def test_SpinningDiskResolution():
    """Unit test of Spinning Disk resolution calculation"""
    spinning = TheoreticalResolution.getInstance("spinning disk")
    results = spinning.getTheoreticalResolution()
    assert np.isclose(results[1], 277.666667, rtol=0.000001)
    assert np.isclose(results[0], 1633.33333, rtol=0.0001)


def test_MultiphotonResolution():
    """Unit test of Multiphoton resolution calculation"""
    multiphoton = TheoreticalResolution.getInstance("multiphoton")
    multiphoton._numericalAperture = 0.5
    results = multiphoton.getTheoreticalResolution()
    assert np.isclose(results[1], 369.46, rtol=0.01)
    assert np.isclose(results[0], 3575.623472489, rtol=0.0001)
    multiphoton._numericalAperture = 0.9
    results = multiphoton.getTheoreticalResolution()
    assert np.isclose(results[1], 206.554266687, rtol=0.000001)
    assert np.isclose(results[0], 1022.466666667, rtol=0.0001)
