from microscopy_metrics.threshold_tool import Threshold
import numpy as np

def create_uniform_image(shape=(100, 100), low=0, high=1):
    """Create an Image with a uniform distribution"""
    return np.random.uniform(low, high, shape)

def test_legacy_threshold_uniform():
    """Unit test of legacy threshold on uniform image"""
    image = create_uniform_image(low=0, high=1)
    thresholder = Threshold.getInstance("legacy")
    threshold = thresholder.getThreshold(image)
    assert np.isclose(threshold, 0.5, rtol=0.1)

def test_otsu_threshold_uniform():
    """Unit test of otsu threshold on uniform image"""
    image = create_uniform_image(low=0, high=1)
    thresholder = Threshold.getInstance("otsu")
    threshold = thresholder.getThreshold(image)
    assert np.isclose(threshold, 0.5, rtol=0.2)

def test_isodata_threshold_uniform():
    """Unit test of isodata threshold on uniform image"""
    image = create_uniform_image(low=0, high=1)
    thresholder = Threshold.getInstance("isodata")
    threshold = thresholder.getThreshold(image)
    assert np.isclose(threshold, 0.5, rtol=0.2)

def test_li_threshold_uniform():
    """Unit test of li threshold on uniform image"""
    image = create_uniform_image(low=0, high=1)
    thresholder = Threshold.getInstance("li")
    threshold = thresholder.getThreshold(image)
    assert np.isclose(threshold, 0.5, rtol=0.3)

def test_minimum_threshold_uniform():
    """Unit test of minimum threshold on uniform image"""
    image = create_uniform_image(low=0, high=1)
    thresholder = Threshold.getInstance("minimum")
    threshold = thresholder.getThreshold(image)
    assert threshold is not None and threshold > 0

def test_triangle_threshold_uniform():
    """Unit test of triangle threshold on uniform image"""
    image = create_uniform_image(low=0, high=1)
    thresholder = Threshold.getInstance("triangle")
    threshold = thresholder.getThreshold(image)
    assert threshold is not None and threshold > 0
