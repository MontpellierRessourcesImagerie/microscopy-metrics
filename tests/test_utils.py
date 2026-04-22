from microscopy_metrics.utils import *

def test_um_to_px():
    """Unit test for the um to px converter"""
    assert umToPx(10,0.0) == 0.0
    axisPhysicalSize = 10.0
    size_um = 100.0
    assert umToPx(size_um,axisPhysicalSize) == 10.0

def test_px_to_um():
    """Unit test for the um to px converter"""
    axisPhysicalSize = 10.0
    size_px = 1.0
    assert pxToUm(size_px,axisPhysicalSize) == 10.0


