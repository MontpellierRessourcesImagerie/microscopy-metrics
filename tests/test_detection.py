import pytest
import numpy as np
from skimage.draw import disk
from microscopy_metrics.detection import *

def create_test_image(shape,bead_positions,radius=3,intensity=255):
    """Create a test image containing beads with known positions"""
    image = np.zeros(shape,dtype=np.uint8)
    for z,y,x in bead_positions:
        rr,cc = disk((x,y), radius,shape=shape[1:])
        image[z,cc,rr] = intensity
    return image

def test_detect_beads_peak_local_max():
    """Unit test for beads detection using peak_local_max"""
    shape = (50,100,100)
    bead_positions = [(10,30,40),(20,50,60),(30,70,80)]
    image = create_test_image(shape=shape,bead_positions=bead_positions)
    detected_beads = detect_psf_peak_local_max(image,threshold_auto=True)
    assert len(detected_beads) > 0
    tolerance = 3
    found = False
    for expected in bead_positions:
        if not found :
            for detected in detected_beads :
                if dist(expected,detected) < tolerance :
                    found = True
                    break
    assert found, "No expected bead found"

def test_detect_beads_blob_log():
    """Unit test for beads detection using blob_log"""
    shape = (50,100,100)
    bead_positions = [(10,30,40),(20,50,60),(30,70,80)]
    image = create_test_image(shape=shape,bead_positions=bead_positions)
    detected_beads = detect_psf_blob_log(image,threshold_auto=True)
    assert len(detected_beads) > 0
    tolerance = 3
    found = False
    for expected in bead_positions:
        if not found :
            for detected in detected_beads :
                if dist(expected,detected) < tolerance :
                    found = True
                    break
    assert found, "No expected bead found"


def test_detect_beads_blob_dog():
    """Unit test for beads detection using blob_dog"""
    shape = (50,100,100)
    bead_positions = [(10,30,40),(20,50,60),(30,70,80)]
    image = create_test_image(shape=shape,bead_positions=bead_positions)
    detected_beads = detect_psf_blob_dog(image,threshold_auto=True)
    assert len(detected_beads) > 0
    tolerance = 3
    found = False
    print(detected_beads)
    for expected in bead_positions:
        if not found :
            for detected in detected_beads :
                if dist(expected,detected) < tolerance :
                    found = True
                    break
    assert found, "No expected bead found"

def test_detect_beads_centroid():
    """Unit test for beads detection using centroid"""
    shape = (50,100,100)
    bead_positions = [(10,30,40),(20,50,60),(30,70,80)]
    image = create_test_image(shape=shape,bead_positions=bead_positions)
    detected_beads = detect_psf_centroid(image,threshold_auto=True)
    assert len(detected_beads) > 0
    print(detected_beads)
    tolerance = 3
    found = False
    for expected in bead_positions:
        if not found :
            for detected in detected_beads :
                if dist(expected,detected) < tolerance :
                    found = True
                    break
    assert found, "No expected bead found"

def test_extract_ROI():
    """Unit test for ROI extraction"""
    shape = (50,100,100)
    bead_positions = [(10,30,40),(20,50,60),(40,70,80)]
    image = create_test_image(shape=shape,bead_positions=bead_positions)
    detected_beads = detect_psf_centroid(image,threshold_auto=True)
    rois,centroid_ROI = extract_Region_Of_Interest(image,detected_beads,crop_factor=5,bead_size=3,rejection_zone=15)
    assert len(centroid_ROI) == 1 and centroid_ROI[0] == 1
