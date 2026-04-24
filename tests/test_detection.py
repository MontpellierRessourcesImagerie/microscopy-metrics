import pytest
import numpy as np
from skimage.draw import disk
from microscopy_metrics.detection import Detection
from microscopy_metrics.BeadAnalyzer import BeadAnalyzer
from microscopy_metrics.ImageAnalyzer import ImageAnalyzer
from microscopy_metrics.detectionTools.detection_tool import DetectionTool
from microscopy_metrics.thresholdTools.threshold_tool import Threshold

@pytest.fixture
def image():
    shape = (50,100,100)
    bead_positions = [(10,30,40),(20,50,60),(40,70,80)]
    image = np.zeros(shape,dtype=np.uint8)
    for z,y,x in bead_positions:
        rr,cc = disk((x,y), 3,shape=shape[1:])
        image[z,cc,rr] = 255
    yield image

def test_detect_beads_peak_local_max(image):
    """Unit test for beads detection using peak_local_max"""
    bead_positions = [(10,30,40),(20,50,60),(40,70,80)]
    detectionTool = DetectionTool.getInstance("peak local maxima")
    detectionTool._image = image
    detectionTool._thresholdTool = Threshold.getInstance("legacy")
    detectionTool.detect()
    detected_beads = detectionTool._centroids
    assert len(detected_beads) > 0
    tolerance = 3
    found = False
    for expected in bead_positions:
        if not found :
            for detected in detected_beads :
                if np.linalg.norm(expected - detected) < tolerance :
                    found = True
                    break
    assert found, "No expected bead found"

def test_detect_beads_blob_log(image):
    """Unit test for beads detection using blob_log"""
    bead_positions = [(10,30,40),(20,50,60),(40,70,80)]
    detectionTool = DetectionTool.getInstance("Laplacian of Gaussian")
    detectionTool._image = image
    detectionTool._thresholdTool = Threshold.getInstance("legacy")
    detectionTool.detect()
    detected_beads = detectionTool._centroids
    assert len(detected_beads) > 0
    tolerance = 3
    found = False
    for expected in bead_positions:
        if not found :
            for detected in detected_beads :
                if np.linalg.norm(expected - detected) < tolerance :
                    found = True
                    break
    assert found, "No expected bead found"


def test_detect_beads_blob_dog(image):
    """Unit test for beads detection using blob_dog"""
    bead_positions = [(10,30,40),(20,50,60),(40,70,80)]
    detectionTool = DetectionTool.getInstance("Difference of Gaussian")
    detectionTool._image = image
    detectionTool._thresholdTool = Threshold.getInstance("otsu")
    detectionTool.detect()
    detected_beads = detectionTool._centroids
    assert len(detected_beads) > 0
    tolerance = 3
    found = False
    for expected in bead_positions:
        if not found :
            for detected in detected_beads :
                if np.linalg.norm(expected - detected) < tolerance :
                    found = True
                    break
    assert found, "No expected bead found"

def test_detect_beads_centroid(image):
    """Unit test for beads detection using centroid"""
    bead_positions = [(10,30,40),(20,50,60),(40,70,80)]
    detectionTool = DetectionTool.getInstance("Centroids")
    detectionTool._image = image
    detectionTool._thresholdTool = Threshold.getInstance("legacy")
    detectionTool.detect()
    detected_beads = detectionTool._centroids
    assert len(detected_beads) > 0
    print(detected_beads)
    tolerance = 3
    found = False
    for expected in bead_positions:
        if not found :
            for detected in detected_beads :
                if np.linalg.norm(expected - detected) < tolerance :
                    found = True
                    break
    assert found, "No expected bead found"

def test_extract_ROI(image):
    """Unit test for ROI extraction"""
    bead_positions = [(10,30,40),(20,50,60),(40,70,80)]
    detector = Detection()
    detector.image = image
    detectionTool = DetectionTool.getInstance("Centroids")
    detectionTool._image = image
    detectionTool._thresholdTool = Threshold.getInstance("legacy")
    detectionTool.detect()
    
    
    detector._imageAnalyzer = ImageAnalyzer(image=image, beadSize=3, pixelSize=[1,1,1])
    for i, centroid in enumerate(detectionTool._centroids):
        detector._imageAnalyzer._beadAnalyzer.append(BeadAnalyzer(id=i, centroid=centroid))
    detector.cropFactor = 5
    detector.beadSize = 3.0
    detector.rejectionDistance = 15.0
    detector.pixelSize = np.array([1,1,1])
    detector.extractRegionOfInterest()
    
    assert len(detector._imageAnalyzer._beadAnalyzer) == 3 and detector._imageAnalyzer._beadAnalyzer[0]._rejected == True
