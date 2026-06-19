import os

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


def test_detection_getterSetters(image):
    """Unit test for Detection class getters and setters"""
    detector = Detection()
    detector.image = image
    assert np.array_equal(detector.image, image), "Image setter/getter failed"
    detector.cropFactor = 5
    assert detector.cropFactor == 5, "cropFactor setter/getter failed"
    detector.beadSize = 3.0
    assert detector.beadSize == 3.0, "beadSize setter/getter failed"
    detector.rejectionDistance = 15.0
    assert detector.rejectionDistance == 15.0, "rejectionDistance setter/getter failed"
    detector.pixelSize = np.array([1,1,1])
    assert np.array_equal(detector.pixelSize, np.array([1,1,1])), "pixelSize setter/getter failed"
    detector.thresholdTool = Threshold.getInstance("legacy")
    assert isinstance(detector.thresholdTool, Threshold), "thresholdTool setter/getter failed"

def test_getMeanIntensity_noBeads(image):
    """Unit test for getMeanIntensity method"""
    detector = Detection()
    detector._imageAnalyzer = ImageAnalyzer()
    mean_intensity = detector.getMeanIntensity()
    assert mean_intensity == 0, "getMeanIntensity returned a non-positive value"

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


def test_run(image, tmp_path):
    """Unit test for the complete detection and analysis pipeline"""
    bead_positions = [(10,30,40),(20,50,60),(40,70,80)]
    detector = Detection()
    detector.image = image
    detectionTool = DetectionTool.getInstance("Centroids")
    detectionTool._image = image
    detectionTool._thresholdTool = Threshold.getInstance("legacy")
    detector._detectionTool = detectionTool
    detector.cropFactor = 5
    detector.beadSize = 3.0
    detector.rejectionDistance = 15.0
    detector.pixelSize = np.array([1,1,1])
    for _ in detector.run(outputDir=tmp_path):
        pass
    assert detector._imageAnalyzer is not None, "_imageAnalyzer was not created"
    assert len(detector._imageAnalyzer._beadAnalyzer) == 3
    assert detector._imageAnalyzer._beadAnalyzer[0]._rejected == True


def test_getActivePath(tmp_path):
    """Unit test for getActivePath method"""
    detector = Detection()
    active_path = detector.getActivePath(0, tmp_path)
    assert active_path == str(tmp_path) + "/bead_0", "getActivePath did not return the expected path"

def test_addRoiOnImage(image):
    """Unit test for addRoiOnImage method"""
    bead_positions = [(10,30,40),(20,50,60),(40,70,80)]
    detector = Detection()
    detector.image = image
    detectionTool = DetectionTool.getInstance("Centroids")
    detectionTool._image = image
    detectionTool._thresholdTool = Threshold.getInstance("legacy")
    detector._detectionTool = detectionTool
    detector.cropFactor = 5
    detector.beadSize = 3.0
    detector.rejectionDistance = 15.0
    detector.pixelSize = np.array([1,1,1])
    for _ in detector.run(outputDir=None, cropPsf=False):
        pass
    annotated_image = detector.addRoiOnImage(roi=detector._imageAnalyzer._beadAnalyzer[1]._roi)
    assert annotated_image is not None, "addRoiOnImage did not return an image"

def test_cropPsf(image, tmp_path):
    """Unit test for cropPsf method"""
    bead_positions = [(10,30,40),(20,50,60),(40,70,80)]
    detector = Detection()
    detector.image = image
    detectionTool = DetectionTool.getInstance("Centroids")
    detectionTool._image = image
    detectionTool._thresholdTool = Threshold.getInstance("legacy")
    detector._detectionTool = detectionTool
    detector.cropFactor = 5
    detector.beadSize = 3.0
    detector.rejectionDistance = 15.0
    detector.pixelSize = np.array([1,1,1])
    for _ in detector.run(outputDir=None, cropPsf=False):
        pass
    detector.cropPsf(tmp_path)
    assert os.path.exists(tmp_path/f"bead_{detector._imageAnalyzer._beadAnalyzer[1]._id}/Localisation.png"), "cropPsf did not create the expected cropped PSF file"


def test_GlobalCropPsf(image, tmp_path):
    """Unit test for GlobalCropPsf method"""
    bead_positions = [(10,30,40),(20,50,60),(40,70,80)]
    detector = Detection()
    detector.image = image
    detectionTool = DetectionTool.getInstance("Centroids")
    detectionTool._image = image
    detectionTool._thresholdTool = Threshold.getInstance("legacy")
    detector._detectionTool = detectionTool
    detector.cropFactor = 5
    detector.beadSize = 3.0
    detector.rejectionDistance = 15.0
    detector.pixelSize = np.array([1,1,1])
    for _ in detector.run(outputDir=None, cropPsf=False):
        pass
    detector.GlobalCropPsf(tmp_path)
    assert os.path.exists(tmp_path/"Localisation.png"), "GlobalCropPsf did not create the expected global cropped PSF file"
