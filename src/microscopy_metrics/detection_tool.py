from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.measure import regionprops, label
from .threshold_tool import Threshold
from scipy import ndimage as ndi
import numpy as np
from abc import ABC, abstractmethod


class Detection_Tool(object):
    _detection_classes = {}

    def __init__(self):
        self._image = None
        self._sigma = 2.0
        self._threshold_tool = None
        self.normalized_image = None
        self.high_passed_im = None
        self.centroids = []

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._detection_classes:
            raise ValueError("Class was already registered")
        cls._detection_classes[name] = cls

    @classmethod
    def get_instance(cls, method_name):
        detection_class = cls._detection_classes[method_name]
        return detection_class()

    def set_normalized_image(self):
        """Method to normalize a 2D or 3D image and erase negative values

        Raises:
            ValueError: This function only operate on 2D or 3D images
        """
        if self._image.ndim not in (2, 3):
            raise ValueError("Image have to be in 2D or 3D.")
        self.normalized_image = self._image.astype(np.float32)
        self.normalized_image = (
            self.normalized_image - np.min(self.normalized_image)
        ) / (np.max(self.normalized_image) - np.min(self.normalized_image) + 1e-6)
        self.normalized_image[self.normalized_image < 0] = 0

    def gaussian_high_pass(self):
        low_pass = ndi.gaussian_filter(self.normalized_image, self._sigma)
        self.high_passed_im = self.normalized_image - low_pass

    @abstractmethod
    def detect(self):
        pass


    @property
    @abstractmethod
    def name(self):
        pass

class Peak_Local_Max_Detector(Detection_Tool):
    name = "peak local maxima"
    
    def __init__(self):
        super(Peak_Local_Max_Detector, self).__init__()
        self._min_distance = 1
    
    def detect(self):
        self.set_normalized_image()
        self.normalized_image = ndi.gaussian_filter(self.normalized_image, sigma=2.0)
        self.gaussian_high_pass()
        self.centroids = peak_local_max(
            self.high_passed_im,
            min_distance=self._min_distance,
            threshold_abs=self._threshold_tool.get_threshold(self.high_passed_im),
        )

class Blob_Log_Detector(Detection_Tool):
    name = "Laplacian of Gaussian"
    
    def __init__(self):
        super(Blob_Log_Detector, self).__init__()
    
    def detect(self):
        self.set_normalized_image()
        blobs = blob_log(
            self.normalized_image,
            max_sigma=self._sigma,
            threshold=self._threshold_tool.get_threshold(self.normalized_image),
        )
        self.centroids = np.array([[blob[0], blob[1], blob[2]] for blob in blobs])

class Blob_Dog_Detector(Detection_Tool):
    name = "Difference of Gaussian"
    
    def __init__(self):
        super(Blob_Dog_Detector, self).__init__()
    
    def detect(self):
        self.set_normalized_image()
        blobs = blob_dog(
            self.normalized_image,
            max_sigma=self._sigma,
            threshold=self._threshold_tool.get_threshold(self.normalized_image),
        )
        self.centroids = np.array([[blob[0], blob[1], blob[2]] for blob in blobs])

class Centroid_Detector(Detection_Tool):
    name = "Centroids"

    def __init__(self):
        super(Centroid_Detector, self).__init__()
    
    def detect(self):
        self.set_normalized_image()
        self.normalized_image = ndi.gaussian_filter(self.normalized_image, sigma=2.0)
        self.gaussian_high_pass()
        binary_image = self.high_passed_im > self._threshold_tool.get_threshold(self.high_passed_im)
        labeled_image = label(binary_image)
        region_props = regionprops(labeled_image)
        tmp_centroids = []
        for prop in region_props:
            tmp_centroids.append(prop.centroid)

        self.centroids = np.array(tmp_centroids)