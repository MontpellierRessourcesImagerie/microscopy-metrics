from skimage.filters import (
    threshold_otsu,
    threshold_isodata,
    threshold_li,
    threshold_minimum,
    threshold_triangle,
)
import numpy as np
from abc import ABC, abstractmethod


class Threshold(object):
    name = "middle"
    _threshold_classes = {}

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._threshold_classes:
            raise ValueError("Class was already registered")
        cls._threshold_classes[name] = cls

    @classmethod
    def get_instance(cls, method_name):
        threshold_class = cls._threshold_classes[method_name]
        return threshold_class()

    def get_threshold(self, image):
        return max(image) / 2

    @property
    @abstractmethod
    def name(self):
        pass


class Threshold_Otsu(Threshold):
    name = "otsu"

    def __init__(self):
        super(Threshold_Otsu, self).__init__()

    def get_threshold(self, image):
        return threshold_otsu(image)


class Threshold_Isodata(Threshold):
    name = "isodata"

    def __init__(self):
        super(Threshold_Isodata, self).__init__()

    def get_threshold(self, image):
        return threshold_isodata(image)


class Threshold_Li(Threshold):
    name = "li"

    def __init__(self):
        super(Threshold_Li, self).__init__()

    def get_threshold(self, image):
        return threshold_li(image)


class Threshold_Minimum(Threshold):
    name = "minimum"

    def __init__(self):
        super(Threshold_Minimum, self).__init__()

    def get_threshold(self, image):
        return threshold_minimum(image)


class Threshold_Triangle(Threshold):
    name = "triangle"

    def __init__(self):
        super(Threshold_Triangle, self).__init__()

    def get_threshold(self, image):
        return threshold_triangle(image)


class Threshold_Legacy(Threshold):
    name = "legacy"

    def __init__(self, nb_iteration=100):
        super(Threshold_Legacy, self).__init__()
        self.nb_iteration = nb_iteration

    def get_threshold(self, image):
        """Apply the Metroloj_Qc's 'legacy threshold'

        Returns:
            float: Value of the threshold
        """
        img_min = np.min(image)
        img_max = np.max(image)
        midpoint = (img_max - img_min) / 2
        image[image < 0] = 0
        for i in range(self.nb_iteration):
            background = image[image <= midpoint]
            signal = image[image > midpoint]
            mean_background = np.mean(background) if len(background) > 0 else img_min
            mean_signal = np.mean(signal) if len(signal) > 0 else img_max
            n_midpoint = (mean_background + mean_signal) / 2
            if abs(n_midpoint - midpoint) < 1e-6:
                break
            midpoint = n_midpoint
        return midpoint


class Threshold_Manual(Threshold):
    name = "manual"

    def __init__(self, rel_threshold=0.5):
        super(Threshold_Manual, self).__init__()
        self.rel_threshold = rel_threshold

    def get_threshold(self, image):
        return self.rel_threshold * np.max(image)
