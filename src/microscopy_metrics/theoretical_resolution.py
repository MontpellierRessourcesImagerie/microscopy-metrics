from abc import ABC,abstractmethod
import math

class Theoretical_Resolution(object):
    """Standard class for theoretical microscope resolution calculation"""
    _microscopes_classes = {}

    def __init__(self):
        self._numerical_aperture = 0.9
        self._emission_wavelength = 490
        self._refractive_index = 1.5

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._microscopes_classes:
            raise ValueError("Class was already registered")
        cls._microscopes_classes[name] = cls

    @classmethod
    def get_instance(cls,method_name):
        microscope_class = cls._microscopes_classes[method_name]
        return microscope_class()

    @property
    @abstractmethod
    def name(self):
        pass


    @property
    def numerical_aperture(self):
        return self._numerical_aperture

    @numerical_aperture.setter
    def numerical_aperture(self, value):
        if not isinstance(value, float):
            raise ValueError(f"Numerical aperture must be a float : {value}")
        self._numerical_aperture = value

    @property
    def emission_wavelength(self):
        return self._emission_wavelength

    @emission_wavelength.setter
    def emission_wavelength(self, value):
        if not isinstance(value, float) and not isinstance(value, int):
            raise ValueError("Emission wavelength must be a number")
        self._emission_wavelength = value / 1000

    @property
    def refractive_index(self):
        return self._refractive_index

    @refractive_index.setter
    def refractive_index(self, value):
        if not isinstance(value, float):
            raise ValueError("Refractive index must be a float")
        self._refractive_index = value

    def get_theoretical_resolution(self):
        return [0, 0, 0]


class Widefield_resolution(Theoretical_Resolution):
    name = "widefield"
    def __init__(self):
        super(Widefield_resolution, self).__init__()

    def get_theoretical_resolution(self):
        r_xy = (0.51 * self._emission_wavelength) / self._numerical_aperture
        r_z = (1.77 * self._refractive_index * self._emission_wavelength) / (
            self._numerical_aperture**2
        )
        return [r_z, r_xy, r_xy]


class Confocal_resolution(Theoretical_Resolution):
    name="confocal"
    def __init__(self):
        super(Confocal_resolution, self).__init__()

    def get_theoretical_resolution(self):
        r_xy = (0.51 * self._emission_wavelength) / self._numerical_aperture
        r_z = (0.88 * self._emission_wavelength) / (
            self._refractive_index
            - math.sqrt(self._refractive_index**2 - self._numerical_aperture**2)
        )
        return [r_z, r_xy, r_xy]


class Spinning_disk_resolution(Theoretical_Resolution):
    name="spinning disk"
    def __init__(self):
        super(Spinning_disk_resolution, self).__init__()

    def get_theoretical_resolution(self):
        r_xy = (0.51 * self._emission_wavelength) / self._numerical_aperture
        r_z = self._emission_wavelength / (
            self._refractive_index
            - math.sqrt(self._refractive_index**2 - self._numerical_aperture**2)
        )
        return [r_z, r_xy, r_xy]


class Multiphoton_resolution(Theoretical_Resolution):
    name="multiphoton"
    def __init__(self):
        super(Multiphoton_resolution, self).__init__()

    def get_theoretical_resolution(self):
        if self._numerical_aperture < 0.7:
            r_xy = (0.377 * self._emission_wavelength) / self._numerical_aperture
        else:
            r_xy = (0.383 * self._emission_wavelength) / (
                self._numerical_aperture**0.91
            )
        r_z = (0.626 * self._emission_wavelength) / (
            self._refractive_index
            - math.sqrt(self._refractive_index**2 - self._numerical_aperture**2)
        )
        return [r_z, r_xy, r_xy]


