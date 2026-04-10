from abc import abstractmethod


class TheoreticalResolution(object):
    """Abstract base class for calculating the theoretical resolution of a microscopy system based on its numerical aperture, emission wavelength, and refractive index.
    This class provides a common interface and shared functionality for different types of microscopes (e.g., widefield, confocal).
    It includes methods for setting the microscope parameters and an abstract method for calculating the theoretical resolution.
    Subclasses must implement the getTheoreticalResolution method to specify the calculation based on the specific microscope type. The class also maintains a registry of microscope classes for easy instantiation based on method names.
    """

    _microscopesClasses = {}

    def __init__(self):
        self._numericalAperture = 0.9
        self._emissionWavelength = 490
        self._refractiveIndex = 1.5

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._microscopesClasses:
            raise ValueError("Class was already registered")
        cls._microscopesClasses[name] = cls

    @classmethod
    def getInstance(cls, methodName):
        """Factory method to create an instance of a microscope class based on the provided method name.
        Args:
            methodName (str): The name of the microscope class to instantiate.
        Returns:
            TheoreticalResolution: An instance of the specified microscope class.
        """
        microscopeClass = cls._microscopesClasses[methodName]
        return microscopeClass()

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def numericalAperture(self):
        return self._numericalAperture

    @numericalAperture.setter
    def numericalAperture(self, value):
        if not isinstance(value, float):
            raise ValueError(f"Numerical aperture must be a float : {value}")
        self._numericalAperture = value

    @property
    def emissionWavelength(self):
        return self._emissionWavelength

    @emissionWavelength.setter
    def emissionWavelength(self, value):
        if not isinstance(value, float) and not isinstance(value, int):
            raise ValueError("Emission wavelength must be a number")
        self._emissionWavelength = value / 1000

    @property
    def refractiveIndex(self):
        return self._refractiveIndex

    @refractiveIndex.setter
    def refractiveIndex(self, value):
        if not isinstance(value, float):
            raise ValueError("Refractive index must be a float")
        self._refractiveIndex = value

    def getTheoreticalResolution(self):
        """Abstract method to calculate the theoretical resolution of the microscope based on its parameters."""
        return [0, 0, 0]
