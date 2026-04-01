from abc import ABC,abstractmethod

class TheoreticalResolution(object):
    """Standard class for theoretical microscope resolution calculation"""
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
        return [0, 0, 0]

