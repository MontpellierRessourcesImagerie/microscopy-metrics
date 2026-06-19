import math
import numpy as np

from abc import abstractmethod


class TheoreticalResolution(object):
    """Abstract base class for calculating the theoretical resolution of a microscopy system based on its numerical aperture, emission wavelength, and refractive index.
    This class provides a common interface and shared functionality for different types of microscopes (e.g., widefield, confocal).
    Subclasses must implement the getTheoreticalResolution method to specify the calculation based on the specific microscope type.
    The class also maintains a registry of microscope classes for easy instantiation based on method names.
    """

    _microscopesClasses = {}

    def __init__(self):
        self._numericalAperture = 0.9
        self._emissionWavelength = 450
        self._refractiveIndex = 1.5
        self._angularAperture = None
        self._excitationWavelength = 225

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
    def excitationWavelength(self):
        return self._excitationWavelength

    @excitationWavelength.setter
    def excitationWavelength(self, value):
        if not isinstance(value, float) and not isinstance(value, int):
            raise ValueError("Excitation wavelength must be a number")
        self._excitationWavelength = value / 1000

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

    def angularAperture(self):
        """Calculates the angular aperture of the microscope based on its numerical aperture and refractive index."""
        if self._angularAperture is None:
            self._angularAperture = np.arcsin(
                self._numericalAperture / self._refractiveIndex
            )
        return self._angularAperture


class WidefieldResolution(TheoreticalResolution):
    """Class for calculating the theoretical resolution of a widefield microscope based on its parameters.
    This class inherits from the TheoreticalResolution base class and implements the method to calculate the theoretical resolution in the XY and Z dimensions using the appropriate formulas for widefield microscopy.
    """

    name = "widefield"

    def __init__(self):
        super(WidefieldResolution, self).__init__()

    def getTheoreticalResolution(self):
        """Calculates the theoretical resolution of a widefield microscope in the XY and Z dimensions based on the emission wavelength, numerical aperture, and refractive index.
        Returns:
            list: A list containing the theoretical resolution in the Z dimension followed by the resolution in the XY dimensions (resZ, resXY, resXY).
        """
        resXY = (0.51 * self._emissionWavelength) / self._numericalAperture
        resZ = (1.77 * self._refractiveIndex * self._emissionWavelength) / (
            self._numericalAperture**2
        )
        return [resZ, resXY, resXY]

    def getSamplingDistance(self):
        """Calculates the recommended sampling distance for a widefield microscope based on the theoretical resolution in the XY and Z dimensions.
        Returns:
            list: A list containing the recommended sampling distance in the Z dimension followed by the distance in the XY dimensions (distZ, distXY, distXY).
        """
        res = self.angularAperture()
        distXY = self._emissionWavelength / (4 * self._numericalAperture)
        distZ = self._emissionWavelength / (
            2 * self._refractiveIndex * (1 - np.cos(res))
        )
        return [distZ, distXY, distXY]
    

class ConfocalResolution(TheoreticalResolution):
    """Class for calculating the theoretical resolution of a confocal microscope based on its parameters.
    This class inherits from the TheoreticalResolution base class and implements the method to calculate the theoretical resolution in the XY and Z dimensions using the appropriate formulas for confocal microscopy.
    """

    name = "confocal"

    def __init__(self):
        super(ConfocalResolution, self).__init__()

    def getTheoreticalResolution(self):
        """Calculates the theoretical resolution of a confocal microscope in the XY and Z dimensions based on the emission wavelength, numerical aperture, and refractive index.
        Returns:
            list: A list containing the theoretical resolution in the Z dimension followed by the resolution in the XY dimensions (resZ, resXY, resXY).
        """
        resXY = (0.51 * self._excitationWavelength) / self._numericalAperture
        resZ = (0.88 * self._excitationWavelength) / (
            self._refractiveIndex
            - math.sqrt(self._refractiveIndex**2 - self._numericalAperture**2)
        )
        return [resZ, resXY, resXY]

    def getSamplingDistance(self):
        """Calculates the recommended sampling distance for a confocal microscope based on the theoretical resolution in the XY and Z dimensions.
        Returns:
            list: A list containing the recommended sampling distance in the Z dimension followed by the distance in the XY dimensions (distZ, distXY, distXY).
        """
        res = self.angularAperture()
        distXY = self._excitationWavelength / (8 * self._numericalAperture)
        distZ = self._excitationWavelength / (
            4 * self._refractiveIndex * (1 - math.cos(res))
        )
        return [distZ, distXY, distXY]
    

class MultiphotonResolution(TheoreticalResolution):
    """Class for calculating the theoretical resolution of a multiphoton microscope based on its parameters.
    This class inherits from the TheoreticalResolution base class and implements the method to calculate the theoretical resolution in the XY and Z dimensions using the appropriate formulas for multiphoton microscopy.
    """

    name = "multiphoton"

    def __init__(self):
        super(MultiphotonResolution, self).__init__()

    def getTheoreticalResolution(self):
        """Calculates the theoretical resolution of a multiphoton microscope in the XY and Z dimensions based on the emission wavelength, numerical aperture, and refractive index.
        Returns:
            list: A list containing the theoretical resolution in the Z dimension followed by the resolution in the XY dimensions (resZ, resXY, resXY).
        """
        if self._numericalAperture < 0.7:
            resXY = (0.377 * self._excitationWavelength) / self._numericalAperture
        else:
            resXY = (0.383 * self._excitationWavelength) / (
                self._numericalAperture**0.91
            )
        resZ = (0.626 * self._excitationWavelength) / (
            self._refractiveIndex
            - math.sqrt(self._refractiveIndex**2 - self._numericalAperture**2)
        )
        return [resZ, resXY, resXY]

    def getSamplingDistance(self, k=2):
        """Calculates the recommended sampling distance for a multiphoton microscope based on the theoretical resolution in the XY and Z dimensions.
        Returns:
            list: A list containing the recommended sampling distance in the Z dimension followed by the distance in the XY dimensions (distZ, distXY, distXY).
        """
        res = self.angularAperture()
        distXY = self._excitationWavelength / (4 * k * self._numericalAperture)
        distZ = self._excitationWavelength / (
            2 * k * self._refractiveIndex * (1 - math.cos(res))
        )
        return [distZ, distXY, distXY]


class SpinningDiskResolution(TheoreticalResolution):
    """Class for calculating the theoretical resolution of a spinning disk confocal microscope based on its parameters.
    This class inherits from the TheoreticalResolution base class and implements the method to calculate the theoretical resolution in the XY and Z dimensions using the appropriate formulas for spinning disk confocal microscopy.
    """

    name = "spinning disk"

    def __init__(self):
        super(SpinningDiskResolution, self).__init__()

    def getTheoreticalResolution(self):
        """Calculates the theoretical resolution of a spinning disk confocal microscope in the XY and Z dimensions based on the emission wavelength, numerical aperture, and refractive index.
        Returns:
            list: A list containing the theoretical resolution in the Z dimension followed by the resolution in the XY dimensions (resZ, resXY, resXY).
        """
        resXY = (0.51 * self._emissionWavelength) / self._numericalAperture
        resZ = self._emissionWavelength / (
            self._refractiveIndex
            - math.sqrt(self._refractiveIndex**2 - self._numericalAperture**2)
        )
        return [resZ, resXY, resXY]

    def getSamplingDistance(self):
        """Calculates the recommended sampling distance for a spinning disk confocal microscope based on the theoretical resolution in the XY and Z dimensions.
        Returns:
            list: A list containing the recommended sampling distance in the Z dimension followed by the distance in the XY dimensions (distZ, distXY, distXY).
        """
        res = self.angularAperture()
        distXY = self._emissionWavelength / (4 * self._numericalAperture)
        distZ = self._emissionWavelength / (
            2 * self._refractiveIndex * (1 - math.cos(res))
        )
        return [distZ, distXY, distXY]
