import numpy as np
import psfmodels as psfm
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates


class PSFGenerator(object):
    """Class for generating a Point Spread Function (PSF) for microscopy images based on specified parameters such as size, pixel size, refractive index, wavelength, and numerical aperture.
    The class provides methods for generating the PSF, computing its full width at half maximum (FWHM), adding noise to the PSF, and visualizing the PSF.
    Attributes:
        size (int): Size of the PSF image (default: 100).
        dxy (float): Pixel size in the XY plane (default: 0.05).
        dz (float): Pixel size in the Z direction (default: 0.05).
        ni0 (float): Refractive index of the immersion medium (default: 1.515).
        ni (float): Refractive index of the sample medium (default: 1.515).
        wvl (float): Wavelength of light used for imaging (default: 0.5).
        NA (float): Numerical aperture of the objective lens (default: 1.4).
        psf (np.ndarray): Generated PSF image.
        fwhm (list): Full width at half maximum values for Z, Y, and X axes.
    """
    def __init__(
        self, size=100, dxy=0.05, dz=0.05, ni0=1.515, ni=1.515, wvl=0.5, NA=1.4
    ):
        self.size : int = size
        self.dxy : float = dxy
        self.dz : float = dz
        self.ni0 : float = ni0
        self.ni : float = ni
        self.wvl : float = wvl
        self.NA : float = NA
        self.psf : np.ndarray = self.generate_psf()
        self.fwhm : list = self.computeFWHM()

    def generate_psf(self):
        """Generates the Point Spread Function (PSF) based on the specified parameters using the psfmodels library.
        Returns:
            np.ndarray: The generated PSF image.
        """
        psf = psfm.make_psf(
            self.size,
            self.size,
            dxy=self.dxy,
            dz=self.dz,
            pz=0.0,
            ni0=self.ni0,
            ni=self.ni,
            wvl=self.wvl,
            NA=self.NA,
        )
        psf = (psf / np.max(psf)) * 255.0
        return psf

    def computeFWHM(self):
        """Computes the full width at half maximum (FWHM) for the generated PSF along the Z, Y, and X axes based on the specified parameters.
        Returns:
            list: A list containing the FWHM values for Z, Y, and X axes.
        """ 
        resXY = (0.51 * self.wvl * 100) / self.NA
        resZ = (1.77 * self.ni * self.wvl * 100) / (self.NA**2)
        return [resZ, resXY, resXY]

    def addNoise(self, mean=0, std=2.0):
        """Adds Gaussian noise to the generated PSF image based on the specified mean and standard deviation.   
        Arguments:
            mean (float): Mean of the Gaussian noise to be added (default: 0).
            std (float): Standard deviation of the Gaussian noise to be added (default: 2.0).
        """
        noise = np.random.normal(mean, std, size=self.psf.shape)
        self.psf += noise
        self.psf = np.clip(self.psf, 0, None)

    def showPSF(self):
        """Displays the generated PSF image using matplotlib, showing the central slices along the Z and Y axes."""
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.psf[self.psf.shape[0] // 2], cmap="gray")
        ax2.imshow(self.psf[:, self.psf.shape[1] // 2], cmap="gray")
        plt.show()


class PSFWithComaticAberration(PSFGenerator):
    """Class for generating a Point Spread Function (PSF) with comatic aberration based on specified parameters such as size, pixel size, refractive index, wavelength, numerical aperture, and intensity of the aberration.
    Attributes:
        size (int): Size of the PSF image (default: 100).
        dxy (float): Pixel size in the XY plane (default: 0.05).
        dz (float): Pixel size in the Z direction (default: 0.05).
        ni0 (float): Refractive index of the immersion medium (default: 1.515).
        ni (float): Refractive index of the sample medium (default: 1.515).
        wvl (float): Wavelength of light used for imaging (default: 0.5).
        NA (float): Numerical aperture of the objective lens (default: 1.4).
        Intensity (float): Intensity of the comatic aberration (default: random value between 0.02 and 0.08).
        psf (np.ndarray): Generated PSF image with comatic aberration.
    """
    def __init__(
        self, size=100, dxy=0.05, dz=0.05, ni0=1.515, ni=1.515, wvl=0.5, NA=1.4, Intensity=None
    ):
        super(PSFWithComaticAberration, self).__init__(size, dxy, dz, ni0, ni, wvl, NA)
        self.intensity : float = Intensity if Intensity is not None else np.random.uniform(0.02, 0.08)
        self.psf : np.ndarray = self.generate_comatic_aberration()

    def generate_comatic_aberration(self):
        """Generates the Point Spread Function (PSF) with comatic aberration based on the specified parameters using the psfmodels library and applies a comatic distortion to the PSF.
        Returns:
            np.ndarray: The generated PSF image with comatic aberration.
        """
        shape = (self.size, self.size, self.size)
        z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
        zC, _, _ = np.array(shape) // 2
        psf_base = np.asarray(self.psf)
        xShift = ((z - zC) ** 2) * self.intensity
        xRenseigne = x - xShift
        psf_banane = map_coordinates(psf_base, [z, y, xRenseigne], order=1)
        return psf_banane


class PSFWithAstigmatismAberration(PSFGenerator):
    """Class for generating a Point Spread Function (PSF) with astigmatism aberration based on specified parameters such as size, pixel size, refractive index, wavelength, numerical aperture, and intensity of the aberration.
    Attributes:
        size (int): Size of the PSF image (default: 100).
        dxy (float): Pixel size in the XY plane (default: 0.05).
        dz (float): Pixel size in the Z direction (default: 0.05).
        ni0 (float): Refractive index of the immersion medium (default: 1.515).
        ni (float): Refractive index of the sample medium (default: 1.515).
        wvl (float): Wavelength of light used for imaging (default: 0.5).
        NA (float): Numerical aperture of the objective lens (default: 1.4).
        psf (np.ndarray): Generated PSF image with astigmatism aberration.
    """
    def __init__(
        self, size=100, dxy=0.05, dz=0.05, ni0=1.515, ni=1.515, wvl=0.5, NA=1.4
    ):
        super(PSFWithAstigmatismAberration, self).__init__(
            size, dxy, dz, ni0, ni, wvl, NA
        )
        self.psf : np.ndarray = self.generate_astigmatism_aberration()

    def generate_astigmatism_aberration(self):
        """Generates the Point Spread Function (PSF) with astigmatism aberration based on the specified parameters using the psfmodels library and applies an astigmatic distortion to the PSF.
        Returns:
            np.ndarray: The generated PSF image with astigmatism aberration.
        """
        randomIntensity = np.random.uniform(0.02, 0.08)
        shape = (self.size, self.size, self.size)
        z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
        zC, yC, xC = np.array(shape) // 2
        psf_base = np.asarray(self.psf)
        stretchFactor = ((z - zC)) * randomIntensity
        xScale = (x - xC) * np.exp(stretchFactor)
        yScale = (y - yC) * np.exp(-stretchFactor)
        xRenseigne = xC + xScale
        yRenseigne = yC + yScale
        psf_astigmatism = map_coordinates(
            psf_base, [z, yRenseigne, xRenseigne], order=1
        )
        return psf_astigmatism


class PSFWithSphericalAberration(PSFGenerator):
    """Class for generating a Point Spread Function (PSF) with spherical aberration based on specified parameters such as size, pixel size, refractive index, wavelength, and numerical aperture.
    Attributes:
        size (int): Size of the PSF image (default: 100).
        dxy (float): Pixel size in the XY plane (default: 0.05).
        dz (float): Pixel size in the Z direction (default: 0.05).
        ni0 (float): Refractive index of the immersion medium (default: 1.515).
        ni (float): Refractive index of the sample medium (default: 1.515).
        wvl (float): Wavelength of light used for imaging (default: 0.5).
        NA (float): Numerical aperture of the objective lens (default: 1.4).
        psf (np.ndarray): Generated PSF image with spherical aberration.
    """
    def __init__(self, size=100, dxy=0.05, dz=0.05, ni0=1.515, wvl=0.5, NA=1.4):
        self.size : int = size
        self.dxy : float = dxy
        self.dz : float = dz
        self.ni0 : float = ni0
        self.ni : float = ni0 - np.random.uniform(0.015, 0.02)
        self.wvl : float = wvl
        self.NA : float = NA
        self.psf : np.ndarray = self.generate_psf()
        self.fwhm : float = self.computeFWHM()


class PSFRandomParameter(object):
    """Class for generating a Point Spread Function (PSF) with random parameters and optional aberration types.
    This class allows for the creation of a PSF with random values for pixel size, refractive index, wavelength, and numerical aperture, as well as the option to apply specific aberration types such as comatic, astigmatism, or spherical aberrations.
    Attributes:
        randomDxy (float): Pixel size in the XY plane (default: 0.05).
        randomDz (float): Pixel size in the Z direction (default: 0.05).
        randomNi (float): Refractive index of the sample medium (default: 1.515).
        randomWvl (float): Wavelength of light used for imaging (default: 0.5).
        randomNA (float): Numerical aperture of the objective lens (default: 1.4).
    
    Arguments:
        size (int): Size of the PSF image (default: 100).
        aberrationType (str): Type of aberration to apply to the PSF. Options include "comatic", "astigmatism", "spherical", or None for no aberration (default: None).
    """
    def __new__(cls, size=100, aberrationType=None):
        randomDxy : float = np.random.uniform(0.03, 0.07)
        randomDz : float = np.random.uniform(0.03, 0.07)
        randomNi : float = np.random.uniform(1.48, 1.52)
        randomWvl : float = np.random.uniform(0.45, 0.55)
        randomNA : float = np.random.uniform(1.2, 1.4)
        if aberrationType == "comatic":
            return PSFWithComaticAberration(
                size, randomDxy, randomDz, randomNi, randomNi, randomWvl, randomNA
            )
        elif aberrationType == "astigmatism":
            return PSFWithAstigmatismAberration(
                size, randomDxy, randomDz, randomNi, randomNi, randomWvl, randomNA
            )
        elif aberrationType == "spherical":
            return PSFWithSphericalAberration(
                size, randomDxy, randomDz, randomNi, randomWvl, randomNA
            )
        else:
            return PSFGenerator(
                size, randomDxy, randomDz, randomNi, randomNi, randomWvl, randomNA
            )
