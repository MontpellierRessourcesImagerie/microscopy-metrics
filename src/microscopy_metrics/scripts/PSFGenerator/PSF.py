import numpy as np
import psfmodels as psfm
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates


class PSFGenerator(object):
    def __init__(
        self, size=100, dxy=0.05, dz=0.05, ni0=1.515, ni=1.515, wvl=0.5, NA=1.4
    ):
        self.size = size
        self.dxy = dxy
        self.dz = dz
        self.ni0 = ni0
        self.ni = ni
        self.wvl = wvl
        self.NA = NA
        self.psf = self.generate_psf()
        self.fwhm = self.computeFWHM()

    def generate_psf(self):
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
        resXY = (0.51 * self.wvl * 100) / self.NA
        resZ = (1.77 * self.ni * self.wvl * 100) / (self.NA**2)
        return [resZ, resXY, resXY]

    def addNoise(self, mean=0, std=2.0):
        noise = np.random.normal(mean, std, size=self.psf.shape)
        self.psf += noise
        self.psf = np.clip(self.psf, 0, None)

    def showPSF(self):
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.psf[self.psf.shape[0] // 2], cmap="gray")
        ax2.imshow(self.psf[:, self.psf.shape[1] // 2], cmap="gray")
        plt.show()


class PSFWithComaticAberration(PSFGenerator):
    def __init__(
        self, size=100, dxy=0.05, dz=0.05, ni0=1.515, ni=1.515, wvl=0.5, NA=1.4, Intensity=None
    ):
        super(PSFWithComaticAberration, self).__init__(size, dxy, dz, ni0, ni, wvl, NA)
        self.intensity = Intensity if Intensity is not None else np.random.uniform(0.02, 0.08)
        self.psf = self.generate_comatic_aberration()

    def generate_comatic_aberration(self):
        shape = (self.size, self.size, self.size)
        z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
        zC, _, _ = np.array(shape) // 2
        psf_base = np.asarray(self.psf)
        xShift = ((z - zC) ** 2) * self.intensity
        xRenseigne = x - xShift
        psf_banane = map_coordinates(psf_base, [z, y, xRenseigne], order=1)
        return psf_banane


class PSFWithAstigmatismAberration(PSFGenerator):
    def __init__(
        self, size=100, dxy=0.05, dz=0.05, ni0=1.515, ni=1.515, wvl=0.5, NA=1.4
    ):
        super(PSFWithAstigmatismAberration, self).__init__(
            size, dxy, dz, ni0, ni, wvl, NA
        )
        self.psf = self.generate_astigmatism_aberration()

    def generate_astigmatism_aberration(self):
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
    def __init__(self, size=100, dxy=0.05, dz=0.05, ni0=1.515, wvl=0.5, NA=1.4):
        self.size = size
        self.dxy = dxy
        self.dz = dz
        self.ni0 = ni0
        self.ni = ni0 - np.random.uniform(0.015, 0.02)
        self.wvl = wvl
        self.NA = NA
        self.psf = self.generate_psf()
        self.fwhm = self.computeFWHM()


class PSFRandomParameter(object):
    def __new__(cls, size=100, aberrationType=None):
        randomDxy = np.random.uniform(0.03, 0.07)
        randomDz = np.random.uniform(0.03, 0.07)
        randomNi = np.random.uniform(1.48, 1.52)
        randomWvl = np.random.uniform(0.45, 0.55)
        randomNA = np.random.uniform(1.2, 1.4)
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
