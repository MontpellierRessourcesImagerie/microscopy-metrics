import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.utils import pxToUm


class Fitting1D(FittingTool):
    """Class for fitting a 1D Gaussian curve to the PSF profile of a microscopy image.
    This class inherits from the FittingTool base class and implements methods specific to 1D Gaussian fitting.
    It includes methods for evaluating the Gaussian function, fitting the curve to the data, and plotting the results.
    """

    name = "1D"
    axes = ["Z", "Y", "X"]

    def __init__(self):
        super().__init__()

    def getCovMatrix(self, image: np.ndarray, centroid: list):
        """Calculates the covariance matrix for a 1D image based on the provided centroid.
        Args:
            image (np.ndarray): The 1D image data for which to calculate the covariance matrix.
            centroid (List(float)): The centroid of the image, used as a reference point for the covariance calculation.
        """
        if image.ndim != 1:
            raise NotImplementedError("getCovMatrix is only implemented for 1D images")
        x = np.arange(image.shape[0]) - centroid[0]
        return np.sqrt(np.sum(x * x * image) / np.sum(image))

    def gauss(self, amp: float, bg: float, mu: float, sigma: float):
        """Generates a 1D Gaussian function based on the provided parameters.
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            mu (float): center of the curve
            sigma (float): standard deviation of the curve
        Returns:
            float: Intensity value at x following the curve
        """
        return lambda x: bg + (amp - bg) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    def evalFun(self, x: np.ndarray, amp: float, bg: float, mu: float, sigma: float):
        """Evaluates the 1D Gaussian function at the given x values.
        Args:
            x (array): x values at which to evaluate the function
            amp (float): amplitude of the curve
            bg (float): background intensity
            mu (float): center of the curve
            sigma (float): standard deviation of the curve
        Returns:
            float: Intensity value at x following the curve
        """
        return self.gauss(amp=amp, bg=bg, mu=mu, sigma=sigma)(x)

    def fitCurve(
        self,
        amp: float,
        bg: float,
        mu: float,
        sigma: float,
        coords: np.ndarray,
        psf: np.ndarray,
    ):
        """
        Fits a 1D Gaussian curve to the given data.
        Args:
            amp (float): Initial guess for the amplitude.
            bg (float): Initial guess for the background.
            mu (float): Initial guess for the center.
            sigma (float): Initial guess for the standard deviation.
            coords (array): Coordinates of the data points.
            psf (array): Intensity values at the data points.
        Returns:
            tuple: Optimal parameters and covariance matrix.
        """
        params = [amp, bg, mu, sigma]
        popt, pcov = curve_fit(
            self.evalFun,
            coords,
            psf,
            p0=params,
            maxfev=5000,
            bounds=([0, -np.inf, 0, 1e-6], [np.inf, np.inf, len(coords), np.inf]),
        )
        return popt, pcov

    def plotSingleFit(
        self, psf: np.ndarray, fineCoords: np.ndarray, outputPath: str, index: int
    ):
        """Plots the original data points, the fitted curve, and key parameters.
        Args:
            coords (array): Coordinates of the data points.
            psf (array): Intensity values at the data points.
            fineCoords (array): Coordinates for plotting the fitted curve.
            outputPath (str): Directory where the plot will be saved.
            index (int): Index of the axis being plotted (0 for Z, 1 for Y, 2 for X).
        """
        fit = self.gauss(
            self.parameters[0],
            self.parameters[1],
            self.parameters[2 + index],
            self.parameters[5 + index],
        )(fineCoords)
        yLim = [0.0, psf.max() * 1.1]
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(self.coords[index], psf, "-", label="PSF", color="k")
        ax1.scatter(
            self.coords[index], psf, color="k", alpha=0.5, label="measurement points"
        )
        ax1.plot(fineCoords, fit, label="1D Curve Fit")
        halfMax = (fit.max() + fit.min()) / 2.0
        ax1.axhline(y=halfMax, color="r", linestyle="--", alpha=0.7, label="FWHM")
        ax1.axhline(
            y=self.parameters[0],
            color="orange",
            linestyle="dotted",
            alpha=0.5,
            label=f"Amplitude: {self.parameters[0]:.2f}",
        )
        ax1.axvline(
            x=self.parameters[2 + index],
            color="purple",
            linestyle="dotted",
            alpha=0.5,
            label=f"Mu: {self.parameters[2 + index]:.2f}",
        )
        ax1.axhline(
            y=self.parameters[1],
            color="black",
            linestyle="--",
            alpha=0.5,
            label=f"BG: {self.parameters[1]:.2f}",
        )
        ax1.plot(self.parameters[2 + index], self.parameters[0], "ko")
        ax1.set_ylim(yLim)
        ax1.set_title(f"{self.axes[index]} Profile")
        ax1.legend()
        fig1.savefig(
            os.path.join(outputPath, f"fit_curve_1D_{self.axes[index]}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig1)

    def plotFit1d(self, outputPath: str):
        """Plots the fitted curves for all three axes.
        Args:
            outputPath (str): Directory where the plots will be saved.
        """
        psf = self._image
        center = self.getLocalCentroid()
        psfs = [
            psf[:, center[1], center[2]],
            psf[center[0], :, center[2]],
            psf[center[0], center[1], :],
        ]
        for i in range(3):
            fine = np.linspace(0, psf.shape[i] - 1, 500)
            self.plotSingleFit(psfs[i], fine, outputPath, i)

    def getCoords(self, psf: np.ndarray):
        """Generates an array of coordinates corresponding to the length of the PSF profile.
        Args:
            psf (array): Intensity values of the PSF profile.
        Returns:
            array: Coordinates corresponding to the PSF profile.
        """
        return np.arange(psf.shape[0])

    def processSingleFit(self, index: int):
        """Processes a single fit for the given index, performing fitting, and plotting.
        Args:
            index (int): ID of the psf.
        """
        imageFloat = self._image.astype(np.float64)
        activePath = self.getActivePath(index)
        physic = self.getLocalCentroid()
        psf = [
            imageFloat[:, physic[1], physic[2]],
            imageFloat[physic[0], :, physic[2]],
            imageFloat[physic[0], physic[1], :],
        ]
        self.coords = [
            self.getCoords(psf[0]),
            self.getCoords(psf[1]),
            self.getCoords(psf[2]),
        ]
        self.parameters[0] = 0.0
        self.parameters[1] = 0.0
        for u in range(3):
            bg = np.median(psf[u])
            amp = psf[u].max()
            sigma = self.getCovMatrix(psf[u], physic)
            mu = np.argmax(psf[u])
            params, pcov = self.fitCurve(amp, bg, mu, sigma, self.coords[u], psf[u])
            self.fwhms[u] = pxToUm(self.fwhm(params[3]), self._spacing[u])
            self.uncertainties[u] = self.uncertainty(pcov)
            self.determinations[u] = self.determination(params, self.coords[u], psf[u])
            self.parameters[0] += params[0] / 3.0
            self.parameters[1] += params[1] / 3.0
            self.parameters[2 + u] = params[2]
            self.parameters[5 + u] = params[3]
            self.pcovs[u] = pcov
        if self._show:
            self.plotFit1d(activePath)
