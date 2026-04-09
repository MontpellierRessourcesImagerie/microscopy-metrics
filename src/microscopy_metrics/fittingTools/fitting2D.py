import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.fittingTools.fitting1D import Fitting1D
from microscopy_metrics.utils import pxToUm


class Fitting2D(FittingTool):
    """Class for fitting a 2D Gaussian curve to the PSF profile of a microscopy image.
    This class inherits from the FittingTool base class and implements methods specific to 2D Gaussian fitting.
    It includes methods for evaluating the Gaussian function, fitting the curve to the data, plotting the results, and calculating the coefficient of determination (R²) for the fit.
    """

    name = "2D"

    def __init__(self):
        super().__init__()

    def gauss(self, amp: float, bg: float, muX: float, muY: float, sigmaX: float, sigmaY: float):
        """Generates a 2D Gaussian function based on the provided parameters.
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY (float): center coordinates of the Gaussian
            sigmaX,sigmaY (float): standard deviation of the Gaussian

        Returns:
            float: Intensity value at (x,y) following the curve
        """

        def fun(coords):
            exponent = -((coords[:, 0] - muX) ** 2) / (2 * sigmaX**2) - (
                coords[:, 1] - muY
            ) ** 2 / (2 * sigmaY**2)
            return bg + (amp - bg) * np.exp(exponent)

        return fun

    def evalFun(self, x: np.ndarray, amp: float, bg: float, muX: float, muY: float, sigmaX: float, sigmaY: float):
        """Evaluates the 2D Gaussian function at the given coordinates.
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY (float): center coordinates of the Gaussian
            sigmaX,sigmaY (float): standard deviation of the Gaussian

        Returns:
            float: Intensity value at (x,y) following the curve
        """
        return self.gauss(
            amp=amp, bg=bg, muX=muX, muY=muY, sigmaX=sigmaX, sigmaY=sigmaY
        )(x)

    def fitCurve(self, amp: float, bg: float, mu: list, sigma: list, coords: np.ndarray, psf: np.ndarray):
        """Fits a 2D Gaussian curve to the provided PSF data using the initial parameters and coordinates.
        Args:
            amp (float): amplitude of the Gaussian
            bg (float): background intensity
            mu (List(float)): center of the Gaussian
            sigma (List(float)): standard deviation of the Gaussian
            coords (np.array(float)): List of X,Y coordinates
            psf (np.ndarray): 1D image of the flatten 2D psf

        Returns:
            List(float),Matrix(float): List of fitted parameters and covariance matrix
        """
        params = [amp, bg, *mu, *sigma]
        popt, pcov = curve_fit(
            self.evalFun,
            coords,
            psf.ravel(),
            p0=params,
            maxfev=5000,
            bounds=(
                [0, -np.inf, 0, 0, 1e-6, 1e-6],
                [
                    np.inf,
                    np.inf,
                    psf.shape[0],
                    psf.shape[1],
                    psf.shape[0],
                    psf.shape[1],
                ],
            ),
        )
        return popt, pcov

    def show2dFit(self, psf: np.ndarray, outputPath: str, params: list):
        """Generates and saves a visualization of the 2D Gaussian fit compared to the original PSF data.

        Args:
            psf (np.ndarray): The original PSF data.
            outputPath (str): The path where the visualization will be saved.
            params (List(float)): The fitted parameters for the 2D Gaussian.
        """
        yy_fine = np.linspace(0, psf.shape[0] - 1, psf.shape[0] * 10)
        xx_fine = np.linspace(0, psf.shape[1] - 1, psf.shape[1] * 10)
        y_fine, x_fine = np.meshgrid(yy_fine, xx_fine, indexing="ij")
        fine_coords_yx = np.stack([y_fine.ravel(), x_fine.ravel()], -1)
        fit = self.gauss(*params)(fine_coords_yx)
        fit = fit.reshape((psf.shape[0] * 10, psf.shape[1] * 10))
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(psf, cmap="viridis")
        ax1.set_title("PSF Data")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(fit, cmap="viridis")
        ax2.set_title("Fit")
        plt.tight_layout()
        fig.savefig(outputPath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plotSingleFit(self, psf: np.ndarray, fineCoords: np.ndarray, fit2D: np.ndarray, outputPath: str, index: int):
        """Generates and saves a visualization comparing the 1D profile of the original PSF data with the 1D profile of the 2D Gaussian fit along a specified axis.
        Args:
            psf (np.ndarray): The original PSF data.
            fineCoords (np.ndarray): The coordinates for the fine grid.
            fit2D (np.ndarray): The 2D Gaussian fit.
            outputPath (str): The path where the visualization will be saved.
            index (int): The index of the axis along which to compare profiles.
        """
        coords = np.arange(self._image.shape[index])
        params = [
            self.params1D[0],
            self.params1D[1],
            self.params1D[2 + index],
            self.params1D[5 + index],
        ]
        fit1D = Fitting1D().gauss(*params)(fineCoords)
        yLim = [0.0, psf.max() * 1.1]
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(coords, psf, "-", label="PSF", color="k")
        ax1.scatter(coords, psf, color="k", alpha=0.5, label="measurement points")
        ax1.plot(fineCoords, fit1D, label="1D Curve Fit")
        halfMax = (fit1D.max() + fit1D.min()) / 2.0
        ax1.axhline(
            y=halfMax, color="g", linestyle="--", alpha=0.7, label="FWHM 1D fit"
        )
        ax1.plot(fineCoords, fit2D, label="2D Curve Fit")
        halfMax = (fit2D.max() + fit2D.min()) / 2.0
        ax1.axhline(
            y=halfMax, color="r", linestyle="--", alpha=0.7, label="FWHM 2D fit"
        )
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
            label=f"Mu: {self.parameters[2+index]:.2f}",
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

    def plotFit3d(self, outputPath: str):
        """Generates and saves a visualization of the 3D Gaussian fit compared to the original PSF data.
        Args:
            outputPath (str): The path where the visualization will be saved.
        """
        center = self.getLocalCentroid()
        psf = self._image.astype(np.float64)
        p = self.parameters
        params2D = [
            [p[0], p[1], p[2], p[3], p[5], p[6]],
            [p[0], p[1], p[3], p[4], p[6], p[7]],
            [p[0], p[1], p[2], p[4], p[5], p[7]],
        ]
        psfs = [
            psf[:, center[1], center[2]],
            psf[center[0], :, center[2]],
            psf[center[0], center[1], :],
        ]
        for i in range(3):
            fine = np.linspace(0, psf.shape[i] - 1, 500)
            if i < 2:
                index = i + 1
                fineCoords = np.column_stack((fine, np.full_like(fine, center[index])))
            else:
                index = 0
                fineCoords = np.column_stack((np.full_like(fine, center[index]), fine))
            self.plotSingleFit(
                psfs[i], fine, self.gauss(*params2D[i])(fineCoords), outputPath, i
            )

    def getCoords(self, psf: np.ndarray):
        """Generates an array of coordinates corresponding to the length of the PSF profile along two axes.
        Args:
            psf (np.ndarray): 2D image to process

        Returns:
            np.ndarray: Coordinates for the 2D fit
        """
        y, x = np.indices(psf.shape)
        return np.stack([y.ravel(), x.ravel()], axis=-1)

    def processSingleFit(self, index: int):
        """Processes a single fit for the PSF data at the specified index, performing a 2D Gaussian fit and calculating relevant metrics.
        Args:
            index (int): The index of the PSF data to be processed.
        """
        imageFloat = self._image.astype(np.float64)
        physic = self.getLocalCentroid()
        psf = [
            imageFloat[:, :, physic[2]],
            imageFloat[physic[0], :, :],
            imageFloat[:, physic[1], :],
        ]
        axe = ["ZY", "YX", "XZ"]
        self._coords = [
            self.getCoords(psf[0]),
            self.getCoords(psf[1]),
            self.getCoords(psf[2]),
        ]
        activePath = self.getActivePath(index)
        self.compute1DParams()
        amp = self.params1D[0]
        bg = self.params1D[1]

        for u in range(3):
            u2 = (u + 1) % 3
            sigma = [self.params1D[5 + u], self.params1D[5 + u2]]
            mu = [self.params1D[2 + u], self.params1D[2 + u2]]

            params, pcov = self.fitCurve(amp, bg, mu, sigma, self._coords[u], psf[u])

            self.parameters[0] += params[0] / 3.0
            self.parameters[1] += params[1] / 3.0

            if u < 2:
                self.parameters[2 + u] += params[2] / 2.0
                self.parameters[2 + u2] += params[3] / 2.0
                self.parameters[5 + u] += params[4] / 2.0
                self.parameters[5 + u2] += params[5] / 2.0
                self.fwhms[u] += pxToUm(self.fwhm(params[4]), self._spacing[u]) / 2.0
                self.fwhms[u2] += pxToUm(self.fwhm(params[5]), self._spacing[u2]) / 2.0
            else:
                self.parameters[2 + u] += params[3] / 2.0
                self.parameters[2 + u2] += params[2] / 2.0
                self.parameters[5 + u] += params[5] / 2.0
                self.parameters[5 + u2] += params[4] / 2.0
                self.fwhms[u] += pxToUm(self.fwhm(params[5]), self._spacing[u]) / 2.0
                self.fwhms[u2] += pxToUm(self.fwhm(params[4]), self._spacing[u2]) / 2.0

            self.uncertainties[u] = self.uncertainty(pcov)
            self.determinations[u] = self.determination(
                params, self._coords[u], psf[u].flatten()
            )
            self.pcovs[u] = pcov
            outputPath = os.path.join(activePath, f"2D_Gaussian_Image_{axe[u]}.png")
            if self._show:
                self.show2dFit(psf[u], outputPath, params)
        if self._show:
            self.plotFit3d(activePath)

    def determination(self, params: list, coords: np.ndarray, psf: np.ndarray):
        """Calculates the coefficient of determination (R²) for the fitted curve against the original PSF data.
        Args:
            params (List[float]): Fitted parameters for the 2D Gaussian curve.
            coords (np.ndarray): Array of coordinates for the 2D fit.
            psf (np.ndarray): Original PSF data.

        Returns:
            float: The coefficient of determination (R²) for the fit.
        """
        psfFit = self.evalFun(coords, *params)
        return r2_score(psf, psfFit)
