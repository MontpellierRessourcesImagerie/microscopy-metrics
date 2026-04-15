import os
import numpy as np
import math
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit

from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.fittingTools.fitting1D import Fitting1D
from microscopy_metrics.utils import pxToUm


class Fitting2DEllips(FittingTool):
    """Class for fitting a 2D Ellipse Gaussian curve to the PSF profile of a microscopy image.
    This class inherits from the FittingTool base class and implements methods specific to 2D Ellipse Gaussian fitting.
    It includes methods for evaluating the 2D Ellipse Gaussian function, fitting the curve to the data, and plotting the results.
    """

    name = "2D Ellipse"

    def __init__(self):
        super().__init__()
        self.thetas = [0, 0, 0]

    def gauss(
        self,
        amp: float,
        bg: float,
        muX: float,
        muY: float,
        a: float,
        b: float,
        c: float,
    ):
        """Generates a 2D Ellipse Gaussian function based on the provided parameters.
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY (float): center coordinates of the Gaussian
            a,b,c (float): coefficients for the ellipse
        Returns:
            float: Intensity value at (x,y) following the curve
        """

        def fun(coords: np.ndarray):
            exponent = (
                (a * ((coords[:, 0] - muX) ** 2))
                + (c * ((coords[:, 1] - muY) ** 2))
                + (2.0 * b * (coords[:, 0] - muX) * (coords[:, 1] - muY))
            )
            return bg + (amp - bg) * np.exp(-exponent)

        return fun

    def evalFun(
        self,
        x: np.ndarray,
        amp: float,
        bg: float,
        muX: float,
        muY: float,
        a: float,
        b: float,
        c: float,
    ):
        """calculates the 2D Ellipse Gaussian function at the given x values.
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY (float): center coordinates of the Gaussian
            a,b,c (float): coefficients for the ellipse
        Returns:
            float: Intensity value at (x,y) following the curve
        """
        return self.gauss(amp=amp, bg=bg, muX=muX, muY=muY, a=a, b=b, c=c)(x)

    def fitCurve(
        self,
        amp: float,
        bg: float,
        mu: list,
        sigma: list,
        coords: np.ndarray,
        psf: np.ndarray,
    ):
        """Fits a 2D Ellipse Gaussian curve to the provided PSF data using the given initial parameters and coordinates.
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
                [0, -np.inf, 0, 0, 1e-6, -np.inf, 1e-6],
                [np.inf, np.inf, psf.shape[0], psf.shape[1], np.inf, np.inf, np.inf],
            ),
        )
        return popt, pcov

    def ellipseParmConversion(self, a: float, b: float, c: float):
        """Converts the parameters of a 2D Ellipse Gaussian fit (a, b, c) into the angle of rotation (theta) and the standard deviations along the major and minor axes (sx, sy).
        Args:
            a (float): coefficient for the ellipse linked to the x-axis
            b (float): coefficient for the ellipse linked to the interaction between x and y
            c (float): coefficient for the ellipse linked to the y-axis
        Returns:
            tuple: The angle of rotation (theta) and the standard deviations along the major and minor axes (sx, sy)
        """
        t = math.sqrt(4.0 * (b**2) + ((a - c) ** 2))
        s = math.sqrt(abs((b**2) - (a * c)))
        sx = math.sqrt(abs(-a + t - c)) / 2.0 / s
        sy = math.sqrt(abs(-a - t - c)) / 2.0 / s
        theta = math.sqrt(1.0 + ((a - c) / t)) / math.sqrt(2.0)
        theta = math.acos(theta)
        theta_sign = 4 * b * (sx**2) * (sy**2) / ((sx**2) - (sy**2))
        theta_sign = max(-1.0, min(1.0, theta_sign))
        theta_sign = np.sign(-0.5 * math.asin(theta_sign))
        theta = ((math.pi / 2) - theta) * theta_sign
        return theta, sx, sy

    def show2dFit(self, psf: np.ndarray, outputPath: str, params: list, theta: float):
        """Generates and saves a plot comparing the original PSF data with the fitted 2D Ellipse Gaussian curve, including the angle of rotation (theta) and the center of the Gaussian.
        Args:
            psf (np.ndarray): The original PSF data.
            outputPath (str): The path where the visualization will be saved.
            params (List(float)): The fitted parameters for the 2D Ellipse Gaussian.
            theta (float): The angle of rotation for the ellipse.
        """
        yy_fine = np.linspace(0, psf.shape[0] - 1, psf.shape[0] * 10)
        xx_fine = np.linspace(0, psf.shape[1] - 1, psf.shape[1] * 10)
        y_fine, x_fine = np.meshgrid(yy_fine, xx_fine, indexing="ij")
        fine_coords_yx = np.stack([y_fine.ravel(), x_fine.ravel()], -1)
        y0 = params[2] * 10
        x0 = params[3] * 10
        L = min(max(psf.shape) * 10 / 4, 30)
        if params[4] > params[5]:
            dy = L * np.cos(theta)
            dx = -L * np.sin(theta)
        else:
            dy = L * np.sin(theta)
            dx = L * np.cos(theta)
        x1 = x0 + dx
        y1 = y0 + dy
        fit = self.gauss(*params)(fine_coords_yx)
        fit = fit.reshape((psf.shape[0] * 10, psf.shape[1] * 10))
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(psf, cmap="viridis")
        ax1.set_title("PSF Data")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(fit, cmap="viridis")
        ax2.set_title("Fit")
        ax2.plot([x0, x1], [y0, y1], color="red", linewidth=2)
        ax2.scatter([x0], [y0], color="red", alpha=0.7)
        ax2.axhline(y=psf.shape[0] * 10 / 2, color="k", alpha=0.5, linestyle="--")
        ax2.axvline(x=psf.shape[1] * 10 / 2, color="k", alpha=0.5, linestyle="--")
        plt.tight_layout()
        fig.savefig(outputPath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plotSingleFit(
        self,
        psf: np.ndarray,
        fineCoords: np.ndarray,
        fit2D: np.ndarray,
        outputPath: str,
        mu: float,
        index: int,
    ):
        """Generates and saves a plot comparing the original PSF data with the fitted 1D Gaussian curve and the 2D Ellipse Gaussian curve, including the center of the Gaussian (mu) and the FWHM for both fits.
        Args:
            psf (np.ndarray): The original PSF data.
            fineCoords (np.ndarray): The coordinates for the fine grid.
            fit2D (np.ndarray): The 2D Ellipse Gaussian fit.
            outputPath (str): The path where the visualization will be saved.
            mu (float): The center of the Gaussian.
            index (int): The index of the axis along which to compare profiles.
        """
        params = [
            self.params1D[0],
            self.params1D[1],
            self.params1D[2 + index],
            self.params1D[5 + index],
        ]
        fit1D = Fitting1D().gauss(*params)(fineCoords)
        coords = np.arange(self._image.shape[index])
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
            x=mu,
            color="purple",
            linestyle="dotted",
            alpha=0.5,
            label=f"Mu: {self.parameters[2 + index]:.2f}",
        )
        ax1.plot(mu, self.parameters[0], "ko")
        ax1.set_ylim(yLim)
        ax1.set_title(f"{self.axes[index]} Profile")
        ax1.legend()
        fig1.savefig(
            os.path.join(outputPath, f"fit_curve_1D_{self.axes[index]}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig1)

    def plotFit3d(self, popt: list, outputPath: str):
        """Generates and saves a 3D plot comparing the original PSF data with the fitted 2D Ellipse Gaussian curve, including the center of the Gaussian (mu) and the angle of rotation (theta).
        Args:
            popt (List(List(float))): The fitted parameters for the 2D Ellipse Gaussian.
            outputPath (str): The path where the visualization will be saved.
        """
        psf = self._image.astype(np.float64)
        center = self.getLocalCentroid()
        psfs = [
            psf[:, center[1], center[2]],
            psf[center[0], :, center[2]],
            psf[center[0], center[1], :],
        ]
        for i in range(3):
            fine = np.linspace(0, psf.shape[i] - 1, 100)
            if i < 2:
                index = i + 1
                fineCoords = np.column_stack((fine, np.full_like(fine, center[index])))
                mu = popt[i][2]
            else:
                index = 0
                fineCoords = np.column_stack((np.full_like(fine, center[index]), fine))
                mu = popt[i][3]
            self.plotSingleFit(
                psfs[i], fine, self.gauss(*popt[i])(fineCoords), outputPath, mu, i
            )

    def getCoords(self, psf: np.ndarray):
        """Generates a list of coordinates for the 2D fit based on the shape of the provided PSF image.
        Args:
            psf (np.ndarray): The PSF image for which to generate the coordinates.

        Returns:
            np.ndarray: A list of coordinates corresponding to the shape of the PSF image, suitable for use in the 2D fitting process.
        """
        y, x = np.indices(psf.shape)
        return np.stack([y.ravel(), x.ravel()], axis=-1)

    def processSingleFit(self, index: int):
        """Processes a single fit for the given index, performing fitting, and plotting.
        Args:
            index (int): ID of the PSF and position in lists
        """
        imageFloat = self._image.astype(np.float64)
        physic = self.getLocalCentroid()
        psf = [
            imageFloat[:, :, physic[2]],
            imageFloat[physic[0], :, :],
            imageFloat[:, physic[1], :],
        ]
        axe = ["ZY", "YX", "XZ"]
        coords = [
            self.getCoords(psf[0]),
            self.getCoords(psf[1]),
            self.getCoords(psf[2]),
        ]
        activePath = self.getActivePath(index)
        self.compute1DParams()
        params2D = []
        amp = self.params1D[0]
        bg = self.params1D[1]

        for u in range(3):
            u2 = (u + 1) % 3
            if u < 2:
                mu = [self.params1D[2 + u], self.params1D[2 + u2]]
            else:
                mu = [self.params1D[u2 + 2], self.params1D[u + 2]]
            sigma = [1, 0, 1]

            params, pcov = self.fitCurve(amp, bg, mu, sigma, coords[u], psf[u])

            params2D.append(params)
            theta, s1, s2 = self.ellipseParmConversion(params[4], params[5], params[6])
            self.thetas[u] = theta
            self.parameters[0] += params[0] / 3.0
            self.parameters[1] += params[1] / 3.0

            if u < 2:
                self.parameters[2 + u] += params[2] / 2.0
                self.parameters[2 + u2] += params[3] / 2.0
                self.parameters[5 + u] += s1 / 2.0
                self.parameters[5 + u2] += s2 / 2.0
                self.fwhms[u] += pxToUm(self.fwhm(s1), self._spacing[u]) / 2.0
                self.fwhms[u2] += pxToUm(self.fwhm(s2), self._spacing[u2]) / 2.0
            else:
                self.parameters[2 + u] += params[3] / 2.0
                self.parameters[2 + u2] += params[2] / 2.0
                self.parameters[5 + u] += s2 / 2.0
                self.parameters[5 + u2] += s1 / 2.0
                self.fwhms[u] += pxToUm(self.fwhm(s1), self._spacing[u]) / 2.0
                self.fwhms[u2] += pxToUm(self.fwhm(s2), self._spacing[u2]) / 2.0

            self.uncertainties[u] = self.uncertainty(pcov)
            self.determinations[u] = self.determination(
                params, coords[u], psf[u].flatten()
            )
            self.pcovs[u] = pcov
            outputPath = os.path.join(activePath, f"2D_Gaussian_Image_{axe[u]}.png")
            if self._show:
                self.show2dFit(psf[u], outputPath, params, theta)
        if self._show:
            self.plotFit3d(params2D, activePath)
            for i in range(3):
                print(f"mean angle {axe[i]}: {math.degrees(self.thetas[i])}")
