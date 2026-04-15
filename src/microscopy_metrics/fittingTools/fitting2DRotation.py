import os
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit

from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.fittingTools.fitting1D import Fitting1D
from microscopy_metrics.utils import pxToUm


class Fitting2DRotation(FittingTool):
    """Class for fitting a 2D Gaussian curve with rotation to the PSF profile of a microscopy image.
    This class inherits from the FittingTool base class and implements methods specific to 2D Gaussian fitting with rotation.
    It includes methods for evaluating the Gaussian function, fitting the curve to the data, and plotting the results.
    """

    name = "2D rotation"

    def __init__(self):
        super().__init__()
        self.thetas = [0, 0, 0]

    def gauss(self, amp: float, bg: float, muX: float, muY: float, sigmaX: float, sigmaY: float, theta: float):
        """Generates a 2D Gaussian function with rotation based on the provided parameters.
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY (float): center coordinates of the Gaussian
            sigmaX,sigmaY (float): standard deviation of the Gaussian
            theta (float): rotation angle in radians
        Returns:
            float: Intensity value at (x,y) following the curve
        """

        def fun(coords):
            x = coords[:, 0]
            y = coords[:, 1]
            x_rot = (x - muX) * np.cos(theta) - (y - muY) * np.sin(theta)
            y_rot = (x - muX) * np.sin(theta) + (y - muY) * np.cos(theta)
            exponent = -((x_rot**2) / (2 * sigmaX**2) + (y_rot**2) / (2 * sigmaY**2))
            return bg + (amp - bg) * np.exp(exponent)

        return fun

    def evalFun(self, x, amp, bg, muX, muY, sigmaX, sigmaY, theta):
        """Evaluates the 2D Gaussian function with rotation at the given x values.
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY (float): center coordinates of the Gaussian
            sigmaX,sigmaY (float): standard deviation of the Gaussian
            theta (float): rotation angle in radians
        Returns:
            float: Intensity value at (x,y) following the curve
        """
        return self.gauss(
            amp=amp, bg=bg, muX=muX, muY=muY, sigmaX=sigmaX, sigmaY=sigmaY, theta=theta
        )(x)

    def fitCurve(self, amp, bg, mu, sigma, theta, coords, psf):
        """Fits the 2D Gaussian function with rotation to the provided PSF data using curve fitting.
        Args:
            amp (float): amplitude of the Gaussian
            bg (float): background intensity
            mu (List(float)): center of the Gaussian
            sigma (List(float)): standard deviation of the Gaussian
            theta (float): rotation angle in radians
            coords (np.array(float)): List of X,Y coordinates
            psf (np.ndarray): 1D image of the flatten 2D psf
        Returns:
            List(float),Matrix(float): List of fitted parameters and covariance matrix
        """
        params = [amp, bg, *mu, *sigma, theta]
        popt, pcov = curve_fit(
            self.evalFun,
            coords,
            psf.ravel(),
            p0=params,
            maxfev=5000,
            bounds=(
                [0, -np.inf, 0, 0, 1e-6, 1e-6, -np.pi],
                [
                    np.inf,
                    np.inf,
                    psf.shape[0],
                    psf.shape[1],
                    psf.shape[0],
                    psf.shape[1],
                    np.pi,
                ],
            ),
        )
        return popt, pcov

    def show2dFit(self, psf, params, outputPath):
        """Generates a visual representation of the fitted 2D Gaussian curve with rotation compared to the original PSF data.
        Args:
            psf (np.ndarray): The original PSF data to compare against the fitted curve.
            params (List(float)): The parameters of the fitted 2D Gaussian curve, including amplitude, background, center coordinates, standard deviations, and rotation angle.
            outputPath (str): The directory where the generated plot will be saved.
        """
        yy_fine = np.linspace(0, psf.shape[0] - 1, psf.shape[0] * 10)
        xx_fine = np.linspace(0, psf.shape[1] - 1, psf.shape[1] * 10)
        y_fine, x_fine = np.meshgrid(yy_fine, xx_fine, indexing="ij")
        fine_coords_yx = np.stack([y_fine.ravel(), x_fine.ravel()], -1)
        y0 = params[2] * 10
        x0 = params[3] * 10
        L = min(max(psf.shape) * 10 / 4, 30)
        theta = params[6]
        if params[4] > params[5]:
            dy = L * np.cos(theta)
            dx = -L * np.sin(theta)
        else:
            dy = L * np.sin(theta)
            dx = L * np.cos(theta)
        display_angle = np.degrees(np.arctan2(dy, dx))
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
        ax2.set_title(f"Fit (Angle: {display_angle:.1f}°)")
        ax2.plot([x0 - dx, x1], [y0 - dy, y1], color="red", linewidth=2)
        ax2.scatter([x0], [y0], color="red", alpha=0.7)
        ax2.axhline(y=psf.shape[0] * 10 / 2, color="k", alpha=0.5, linestyle="--")
        ax2.axvline(x=psf.shape[1] * 10 / 2, color="k", alpha=0.5, linestyle="--")
        plt.tight_layout()
        fig.savefig(outputPath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plotSingleFit(self, coords, psf, fineCoords, fit2D, outputPath, index):
        """Plots the original data points, the fitted curve, and key parameters for a single axis.
        Args:
            coords (array): Coordinates of the data points.
            psf (array): Intensity values at the data points.
            fineCoords (array): Coordinates for plotting the fitted curve.
            fit2D (array): Intensity values of the 2D fit at the fine coordinates.
            outputPath (str): Directory where the plot will be saved.
            index (int): Index of the axis being plotted (0 for Z, 1 for Y, 2 for X).
        """
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

    def plotFit3d(self, outputPath):
        """Plots the fitted 2D Gaussian curve in 3D.
        Args:
            outputPath (str): Directory where the plot will be saved.
        """
        psf = self._image.astype(np.float64)
        p = self.parameters
        params2D = [
            [p[0], p[1], p[2], p[3], p[5], p[6], self.thetas[0]],
            [p[0], p[1], p[3], p[4], p[6], p[7], self.thetas[1]],
            [p[0], p[1], p[2], p[4], p[5], p[7], self.thetas[2]],
        ]
        center = self.getLocalCentroid()
        psfs = [
            psf[:, center[1], center[2]],
            psf[center[0], :, center[2]],
            psf[center[0], center[1], :],
        ]
        for i in range(3):
            coords = np.arange(psf.shape[i])
            fine = np.linspace(0, psf.shape[i] - 1, 100)
            if i < 2:
                index = i + 1
                fineCoords = np.column_stack((fine, np.full_like(fine, center[index])))
            else:
                index = 0
                fineCoords = np.column_stack((np.full_like(fine, center[index]), fine))
            self.plotSingleFit(
                coords,
                psfs[i],
                fine,
                self.gauss(*params2D[i])(fineCoords),
                outputPath,
                i,
            )

    def getCoords(self, psf: np.ndarray):
        """Generates an array of coordinates corresponding to the shape of the PSF image, suitable for use in the 2D fitting process.
        Args:
            psf (np.ndarray): 2D image to process for coordinate generation
        Returns:
            np.ndarray: Coordinates for the 2D fit
        """
        y, x = np.indices(psf.shape)
        return np.stack([y.ravel(), x.ravel()], axis=-1)

    def processSingleFit(self, index):
        """Processes a single fit for the given index, performing fitting, and plotting.
        Args:
            index (int): ID of the PSF and position in lists where results are stored.
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
        visual_angles = []
        amp = self.params1D[0]
        bg = self.params1D[1]
        for u in range(3):
            u2 = (u + 1) % 3
            if u < 2:
                sigma = [self.params1D[5 + u], self.params1D[5 + u2]]
                mu = [self.params1D[2 + u], self.params1D[2 + u2]]
            else:
                sigma = [self.params1D[5 + u2], self.params1D[5 + u]]
                mu = [self.params1D[2 + u2], self.params1D[2 + u]]
            params, pcov = self.fitCurve(amp, bg, mu, sigma, 0, coords[u], psf[u])
            self.parameters[0] += params[0] / 3.0
            self.parameters[1] += params[1] / 3.0
            self.thetas[u] = params[6]
            if params[4] > params[5]:
                angle_app = math.degrees(
                    math.atan2(np.cos(params[6]), -np.sin(params[6]))
                )
            else:
                angle_app = math.degrees(
                    math.atan2(np.sin(params[6]), np.cos(params[6]))
                )
            visual_angles.append(angle_app)

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
                params, coords[u], psf[u].flatten()
            )
            self.pcovs[u] = pcov
            outputPath = os.path.join(activePath, f"2D_Gaussian_Image_{axe[u]}.png")
            if self._show:
                self.show2dFit(psf[u], params, outputPath)
        if self._show:
            self.plotFit3d(activePath)
            for i in range(3):
                print(
                    f"Plane {axe[i]} | Local Angle: {math.degrees(self.thetas[i]):.2f}° => Major axis: {visual_angles[i]:.2f}°"
                )
