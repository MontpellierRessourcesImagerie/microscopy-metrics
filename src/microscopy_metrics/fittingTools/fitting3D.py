import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit

from microscopy_metrics.utils import pxToUm
from microscopy_metrics.fittingTools.fitting1D import Fitting1D
from microscopy_metrics.fittingTools.fittingTool import FittingTool


class Fitting3D(FittingTool):
    """Class for fitting a 3D Gaussian curve to the PSF profile of a microscopy image.
    This class inherits from the FittingTool base class and implements methods specific to 3D Gaussian fitting.
    It includes methods for evaluating the Gaussian function, fitting the curve to the data, and plotting the results.
    """

    name = "3D"

    def __init__(self):
        super().__init__()

    def gauss(
        self,
        amp: float,
        bg: float,
        muX: float,
        muY: float,
        muZ: float,
        sigmaX: float,
        sigmaY: float,
        sigmaZ: float,
    ):
        """Generates a 3D Gaussian function based on the provided parameters.
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY,muZ (float): center coordinates of the Gaussian
            sigmaX,sigmaY,sigmaZ (float): standard deviation of the Gaussian
        Returns:
            float: Intensity value at (x,y,z) following the curve
        """

        def fun(coords):
            exponent = (
                -((coords[:, 0] - muX) ** 2) / (2 * sigmaX**2)
                - (coords[:, 1] - muY) ** 2 / (2 * sigmaY**2)
                - (coords[:, 2] - muZ) ** 2 / (2 * sigmaZ**2)
            )
            return bg + (amp - bg) * np.exp(exponent)

        return fun

    def evalFun(
        self,
        x: np.ndarray,
        amp: float,
        bg: float,
        muX: float,
        muY: float,
        muZ: float,
        sigmaX: float,
        sigmaY: float,
        sigmaZ: float,
    ):
        """Evaluates the 3D Gaussian function at the given x values.
        Args:
            x (array): x values at which to evaluate the function
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY,muZ (float): center coordinates of the Gaussian
            sigmaX,sigmaY,sigmaZ (float): standard deviation of the Gaussian
        Returns:
            float: Intensity value at (x,y,z) following the curve
        """
        return self.gauss(
            amp=amp,
            bg=bg,
            muX=muX,
            muY=muY,
            muZ=muZ,
            sigmaX=sigmaX,
            sigmaY=sigmaY,
            sigmaZ=sigmaZ,
        )(x)

    def fitCurve(
        self,
        amp: float,
        bg: float,
        mu: list,
        sigma: list,
        coords: np.ndarray,
        psf: np.ndarray,
    ):
        """Fits a 3D Gaussian curve to the provided PSF data.
        Args:
            amp (float): amplitude of the Gaussian
            bg (float): background intensity
            mu (List(float)): center of the Gaussian
            sigma (List(float)): standard deviation of the Gaussian
            coords (np.array(float)): List of X,Y,Z coordinates
            psf (np.ndarray): 1D image of the flatten 3D psf
        Returns:
            List(float),Matrix(float): List of fitted parameters and covariance matrix
        """
        params = [amp, bg, *mu, *sigma]
        bounds = (
            [0, -np.inf, 0, 0, 0, 1e-6, 1e-6, 1e-6],
            [
                np.inf,
                np.inf,
                psf.shape[0],
                psf.shape[1],
                psf.shape[2],
                np.inf,
                np.inf,
                np.inf,
            ],
        )
        popt, pcov = curve_fit(
            self.evalFun,
            coords,
            psf.ravel(),
            p0=params,
            maxfev=5000,
            bounds=bounds,
        )
        return popt, pcov

    def show2dFit(self, psf: np.ndarray, outputPath: str):
        """Plots the 2D slices of the PSF data and the corresponding fitted Gaussian curves, and saves the plots to the specified output path.
        Args:
            psf (np.ndarray): 3D image of the PSF data
            outputPath (str): Path to the folder where the plots will be saved
        """
        center = self.getLocalCentroid()
        fitShapeZ = max(psf.shape[0] * 5, 256)
        fitShapeY = max(psf.shape[1] * 5, 128)
        fitShapeX = max(psf.shape[2] * 5, 128)
        z, y, x = np.indices((fitShapeZ, fitShapeY, fitShapeX))
        z_fine = z * (psf.shape[0] / fitShapeZ)
        y_fine = y * (psf.shape[1] / fitShapeY)
        x_fine = x * (psf.shape[2] / fitShapeX)
        fine_coords_zyx = np.stack([z_fine.ravel(), y_fine.ravel(), x_fine.ravel()], -1)
        fit = self.gauss(*self.parameters)(fine_coords_zyx).reshape(
            (fitShapeZ, fitShapeY, fitShapeX)
        )
        centerFitZ = int(fitShapeZ * (center[0] / psf.shape[0]))
        centerFitY = int(fitShapeY * (center[1] / psf.shape[1]))
        centerFitX = int(fitShapeX * (center[2] / psf.shape[2]))
        slices = [
            ("YX", psf[center[0], :, :], fit[centerFitZ, :, :]),
            ("XZ", psf[:, center[1], :], fit[:, centerFitY, :]),
            ("ZY", psf[:, :, center[2]], fit[:, :, centerFitX]),
        ]
        for name, psf_slice, fit_slice in slices:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(psf_slice, cmap="viridis")
            ax1.set_title("PSF Data")
            ax2.imshow(fit_slice, cmap="viridis")
            ax2.set_title("Fit")
            plt.tight_layout()
            fig.savefig(
                os.path.join(outputPath, f"2D_Gaussian_Image_{name}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    def plotSingleFit(
        self,
        coords: np.ndarray,
        psf: np.ndarray,
        fineCoords: np.ndarray,
        fit3D: np.ndarray,
        outputPath: str,
        index: int,
    ):
        """Plots the original data points, the fitted curve, and key parameters for a single axis.
        Args:
            coords (array): Coordinates of the data points.
            psf (array): Intensity values at the data points.
            fineCoords (array): Coordinates for plotting the fitted curve.
            fit3D (array): Intensity values of the 3D fit at the fine coordinates.
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
        ax1.plot(fineCoords, fit3D, label="3D Curve Fit")
        halfMax = (fit3D.max() + fit3D.min()) / 2.0
        ax1.axhline(
            y=halfMax, color="r", linestyle="--", alpha=0.7, label="FWHM 3D fit"
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
        """Plots the fitted 3D Gaussian curve along with the original PSF data, and saves the plot to the specified output path.
        Args:
            outputPath (str): Path to the folder where the plot will be saved
        """
        center = self.getLocalCentroid()
        psf = self._image.astype(np.float64)
        psfs = [
            psf[:, center[1], center[2]],
            psf[center[0], :, center[2]],
            psf[center[0], center[1], :],
        ]
        for i in range(3):
            coords = np.arange(psf.shape[i])
            fine = np.linspace(0, psf.shape[i] - 1, 100)
            fineCoords = np.full((100, 3), center, dtype=np.float64)
            fineCoords[:, i] = fine
            self.plotSingleFit(
                coords,
                psfs[i],
                fine,
                self.gauss(*self.parameters)(fineCoords),
                outputPath,
                i,
            )

    def getCoords(self, psf: np.ndarray) -> np.ndarray:
        """Generates an array of coordinates corresponding to the length of the PSF profile for 3D fitting.
        Args:
            psf (np.ndarray): 3D image of the PSF data
        Returns:
            np.ndarray: Array of coordinates for each axis (Z, Y, X) corresponding to the PSF profile.
        """
        z, y, x = np.indices(psf.shape)
        return np.stack([z.ravel(), y.ravel(), x.ravel()], axis=-1)

    def processSingleFit(self, index: int):
        """Processes a single fit for the given index, performing fitting, plotting, and calculating metrics.
        Args:
            index (int): ID of the PSF and position in lists for which to perform the fit.
        """
        psf = self._image.astype(np.float64)
        coords = self.getCoords(psf)
        activePath = self.getActivePath(index)
        self.compute1DParams()
        bg = self.params1D[1]
        amp = self.params1D[0]
        mu = [self.params1D[2], self.params1D[3], self.params1D[4]]
        sigma = [self.params1D[5], self.params1D[6], self.params1D[7]]

        params, pcov = self.fitCurve(amp, bg, mu, sigma, coords, psf)

        self.parameters = params
        self.fwhms = [
            pxToUm(self.fwhm(params[5]), self._spacing[0]),
            pxToUm(self.fwhm(params[6]), self._spacing[1]),
            pxToUm(self.fwhm(params[7]), self._spacing[2]),
        ]
        self.uncertainties = [self.uncertainty(pcov)] * 3
        self.determinations = [self.determination(params, coords, psf.flatten())] * 3
        self.pcovs = [pcov] * 3

        if self._show:
            self.plotFit3d(activePath)
            self.show2dFit(psf, activePath)
