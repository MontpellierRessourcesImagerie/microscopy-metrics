import math
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit

from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.fittingTools.fitting1D import Fitting1D
from microscopy_metrics.utils import pxToUm


class Fitting3DRotation(FittingTool):
    """Class for fitting a 3D Gaussian curve with rotation to the PSF profile of a microscopy image.
    This class inherits from the FittingTool base class and implements methods specific to 3D Gaussian fitting with rotation.
    It includes methods for evaluating the Gaussian function, fitting the curve to the data, and plotting the results.
    """

    name = "3D Rotation"

    def __init__(self):
        super().__init__()
        self.thetas = [0, 0, 0]

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
        thetaX: float,
        thetaY: float,
        thetaZ: float,
    ):
        """Generates a 3D Gaussian function with rotation based on the provided parameters.
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY,muZ (float): center coordinates of the Gaussian
            sigmaX,sigmaY,sigmaZ (float): standard deviation of the Gaussian
            thetaX,thetaY,thetaZ (float): rotation angles around each axis
        Returns:
            float: Intensity value at (x,y,z) following the curve
        """

        def fun(coords: np.ndarray):
            x = coords[:, 0] - muX
            y = coords[:, 1] - muY
            z = coords[:, 2] - muZ
            cx, sx = np.cos(thetaX), np.sin(thetaX)
            cy, sy = np.cos(thetaY), np.sin(thetaY)
            cz, sz = np.cos(thetaZ), np.sin(thetaZ)
            x_rot = (
                (cy * cz) * x
                + (cz * sx * sy - cx * sz) * y
                + (cx * cz * sy + sx * sz) * z
            )
            y_rot = (
                (cy * sz) * x
                + (cx * cz + sx * sy * sz) * y
                + (-cz * sx + cx * sy * sz) * z
            )
            z_rot = (-sy) * x + (cy * sx) * y + (cy * cx) * z
            exponent = (
                -(x_rot**2) / (2 * sigmaX**2)
                - (y_rot**2) / (2 * sigmaY**2)
                - (z_rot**2) / (2 * sigmaZ**2)
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
        thetaX: float,
        thetaY: float,
        thetaZ: float,
    ):
        """Evaluates the 3D Gaussian function with rotation at the given x values.
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY,muZ (float): center coordinates of the Gaussian
            sigmaX,sigmaY,sigmaZ (float): standard deviation of the Gaussian
            thetaX,thetaY,thetaZ (float): rotation angles around each axis
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
            thetaX=thetaX,
            thetaY=thetaY,
            thetaZ=thetaZ,
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
        """Fits a 3D Gaussian curve with rotation to the provided PSF data.
        Args:
            amp (float): amplitude of the Gaussian
            bg (float): background intensity
            mu (list): center of the Gaussian
            sigma (list): standard deviation of the Gaussian
            coords (np.array(float)): List of X,Y,Z coordinates
            psf (np.ndarray): 1D image of the flatten 2D psf
        Returns:
            list,Matrix(float): List of fitted parameters and covariance matrix
        """
        params = [amp, bg, *mu, *sigma, 0, 0, 0]
        bounds = (
            [0, -np.inf, 0, 0, 0, 1e-6, 1e-6, 1e-6, -np.pi, -np.pi, -np.pi],
            [
                np.inf,
                np.inf,
                psf.shape[0],
                psf.shape[1],
                psf.shape[2],
                np.inf,
                np.inf,
                np.inf,
                np.pi,
                np.pi,
                np.pi,
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

    def show2dFit(self, outputPath: str):
        """Plots the fitted 3D Gaussian curve with rotation against the original PSF data for all three axes.
        Args:
            outputPath (str): Directory where the plots will be saved.
        """
        center = self.getLocalCentroid()
        psf = self._image.astype(np.float64)
        fitShapeZ = min(psf.shape[0] * 5, 256)
        fitShapeY = min(psf.shape[1] * 5, 128)
        fitShapeX = min(psf.shape[2] * 5, 128)
        zz_fine = np.linspace(0, psf.shape[0], fitShapeZ)
        yy_fine = np.linspace(0, psf.shape[1], fitShapeY)
        xx_fine = np.linspace(0, psf.shape[2], fitShapeX)
        z_fine, y_fine, x_fine = np.meshgrid(zz_fine, yy_fine, xx_fine, indexing="ij")
        fine_coords_zyx = np.stack([z_fine.ravel(), y_fine.ravel(), x_fine.ravel()], -1)
        z0 = self.parameters[2] * (fitShapeZ / psf.shape[0])
        y0 = self.parameters[3] * (fitShapeY / psf.shape[1])
        x0 = self.parameters[4] * (fitShapeX / psf.shape[2])
        L = min(max(psf.shape) * 5 / 4, 30)
        thetaX = self.thetas[0]
        thetaY = self.thetas[1]
        thetaZ = self.thetas[2]
        cx, sx = np.cos(thetaX), np.sin(thetaX)
        cy, sy = np.cos(thetaY), np.sin(thetaY)
        cz, sz = np.cos(thetaZ), np.sin(thetaZ)
        R = np.array(
            [
                [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
                [cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz],
                [-sy, cy * sx, cy * cx],
            ]
        ).T
        idx_major = np.argmax(
            [self.parameters[5], self.parameters[6], self.parameters[7]]
        )
        v_major = R[:, idx_major]
        vz, vy, vx = v_major
        L = min(max(psf.shape) * 5 / 4, 30)
        params = [*self.parameters, *self.thetas]
        fit = self.gauss(*params)(fine_coords_zyx)
        fit = fit.reshape((fitShapeZ, fitShapeY, fitShapeX))
        angle_yx = np.degrees(np.arctan2(vy, vx))
        dx_yx = L * np.cos(np.arctan2(vy, vx))
        dy_yx = L * np.sin(np.arctan2(vy, vx))
        x1_yx = x0 + dx_yx
        y1_yx = y0 + dy_yx
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(psf[center[0]], cmap="viridis")
        ax1.set_title("PSF Data")
        ax2 = fig.add_subplot(1, 2, 2)
        centerFit = int(fitShapeZ * (center[0] / psf.shape[0]))
        ax2.imshow(fit[centerFit], cmap="viridis")
        ax2.set_title(f"Fit (YX angle: {angle_yx:.1f}°)")
        ax2.plot([x0 - dx_yx, x1_yx], [y0 - dy_yx, y1_yx], color="red", linewidth=2)
        ax2.scatter([x0], [y0], color="red", alpha=0.7)
        ax2.axhline(y=fitShapeY / 2, color="k", alpha=0.5, linestyle="--")
        ax2.axvline(x=fitShapeX / 2, color="k", alpha=0.5, linestyle="--")
        plt.tight_layout()
        fig.savefig(
            os.path.join(outputPath, "2D_Gaussian_Image_YX.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        angle_xz = np.degrees(np.arctan2(vz, vx))
        dx_xz = L * np.cos(np.arctan2(vz, vx))
        dz_xz = L * np.sin(np.arctan2(vz, vx))
        z1_xz = z0 + dz_xz
        x1_xz = x0 + dx_xz
        fig2 = plt.figure(figsize=(10, 5))
        ax1 = fig2.add_subplot(1, 2, 1)
        ax1.imshow(psf[:, center[1], :], cmap="viridis")
        ax1.set_title("PSF Data")
        ax2 = fig2.add_subplot(1, 2, 2)
        centerFit = int(fitShapeY * (center[1] / psf.shape[1]))
        ax2.imshow(fit[:, centerFit, :], cmap="viridis")
        ax2.set_title(f"Fit (XZ angle: {angle_xz:.1f}°)")
        ax2.plot([x0 - dx_xz, x1_xz], [z0 - dz_xz, z1_xz], color="red", linewidth=2)
        ax2.scatter([x0], [z0], color="red", alpha=0.7)
        ax2.axhline(y=fitShapeZ / 2, color="k", alpha=0.5, linestyle="--")
        ax2.axvline(x=fitShapeX / 2, color="k", alpha=0.5, linestyle="--")
        plt.tight_layout()
        fig2.savefig(
            os.path.join(outputPath, "2D_Gaussian_Image_XZ.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig2)
        angle_zy = np.degrees(np.arctan2(vz, vy))
        dy_zy = L * np.cos(np.arctan2(vz, vy))
        dz_zy = L * np.sin(np.arctan2(vz, vy))
        z1_zy = z0 + dz_zy
        y1_zy = y0 + dy_zy
        fig3 = plt.figure(figsize=(10, 5))
        ax1 = fig3.add_subplot(1, 2, 1)
        ax1.imshow(psf[:, :, center[2]], cmap="viridis")
        ax1.set_title("PSF Data")
        ax2 = fig3.add_subplot(1, 2, 2)
        centerFit = int(fitShapeX * (center[2] / psf.shape[2]))
        ax2.imshow(fit[:, :, centerFit], cmap="viridis")
        ax2.set_title(f"Fit (ZY angle: {angle_zy:.1f}°)")
        ax2.plot([y0 - dy_zy, y1_zy], [z0 - dz_zy, z1_zy], color="red", linewidth=2)
        ax2.scatter([y0], [z0], color="red", alpha=0.7)
        ax2.axhline(y=fitShapeZ / 2, color="k", alpha=0.5, linestyle="--")
        ax2.axvline(x=fitShapeY / 2, color="k", alpha=0.5, linestyle="--")
        plt.tight_layout()
        fig3.savefig(
            os.path.join(outputPath, "2D_Gaussian_Image_ZY.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig3)

    def plotSingleFit(
        self,
        coords: np.ndarray,
        psf: np.ndarray,
        fineCoords: np.ndarray,
        fit3D: np.ndarray,
        outputPath: str,
        index: int,
    ):
        """Plots the fitted 3D Gaussian curve with rotation against the original PSF data for a single axis.
        Args:
            coords (np.ndarray): Coordinates of the data points along the axis being plotted.
            psf (np.ndarray): Intensity values at the data points along the axis being plotted.
            fineCoords (np.ndarray): Coordinates for plotting the fitted curve along the axis being plotted.
            fit3D (np.ndarray): Intensity values of the fitted 3D Gaussian curve with rotation along the axis being plotted.
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
            label=f"Mu: {self.parameters[2 + index]:.2f}",
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
        """Plots the fitted 3D Gaussian curve with rotation against the original PSF data for all three axes.
        Args:
            outputPath (str): Directory where the plots will be saved.
        """
        psf = self._image.astype(np.float64)
        center = self.getLocalCentroid()
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

            params = [*self.parameters, *self.thetas]
            self.plotSingleFit(
                coords,
                psfs[i],
                fine,
                self.gauss(*params)(fineCoords),
                outputPath,
                i,
            )

    def getCoords(self, psf: np.ndarray):
        """Generates an array of coordinates corresponding to the length of the PSF profile for 3D fitting.
        Args:
            psf (np.ndarray): 3D image to process
        Returns:
            np.ndarray: Coordinates for the 3D fit
        """
        z, y, x = np.indices(psf.shape)
        return np.stack([z.ravel(), y.ravel(), x.ravel()], axis=-1)

    def processSingleFit(self, index: int):
        """Processes a single fit for the given index, performing fitting, plotting, and calculating metrics.
        Args:
            index (int): ID of the PSF and position in lists for which to perform the fit.
        Returns:
            List(parameters): A list containing metrics, fwhm, parameters and covariance matrix of the fit.
        """
        psf = self._image.astype(np.float64)
        self.coords = self.getCoords(psf)
        activePath = self.getActivePath(index)
        self.compute1DParams()
        bg = self.params1D[1]
        amp = self.params1D[0]
        mu = [self.params1D[2], self.params1D[3], self.params1D[4]]
        sigma = [self.params1D[5], self.params1D[6], self.params1D[7]]
        params, pcov = self.fitCurve(amp, bg, mu, sigma, self.coords, psf)
        self.thetas = params[8:11]
        self.parameters[0] = params[0]
        self.parameters[1] = params[1]
        self.parameters[2:5] = params[2:5]
        self.parameters[5:8] = params[5:8]
        self.pcovs = [pcov] * 3
        if self._show:
            self.plotFit3d(activePath)
            self.show2dFit(activePath)
            thetaX, thetaY, thetaZ = self.thetas
            cx, sx, cy, sy, cz, sz = (
                np.cos(thetaX),
                np.sin(thetaX),
                np.cos(thetaY),
                np.sin(thetaY),
                np.cos(thetaZ),
                np.sin(thetaZ),
            )
            R = np.array(
                [
                    [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
                    [cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz],
                    [-sy, cy * sx, cy * cx],
                ]
            ).T
            idx_major = np.argmax(
                [self.parameters[5], self.parameters[6], self.parameters[7]]
            )
            major_axis = self.axes[idx_major]
            vz, vy, vx = R[:, idx_major]
            print("---")
            print(
                f"Thetas : X={math.degrees(thetaX):.2f}°, Y={math.degrees(thetaY):.2f}°, Z={math.degrees(thetaZ):.2f}°"
            )
            print(
                f"Major axis  : {major_axis} (sigma = {self.parameters[5+idx_major]:.2f} px)"
            )
            print(
                f"Angles of major axis : YX={np.degrees(np.arctan2(vy, vx)):.2f}°, XZ={np.degrees(np.arctan2(vz, vx)):.2f}°, ZY={np.degrees(np.arctan2(vz, vy)):.2f}°"
            )
            print("---")
        self.fwhms = [
            pxToUm(self.fwhm(self.parameters[5]), self._spacing[0]),
            pxToUm(self.fwhm(self.parameters[6]), self._spacing[1]),
            pxToUm(self.fwhm(self.parameters[7]), self._spacing[2]),
        ]
        self.uncertainties = [self.uncertainty(pcov)] * 3
        self.determinations = [
            self.determination(
                [*self.parameters, *self.thetas], self.coords, psf.flatten()
            )
        ] * 3
        self.pcovs = [pcov] * 3
