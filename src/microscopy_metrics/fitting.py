from os import sched_get_priority_max

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import (
    threshold_otsu,
    threshold_isodata,
    threshold_li,
    threshold_minimum,
    threshold_triangle,
)
from skimage.measure import regionprops, label
from skimage.exposure import adjust_sigmoid
from .utils import *
import math
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os
from copy import copy
from sklearn.metrics import r2_score
from abc import abstractmethod

class FittingTool(object):
    _fittingClasses={}
    def __init__(self):
        self._image = None
        self._centroid = []
        self._spacing = [1,1,1]
        self._roi = []
        self._outputDir = ""
        self._results = []

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._fittingClasses:
            raise ValueError("Class was already registered")
        cls._fittingClasses[name] = cls

    @classmethod
    def getInstance(cls, methodName):
        fitClass = cls._fittingClasses[methodName]
        return fitClass()

    def getCovMatrix(self, image, spacing, centroid):
        """Function to get covariance matrix of a 1D,2D or 3D image

        Args:
            image (np.ndarray): Input image
            spacing (List(float)): Size scale of the image
            centroid (List(float)): coordinates of the centroid
        """

        def cov(x, y, i):
            return np.sum(x * y * i) / np.sum(i)

        extends = [np.arange(l) * s for l, s in zip(image.shape, spacing)]
        grids = np.meshgrid(*extends, indexing="ij")

        if image.ndim == 1:
            x = grids[0].ravel() - centroid[0] * spacing[0]
            return cov(x, x, image.ravel())
        elif image.ndim == 2:
            y = grids[0].ravel() - centroid[0] * spacing[0]
            x = grids[1].ravel() - centroid[1] * spacing[1]
            cxx = cov(x, x, image.ravel())
            cyy = cov(y, y, image.ravel())
            cxy = cov(x, y, image.ravel())
            return np.array([[cxx, cxy], [cxy, cyy]])
        elif image.ndim == 3:
            z = grids[0].ravel() - centroid[0] * spacing[0]
            y = grids[1].ravel() - centroid[1] * spacing[1]
            x = grids[2].ravel() - centroid[2] * spacing[2]
            cxx = cov(x, x, image.ravel())
            cyy = cov(y, y, image.ravel())
            czz = cov(z, z, image.ravel())
            cxy = cov(x, y, image.ravel())
            cxz = cov(x, z, image.ravel())
            cyz = cov(y, z, image.ravel())
            return np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]])
        else:
            NotImplementedError()

    def fwhm(self, sigma):
        """
        Args:
            sigma (float): Gaussian width parameter

        Returns:
            float: Full width half maximum
        """
        return 2 * np.sqrt(2 * np.log(2)) * sigma

    @abstractmethod
    def gauss(self, amp, bg, mu, sigma):
        pass

    @abstractmethod
    def evalFun(self, x, amp, bg, mu, sigma):
        pass

    @abstractmethod
    def fitCurve(self, amp, bg, mu, sigma, coords, psf):
        pass

    @abstractmethod
    def processSingleFit(self, index):
        pass

    def setNormalizedImage(self):
        """Method to normalize a 2D or 3D image and erase negative values

        Raises:
            ValueError: This function only operate on 2D or 3D images

        Returns:
            np.ndarray: Image normalized
        """
        if self._image.ndim not in (2, 3):
            raise ValueError("Image have to be in 2D or 3D.")
        imageFloat = self._image.astype(np.float32)
        imageFloat = (imageFloat - np.min(imageFloat)) / (
                np.max(imageFloat) - np.min(imageFloat) + 1e-6
        )
        imageFloat[imageFloat < 0] = 0
        return imageFloat

    def getActivePath(self, index):
        """
        Args:
            index (int): Bead ID corresping to it's position in the list

        Returns:
            Path: Folder's path found (or created) for the selected bead
        """
        activePath = os.path.join(self._outputDir, f"bead_{index}")
        if not os.path.exists(activePath):
            os.makedirs(activePath)
        return activePath

    def uncertainty(self, pcov):
        """Measure uncertainty of parameters returned by Gaussian fit.

        Parameters
        ----------
        pcov : list
            The covariance matrix between parameters
        Returns
        -------
        perr : np.array
            The list of uncertainty factor for each parameter
        """
        perr = np.sqrt(np.diag(pcov))
        return perr

    @abstractmethod
    def determination(self, params, coords, psf):
        pass

class Fitting1D(FittingTool):
    name = "1D"
    def __init__(self):
        super().__init__()


    def gauss(self, amp, bg, mu, sigma):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            mu (float): center of the curve
            sigma (float): standard deviation of the curve

        Returns:
            float: Intensity value at x following the curve
        """
        return lambda x: amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + bg


    def evalFun(self, x, amp, bg, mu, sigma):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            mu (float): center of the curve
            sigma (float): standard deviation of the curve

        Returns:
            float: Intensity value at x following the curve
        """
        return self.gauss(amp=amp, bg=bg, mu=mu, sigma=sigma)(x)


    def fitCurve(self, amp, bg, mu, sigma, coords, psf):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            mu (float): center of the curve
            sigma (float): standard deviation of the curve
            coords (np.array(float)): List of X coordinates
            psf (np.ndarray): 1D image of the psf

        Returns:
            List(float),Matrix(float): List of fitted parameters and covariance matrix
        """
        params = [amp, bg, mu, sigma]
        popt, pcov = curve_fit(
            self.evalFun,
            coords,
            psf,
            p0=params,
            maxfev=5000,
            bounds=([0, 0, 0, 0], [2, 1, max(coords), len(coords)]),
        )
        return popt, pcov


    def plotFit1d(self, psf1d, coords, params, prefix, ylim=None, ax=None):
        """Function to display fitted Gaussian curve with original psf

        Args:
            psf1d (np.ndarray): Original psf
            coords (np.array(float)): List of X,Y coordinates
            params (List(float)): Fitted parameters of the Gaussian function
            prefix (String): First word of each item in the legend
            ylim (float, optional): Maximum y value to display. Defaults to None.
            ax (Axes, optional): Subplot with the curve. Defaults to None.
        """
        if ax is None:
            ax = plt.gca()

        if ylim is None:
            ylim = [0, psf1d.max() * 1.1]

        fineCoords = np.linspace(coords[0], coords[-1], 500)
        ax.plot(coords, psf1d, "-", label="measurement", color="k")
        ax.scatter(coords, psf1d, color="k", alpha=0.5, label="measurement points")
        ax.plot(coords, [params[1]] * len(coords), "--", label=f"{prefix} background")
        ax.plot(
            coords,
            [params[1] + params[0]] * len(coords),
            "--",
            label=f"{prefix} amplitude",
        )
        ax.plot(
            [params[2]] * 2,
            [params[1], params[1] + params[0]],
            "--",
            label=f"{prefix} location",
        )
        ax.plot(
            fineCoords,
            self.gauss(*params)(fineCoords),
            "--",
            label=f"{prefix} Gaussian",
        )
        ax.set_ylim(ylim)
        ax.legend(loc="upper right")


    def processSingleFit(self, index):
        """
        Args:
            index (int): ID of the psf also position in lists

        Returns:
            List(parameters): A list containing metrics, fwhm, parameters and covariance matrix of the fit.
        """
        result = [index]
        for _ in range(5):
            result.append([])
        imageFloat = self.setNormalizedImage()
        activePath = self.getActivePath(index)
        physic = [
            int(self._centroid[0]),
            int(self._centroid[1] - self._roi[0][1]),
            int(self._centroid[2] - self._roi[0][2]),
        ]
        psf = [
            imageFloat[:, physic[1], physic[2]],
            imageFloat[physic[0], :, physic[2]],
            imageFloat[physic[0], physic[1], :],
        ]
        axe = ["Z", "Y", "X"]
        coords = [
            np.arange(len(psf[0])),
            np.arange(len(psf[1])),
            np.arange(len(psf[2])),
        ]
        for u in range(3):
            lim = [0, psf[u].max() * 1.1]
            bg = np.median(psf[u])
            amp = psf[u].max() - bg
            sigma = np.sqrt(
                self.getCovMatrix(
                    np.clip(psf[u] - bg, 0, psf[u].max()),
                    [self._spacing[u]],
                    (self._centroid - self._roi[0]),
                )
            )
            mu = np.argmax(psf[u])
            params, pcov = self.fitCurve(amp, bg, mu, sigma, coords[u], psf[u])
            with plt.ioff():
                fig = plt.figure(figsize=(15, 5))
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.set_title(f"Fitted curve of the PSF along axis {axe[u]}")
                self.plotFit1d(psf[u], coords[u], params, "Fit", lim, ax=ax2)
                ax2.set_xlabel("Position in the PSF (pixels)")
                ax2.set_ylabel("Intensity level")
                outputPath = os.path.join(activePath, f"fit_curve_1D_{axe[u]}.png")
                fig.savefig(outputPath, dpi=300, bbox_inches="tight")
                plt.close(fig)

            result[1].append(pxToUm(self.fwhm(params[3]), self._spacing[u]))
            result[2].append(self.uncertainty(pcov))
            result[3].append(self.determination(params, coords[u], psf[u]))
            result[4].append(params)
            result[5].append(pcov)
        return result

    
    def determination(self, params, coords, psf):
        """Measure determination coefficient of parameters returned by Gaussian fit.

        Parameters
        ----------
        params : list
            The covariance matrix between parameters
        coords : np.array
            The list of coordinates x in the profile
        psf : np.array
            The initial profile
        Returns
        -------
        r_square : float
            The coefficient representing the quality of the fit
        """
        psfFit = self.evalFun(coords, *params)
        rSquared = r2_score(psf, psfFit)
        return rSquared


class Fitting2D(FittingTool):
    name = "2D"
    def __init__(self):
        super().__init__()

    def gauss(self, amp, bg, muX, muY, cxx, cxy, cyy):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY (float): center coordinates of the Gaussian
            cxx,cxy,cyy (float): standard deviation of the Gaussian

        Returns:
            float: Intensity value at (x,y) following the curve
        """

        def fun(coords):
            covInv = np.linalg.inv(np.array([[cxx, cxy], [cxy, cyy]]))
            exponent = -0.5 * (
                    covInv[0, 0] * (coords[:, 0] - muX) ** 2
                    + 2 * covInv[0, 1] * (coords[:, 0] - muX) * (coords[:, 1] - muY)
                    + covInv[1, 1] * (coords[:, 1] - muY) ** 2
            )

            return amp * np.exp(exponent) + bg

        return fun

    def evalFun(self, x, amp, bg, muX, muY, cxx, cxy, cyy):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY (float): center coordinates of the Gaussian
            cxx,cxy,cyy (float): standard deviation of the Gaussian

        Returns:
            float: Intensity value at (x,y) following the curve
        """
        return self.gauss(amp=amp, bg=bg, muX=muX, muY=muY, cxx=cxx, cxy=cxy, cyy=cyy)(x)

    def fitCurve(self, amp, bg, mu, sigma, coords, psf):
        """
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
                [0, 0, 0, 0, 1e-6, 0, 1e-6],
                [2, 1, psf.shape[0], psf.shape[1], psf.shape[0], 0.5, psf.shape[1]],
            ),
        )
        return popt, pcov

    def show2dFit(self, psf, fit, outputPath):
        fig = plt.figure(figsize=(10, 5))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(psf, cmap="viridis")
        ax1.set_title("PSF Data")

        ax2 = fig.add_subplot(1, 2, 2)
        fitReshaped = fit.reshape(
            psf.shape
        )
        ax2.imshow(fitReshaped, cmap="viridis")
        ax2.set_title("Fit")

        plt.tight_layout()
        fig.savefig(outputPath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plotFit2d(self,psf,center,params,popt,outputPath,coords):
        yLim = [0.0, psf.max() * 1.1]
        yy_fine = np.linspace(0, psf.shape[0] - 1, 500)
        xx_fine = np.linspace(0, psf.shape[1] - 1, 500)
        fineCoordsY = np.column_stack((yy_fine, np.full_like(yy_fine, center[1])))
        fineCoordsX = np.column_stack((np.full_like(xx_fine, center[0]), xx_fine))
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(coords[0],psf[:,center[1]],'-',label="PSF",color="k")
        ax1.scatter(coords[0], psf[:, center[1]], color="k", alpha=0.5, label="measurement points")
        ax1.plot(fineCoordsY[:, 0], self.gauss(*params)(fineCoordsY), label="1D Curve Fit")
        ax1.plot(fineCoordsY[:, 0], self.gauss(*popt)(fineCoordsY), label="2D Curve Fit")
        ax1.set_ylim(yLim)
        ax1.legend()
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(coords[1], psf[center[0]], '-', label='PSF', color='k')
        ax2.scatter(coords[1], psf[center[0]], color='k', alpha=0.5, label="measurement points")
        ax2.plot(fineCoordsX[:, 1], self.gauss(*params)(fineCoordsX), label="1D Curve Fit")
        ax2.plot(fineCoordsX[:, 1], self.gauss(*popt)(fineCoordsX), label="2D Curve Fit")
        ax2.set_ylim(yLim)
        ax2.legend()
        fig.savefig(outputPath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def getCoords(self, psf):
        """Function to get a 1D list of 2D coordinates for the Gaussian fitting

        Args:
            psf (np.ndarray): 2D image to process
            axe1 (int): index of the first axe of the image
            axe2 (int): index of the second axe of the image

        Returns:
            List(List(float)): Coordinates for the 2D fit
        """
        yy = np.arange(psf.shape[0])
        xx = np.arange(psf.shape[1])
        y, x = np.meshgrid(yy, xx, indexing="ij")
        return np.stack([y.ravel(), x.ravel()], -1)

    def processSingleFit(self, index):
        """
        Args:
            index (int): ID of the psf also position in lists

        Returns:
            List(parameters): A list containing metrics, fwhm, parameters and covariance matrix of the fit.
        """
        result = [index]
        for _ in range(3):
            result.append([])
        for _ in range(3):
            result[1].append(0.0)
        imageFloat = self.setNormalizedImage()
        physic = [
            int(self._centroid[0]),
            int(self._centroid[1] - self._roi[0][1]),
            int(self._centroid[2] - self._roi[0][2]),
        ]
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
        fitTool1D = Fitting1D()
        fitTool1D._image = self._image
        fitTool1D._roi = self._roi
        fitTool1D._spacing = self._spacing
        fitTool1D._outputDir = self._outputDir
        fitTool1D._centroid = self._centroid
        results1D = fitTool1D.processSingleFit(index)
        params1D = results1D[4]
        pcovs1D = results1D[5]
        for u in range(3):
            lim = [0, psf[u].max() * 1.1]
            bg = params1D[u][1]
            amp = params1D[u][0]
            if u + 1 < 3:
                u2 = u + 1
                sigma = [params1D[u][3], 0, params1D[u2][3]]
                mu = [params1D[u][2], params1D[u2][2]]
            else:
                u2 = 0
                sigma = [params1D[u2][3], 0, params1D[u][3]]
                mu = [params1D[u2][2], params1D[u][2]]
            params, pcov = self.fitCurve(amp, bg, mu, sigma, coords[u], psf[u])
            outputPath = os.path.join(activePath, f"fit_curve_1D_{axe[u][0]}.png")
            psfFit = self.evalFun(coords[u], *params)
            self.show2dFit(psf[u], psfFit, outputPath)
            x, y = np.arange(psf[u].shape[0]), np.arange(psf[u].shape[1])
            if u + 1 < 3:
                coordsTmp = [x, y]
                self.plotFit2d(psf[u], [physic[u],physic[u2]],[amp, bg, *mu, *sigma],params,outputPath,coordsTmp)
            else :
                coordsTmp = [x, y]
                self.plotFit2d(psf[u], [physic[u2],physic[u]],[amp, bg, *mu, *sigma],params,outputPath,coordsTmp)
            pcov[0, 0] += pcovs1D[u][0, 0]
            pcov[1, 1] += pcovs1D[u][1, 1]
            pcov[2, 2] += pcovs1D[u][2, 2]
            pcov[3, 3] += pcovs1D[u2][2, 2]
            if u + 1 < 3:
                pcov[4, 4] += pcovs1D[u][3, 3]
                pcov[6, 6] += pcovs1D[u2][3, 3]
                result[1][u] += pxToUm(self.fwhm(params[4]), self._spacing[u])
                result[1][u2] += pxToUm(self.fwhm(params[6]), self._spacing[u2])
            else:
                pcov[4, 4] += pcovs1D[u2][3, 3]
                pcov[6, 6] += pcovs1D[u][3, 3]
                result[1][u] += pxToUm(self.fwhm(params[6]), self._spacing[u])
                result[1][u2] += pxToUm(self.fwhm(params[4]), self._spacing[u2])
            result[2].append(self.uncertainty(pcov))
            result[3].append(self.determination(params, coords[u], psf[u].flatten()))
        for i in range(len(result[1])):
            result[1][i] /= 2
        return result

    def determination(self, params, coords, psf):
        """Measure determination coefficient of parameters returned by 2D Gaussian fit.

        Parameters
        ----------
        params : list
            The covariance matrix between parameters
        coords : np.array
            The list of coordinates x in the profile
        psf : np.array
            The initial profile
        Returns
        -------
        r_square : float
            The coefficient representing the quality of the fit
        """
        psfFit = self.evalFun(coords, *params)

        meanIntensity = np.mean(psf)
        varData = np.sum((psf - meanIntensity) ** 2)
        varResidual = np.sum((psf - psfFit) ** 2)

        rSquared = 1 - (varResidual / varData)
        return rSquared
    
class Fitting3D(FittingTool):
    name = "3D"
    def __init__(self):
        super().__init__()

    def gauss(self, amp, bg, muX, muY,muZ, cxx, cxy,cxz, cyy,cyz,czz):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY,muZ (float): center coordinates of the Gaussian
            cxx,cxy,cxz,cyy,cyz,czz (float): standard deviation of the Gaussian

        Returns:
            float: Intensity value at (x,y,z) following the curve
        """

        def fun(coords):
            covInv = np.linalg.inv(np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]]))
            exponent = -0.5 * (
                    covInv[0, 0] * (coords[:, 2] - muX) ** 2
                    + 2 * covInv[0, 1] * (coords[:, 2] - muX) * (coords[:, 1] - muY)
                    + 2 * covInv[0, 2] * (coords[:, 2] - muX) * (coords[:, 0] - muZ)
                    + covInv[1, 1] * (coords[:, 1] - muY) ** 2
                    + 2 * covInv[1, 2] * (coords[:, 1] - muY) * (coords[:, 0] - muZ)
                    + covInv[2, 2] * (coords[:, 0] - muZ) ** 2
            )

            return amp * np.exp(exponent) + bg
        return fun

    def evalFun(self, x, amp, bg, muX, muY,muZ, cxx, cxy,cxz, cyy,cyz,czz):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY (float): center coordinates of the Gaussian
            cxx,cxy,cyy (float): standard deviation of the Gaussian

        Returns:
            float: Intensity value at (x,y) following the curve
        """
        return self.gauss(amp=amp, bg=bg, muX=muX, muY=muY,muZ=muZ, cxx=cxx, cxy=cxy,cxz=cxz, cyy=cyy,cyz=cyz,czz=czz)(x)

    def fitCurve(self, amp, bg, mu, sigma, coords, psf):
        """
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
                [0, 0, 0, 0, 0, 1e-6, 0, 0, 1e-6, 0, 1e-6],
                [2, 1, psf.shape[0], psf.shape[1],psf.shape[2], psf.shape[0], 0.5, 0.5, psf.shape[1], 0.5, psf.shape[2]],
            ),
        )
        return popt, pcov

    def show2dFit(self, psf, fit, outputPath):
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

    def plotFit3d(self, psf, center, params, popt, outputPath, coords):
        yLim = [0.0, psf.max() * 1.1]
        zz_fine = np.linspace(0, psf.shape[0] - 1, 500)
        yy_fine = np.linspace(0, psf.shape[1] - 1, 500)
        xx_fine = np.linspace(0, psf.shape[2] - 1, 500)
        fineCoordsZ = np.column_stack((zz_fine, np.full_like(zz_fine, center[1]), np.full_like(zz_fine, center[2])))
        fineCoordsY = np.column_stack((np.full_like(yy_fine, center[0]), yy_fine, np.full_like(yy_fine, center[2])))
        fineCoordsX = np.column_stack((np.full_like(xx_fine, center[0]), np.full_like(xx_fine, center[1]), xx_fine))
        fig = plt.figure(figsize=(20, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(coords[0], psf[:, center[1], center[2]], '-', label="PSF", color="k")
        ax1.scatter(coords[0], psf[:, center[1], center[2]], color="k", alpha=0.5, label="measurement points")
        ax1.plot(fineCoordsZ[:, 0], self.gauss(*params)(fineCoordsZ), label="1D Curve Fit")
        ax1.plot(fineCoordsZ[:, 0], self.gauss(*popt)(fineCoordsZ), label="3D Curve Fit")
        ax1.set_ylim(yLim)
        ax1.set_title('Z Profile')
        ax1.legend()
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(coords[1], psf[center[0], :, center[2]], '-', label='PSF', color='k')
        ax2.scatter(coords[1], psf[center[0], :, center[2]], color='k', alpha=0.5, label="measurement points")
        ax2.plot(fineCoordsY[:, 1], self.gauss(*params)(fineCoordsY), label="1D Curve Fit")
        ax2.plot(fineCoordsY[:, 1], self.gauss(*popt)(fineCoordsY), label="3D Curve Fit")
        ax2.set_ylim(yLim)
        ax2.set_title('Y Profile')
        ax2.legend()
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(coords[2], psf[center[0], center[1], :], '-', label='PSF', color='k')
        ax3.scatter(coords[2], psf[center[0], center[1], :], color='k', alpha=0.5, label="measurement points")
        ax3.plot(fineCoordsX[:, 2], self.gauss(*params)(fineCoordsX), label="1D Curve Fit")
        ax3.plot(fineCoordsX[:, 2], self.gauss(*popt)(fineCoordsX), label="3D Curve Fit")
        ax3.set_ylim(yLim)
        ax3.set_title('X Profile')
        ax3.legend()
        fig.savefig(outputPath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def getCoords(self, psf):
        """Function to get a 1D list of 2D coordinates for the Gaussian fitting

        Args:
            psf (np.ndarray): 2D image to process
            axe1 (int): index of the first axe of the image
            axe2 (int): index of the second axe of the image

        Returns:
            List(List(float)): Coordinates for the 2D fit
        """
        zz = np.arange(psf.shape[0])
        yy = np.arange(psf.shape[1])
        xx = np.arange(psf.shape[2])
        z, y, x = np.meshgrid(zz, yy, xx, indexing="ij")
        return np.stack([z.ravel(), y.ravel(), x.ravel()], -1)

    def processSingleFit(self, index):
        """
        Args:
            index (int): ID of the psf also position in lists

        Returns:
            List(parameters): A list containing metrics, fwhm, parameters and covariance matrix of the fit.
        """
        result = [index]
        for _ in range(3):
            result.append([])
        for _ in range(3):
            result[1].append(0.0)
        physic = [
            int(self._centroid[0]),
            int(self._centroid[1] - self._roi[0][1]),
            int(self._centroid[2] - self._roi[0][2]),
        ]
        psf = self.setNormalizedImage()
        coords = self.getCoords(psf)
        activePath = self.getActivePath(index)
        fitTool1D = Fitting1D()
        fitTool1D._image = self._image
        fitTool1D._roi = self._roi
        fitTool1D._spacing = self._spacing
        fitTool1D._outputDir = self._outputDir
        fitTool1D._centroid = self._centroid
        results1D = fitTool1D.processSingleFit(index)
        params1D = results1D[4]
        pcovs1D = results1D[5]
        bg = (params1D[0][1] + params1D[1][1] + params1D[2][1])/3
        amp = (params1D[0][0] + params1D[1][0] + params1D[2][0])/3
        mu = [params1D[0][2], params1D[1][2], params1D[2][2]]
        sigma = [params1D[0][3], 0, 0, params1D[1][3], 0, params1D[2][3]]
        params, pcov = self.fitCurve(amp, bg, mu, sigma, coords, psf)
        outputPath = os.path.join(activePath, f"fit_curve_1D_Z.png")
        x, y, z = np.arange(psf.shape[0]), np.arange(psf.shape[1]), np.arange(psf.shape[2])
        coordsTmp = [x, y, z]
        self.plotFit3d(psf,physic,[amp,bg,*mu,*sigma],params,outputPath,coordsTmp)
        zz_fine = np.linspace(0, psf.shape[0], 500)
        yy_fine = np.linspace(0, psf.shape[1], 500)
        xx_fine = np.linspace(0, psf.shape[2], 500)
        z_fine, y_fine, x_fine = np.meshgrid(zz_fine, yy_fine, xx_fine, indexing="ij")
        fine_coords_zyx = np.stack([z_fine.ravel(), y_fine.ravel(), x_fine.ravel()], -1)

        cv3D = self.gauss(*params)(fine_coords_zyx)
        cv3D = cv3D.reshape((500,500,500))
        #self.show2dFit(psf[physic[0]],cv3D[physic[0]],outputPath)
        pcov[0, 0] = (pcov[0,0] + pcovs1D[0][0, 0] + pcovs1D[1][0, 0] + pcovs1D[2][0, 0])/4
        pcov[1, 1] = (pcov[1,1] + pcovs1D[0][1, 1] + pcovs1D[1][1, 1] + pcovs1D[2][1, 1])/4
        pcov[2, 2] = (pcov[2,2] + pcovs1D[0][2, 2])/2
        pcov[3, 3] = (pcov[3,3] + pcovs1D[1][2, 2])/2
        pcov[4, 4] = (pcov[4,4] + pcovs1D[2][2, 2])/2
        pcov[5, 5] = (pcov[5,5] + pcovs1D[0][3, 3])/2
        pcov[8, 8] = (pcov[8,8] + pcovs1D[1][3, 3])/2
        pcov[10, 10] = (pcov[10,10] + pcovs1D[2][3, 3])/2
        result[1] = [
            pxToUm(self.fwhm(params[5]), self._spacing[0]),
            pxToUm(self.fwhm(params[8]), self._spacing[1]),
            pxToUm(self.fwhm(params[10]), self._spacing[2]),
        ]
        tmp = self.uncertainty(pcov)
        result[2] = [tmp,tmp,tmp]
        tmp = self.determination(params, coords, psf.flatten())
        result[3] = [tmp,tmp,tmp]
        return result

    def determination(self, params, coords, psf):
        """Measure determination coefficient of parameters returned by 2D Gaussian fit.

        Parameters
        ----------
        params : list
            The covariance matrix between parameters
        coords : np.array
            The list of coordinates x in the profile
        psf : np.array
            The initial profile
        Returns
        -------
        r_square : float
            The coefficient representing the quality of the fit
        """
        psfFit = self.evalFun(coords, *params)

        meanIntensity = np.mean(psf)
        varData = np.sum((psf - meanIntensity) ** 2)
        varResidual = np.sum((psf - psfFit) ** 2)

        rSquared = 1 - (varResidual / varData)
        return rSquared


class Fitting(object):
    def __init__(self):
        self._images = []
        self._centroids = []
        self._spacing = [1, 1, 1]
        self._rois = []
        self._outputDir = ""
        self.results = []
        self.fitType = "1D"

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        if len(images) == 0 or images is None:
            raise ValueError("Please, send at list one image")
        self._images = images

    @property
    def centroids(self):
        return self._centroids

    @centroids.setter
    def centroids(self, centroids):
        if len(centroids) == 0 or centroids is None:
            raise ValueError("Please, send at list one centroid")
        self._centroids = centroids

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, value):
        if value is None or len(value) == 0:
            raise ValueError("Shape format not compatible with current image")
        self._spacing = value

    @property
    def rois(self):
        return self._rois

    @rois.setter
    def rois(self, rois):
        if len(rois) == 0 or rois is None:
            raise ValueError("Please, send at list one ROI")
        self._rois = rois

    @property
    def outputDir(self):
        return self._outputDir

    @outputDir.setter
    def outputDir(self, value):
        if value is None or not os.path.exists(value):
            raise ValueError("The outputDir is wrong")
        self._outputDir = value


    def runFitting(self, index):
        fitTool = FittingTool.getInstance(self.fitType)
        fitTool._image = self._images[index]
        fitTool._centroid = self._centroids[index]
        fitTool._spacing = self.spacing
        fitTool._roi = self._rois[index]
        fitTool._outputDir = self._outputDir
        return fitTool.processSingleFit(index)
        

    def computeFitting(self):
        self.results = []
        workers = int(os.cpu_count() * 0.5)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.runFitting, i): i
                for i, roi in enumerate(self._rois)
            }

            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
