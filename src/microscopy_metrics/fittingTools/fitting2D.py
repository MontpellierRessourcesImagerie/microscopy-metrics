import os
import numpy as np
from scipy.optimize import curve_fit

from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.fittingTools.fitting1D import Fitting1D
from microscopy_metrics.utils import pxToUm
from sklearn.metrics import r2_score
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class Fitting2D(FittingTool):
    name = "2D"
    def __init__(self):
        super().__init__()

    def gauss(self, amp, bg, muX, muY, sigmaX, sigmaY):
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
            exponent = (-(coords[:,0] - muX) ** 2 / (2 * sigmaX ** 2)-(coords[:,1] - muY) ** 2 / (2 * sigmaY ** 2)) 
            return bg + (amp-bg) * np.exp(exponent)
        return fun

    def evalFun(self, x, amp, bg, muX, muY, sigmaX, sigmaY):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY (float): center coordinates of the Gaussian
            cxx,cxy,cyy (float): standard deviation of the Gaussian

        Returns:
            float: Intensity value at (x,y) following the curve
        """
        return self.gauss(amp=amp, bg=bg, muX=muX, muY=muY, sigmaX=sigmaX, sigmaY=sigmaY)(x)

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
            bounds=([0, -np.inf, 0, 0, 1e-6, 1e-6],[np.inf, np.inf, psf.shape[0], psf.shape[1], psf.shape[0], psf.shape[1]],),
        )
        return popt, pcov

    def show2dFit(self, psf, outputPath,params):
        yy_fine = np.linspace(0,psf.shape[0] - 1, psf.shape[0] * 10)
        xx_fine = np.linspace(0, psf.shape[1] - 1, psf.shape[1] * 10)
        y_fine, x_fine = np.meshgrid(yy_fine, xx_fine, indexing="ij")
        fine_coords_yx = np.stack([y_fine.ravel(), x_fine.ravel()], -1)
        fit = self.gauss(*params)(fine_coords_yx)
        fit = fit.reshape((psf.shape[0] * 10,psf.shape[1] * 10))
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

    def plotSingleFit(self,coords,psf,fineCoords,fit1D,fit2D,axeStr,outputPath,index):
        yLim = [0.0, psf.max() * 1.1]
        fig1,ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(coords, psf, '-', label="PSF", color="k")
        ax1.scatter(coords, psf, color="k", alpha=0.5, label="measurement points")
        ax1.plot(fineCoords, fit1D, label="1D Curve Fit")
        halfMax = (fit1D.max() + fit1D.min()) / 2.0
        ax1.axhline(y=halfMax, color='g', linestyle='--', alpha=0.7, label='FWHM 1D fit')
        ax1.plot(fineCoords, fit2D, label="2D Curve Fit")
        halfMax = (fit2D.max() + fit2D.min()) / 2.0
        ax1.axhline(y=halfMax, color='r', linestyle='--', alpha=0.7, label='FWHM 2D fit')
        ax1.axhline(y=self.parameters[0], color='orange', linestyle='dotted', alpha=0.5, label=f"Amplitude: {self.parameters[0]:.2f}")
        ax1.axvline(x=self.parameters[2+index], color='purple', linestyle='dotted',alpha=0.5, label=f"Mu: {self.parameters[2+index]:.2f}")
        ax1.plot(self.parameters[2+index],self.parameters[0],'go', color='black')
        ax1.set_ylim(yLim)
        ax1.set_title(f'{axeStr} Profile')
        ax1.legend()
        fig1.savefig(os.path.join(outputPath,f"fit_curve_1D_{axeStr}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig1)

    def plotFit3d(self, center, params1D, outputPath, coords):
        psf = self.setNormalizedImage()
        axes = ["Z","Y","X"]
        params2D = [
            [self.parameters[0],self.parameters[1],self.parameters[2],self.parameters[3],self.parameters[5],self.parameters[6]],
            [self.parameters[0],self.parameters[1],self.parameters[3],self.parameters[4],self.parameters[6],self.parameters[7]],
            [self.parameters[0],self.parameters[1],self.parameters[2],self.parameters[4],self.parameters[5],self.parameters[7]]
        ]
        psfs = [
            psf[:, center[1], center[2]],
            psf[center[0], :, center[2]],
            psf[center[0], center[1], :]
        ]
        for i in range (3):
            params = [params1D[0], params1D[1], params1D[2+i], params1D[5+i]]
            fine = np.linspace(0, psf.shape[i]-1, 500)
            if i < 2 :
                index = i+1
                fineCoords = np.column_stack((fine, np.full_like(fine, center[index])))
            else :
                index = 0
                fineCoords = np.column_stack((np.full_like(fine, center[index]),fine))
            self.plotSingleFit(coords[i],psfs[i],fine,Fitting1D().gauss(*params)(fine),self.gauss(*params2D[i])(fineCoords),axes[i],outputPath,i)

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
        imageFloat = self.setNormalizedImage()
        physic = self.getLocalCentroid()
        """ MIP
        psf = [
            self.mip3d(imageFloat,2),
            self.mip3d(imageFloat,0),
            self.mip3d(imageFloat,1)
        ]
        """
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
        fitTool1D = FittingTool.getInstance("1D")
        fitTool1D._show = self._show
        fitTool1D._image = self._image
        fitTool1D._roi = self._roi
        fitTool1D._spacing = self._spacing
        fitTool1D._outputDir = self._outputDir
        fitTool1D._centroid = self._centroid
        fitTool1D.processSingleFit(index)
        params1D = fitTool1D.parameters
        
        amp = params1D[0]
        bg = params1D[1]
        for u in range(3):
            if u + 1 < 3:
                u2 = u + 1
                sigma = [params1D[5+u], params1D[5+u2]]
                mu = [params1D[2+u], params1D[2+u2]]
            else:
                u2 = 0
                sigma = [params1D[5+u2], params1D[5+u]]
                mu = [params1D[2+u2], params1D[2+u]]
            params, pcov = self.fitCurve(amp, bg, mu, sigma, coords[u], psf[u])
            self.parameters[0] += params[0] / 3.0
            self.parameters[1] += params[1] / 3.0
            if u+1 <3 : 
                self.parameters[2+u] += params[2] / 2.0
                self.parameters[2+u2] += params[3] / 2.0
                self.parameters[5+u] += params[4] / 2.0
                self.parameters[5+u2] += params[5] / 2.0
                self.fwhms[u] += pxToUm(self.fwhm(params[4]), self._spacing[u]) / 2.0
                self.fwhms[u2] += pxToUm(self.fwhm(params[5]), self._spacing[u2]) / 2.0
            else :
                self.parameters[2+u] += params[3] / 2.0
                self.parameters[2+u2] += params[2] / 2.0
                self.parameters[5+u] += params[5] / 2.0
                self.parameters[5+u2] += params[4] / 2.0
                self.fwhms[u] += pxToUm(self.fwhm(params[5]), self._spacing[u]) / 2.0
                self.fwhms[u2] += pxToUm(self.fwhm(params[4]), self._spacing[u2]) / 2.0
            self.uncertainties[u] = self.uncertainty(pcov)
            self.determinations[u] = self.determination(params, coords[u], psf[u].flatten())
            self.pcovs[u] = pcov
            outputPath = os.path.join(activePath, f"2D_Gaussian_Image_{axe[u]}.png")
            if self._show : self.show2dFit(psf[u], outputPath,params)
        x, y, z = np.arange(imageFloat.shape[0]), np.arange(imageFloat.shape[1]), np.arange(imageFloat.shape[2])
        coordsTmp = [x, y, z]
        if self._show : self.plotFit3d(physic,params1D,activePath,coordsTmp)

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