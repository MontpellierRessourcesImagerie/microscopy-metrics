import os
import numpy as np
from scipy.optimize import curve_fit

from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.utils import pxToUm
from sklearn.metrics import r2_score
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

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
        return lambda x: bg + (amp-bg) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


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

    def plotSingleFit(self,coords,psf,fineCoords,fit,axeStr,outputPath,params):
        yLim = [0.0,psf.max() * 1.1]
        fig1,ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(coords, psf, '-', label="PSF", color="k")
        ax1.scatter(coords, psf, color="k", alpha=0.5, label="measurement points")
        ax1.plot(fineCoords, fit, label="1D Curve Fit")
        halfMax = (fit.max() + fit.min()) / 2.0
        ax1.axhline(y=halfMax, color='r', linestyle='--', alpha=0.7, label='FWHM')
        ax1.axhline(y=params[0], color='orange', linestyle='dotted', alpha=0.5, label=f"Amplitude: {params[0]:.2f}")
        ax1.axvline(x=params[2], color='purple', linestyle='dotted',alpha=0.5, label=f"Mu: {params[2]:.2f}")
        ax1.axhline(y=params[1], color='black', linestyle='--',alpha=0.5, label=f"BG: {params[1]:.2f}")
        ax1.plot(params[2],params[0],'go', color='black')
        ax1.set_ylim(yLim)
        ax1.set_title(f'{axeStr} Profile')
        ax1.legend()
        fig1.savefig(os.path.join(outputPath,f"fit_curve_1D_{axeStr}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig1)

    def plotFit1d(self, center, outputPath, coords):
        psf = self.setNormalizedImage()
        axes = ["Z","Y","X"]
        psfs = [psf[:, center[1], center[2]], psf[center[0], :, center[2]], psf[center[0], center[1], :]]
        for i in range(3):
            params = [self.parameters[0], self.parameters[1], self.parameters[2+i], self.parameters[5+i]]
            fine = np.linspace(0, psf.shape[i] - 1, 500)
            self.plotSingleFit(coords[i],psfs[i],fine,self.gauss(*params)(fine),axes[i],outputPath,params)

    def getCoords(self,psf):
        return np.arange(psf.shape[0])


    def processSingleFit(self, index):
        """
        Args:
            index (int): ID of the psf also position in lists

        Returns:
            List(parameters): A list containing metrics, fwhm, parameters and covariance matrix of the fit.
        """
        imageFloat = self.setNormalizedImage()
        activePath = self.getActivePath(index)
        physic = self.getLocalCentroid()
        psf = [
            imageFloat[:, physic[1], physic[2]],
            imageFloat[physic[0], :, physic[2]],
            imageFloat[physic[0], physic[1], :],
        ]
        coords = [
            self.getCoords(psf[0]),
            self.getCoords(psf[1]),
            self.getCoords(psf[2]),
        ]
        self.parameters[0] = 0.0
        self.parameters[1] = 0.0
        for u in range(3):
            bg = np.median(psf[u])
            amp = psf[u].max() - bg
            sigma = self.getCovMatrix(psf[u],physic)
            mu = np.argmax(psf[u])
            params, pcov = self.fitCurve(amp, bg, mu, sigma, coords[u], psf[u])
            self.fwhms[u] = pxToUm(self.fwhm(params[3]), self._spacing[u])
            self.uncertainties[u] = self.uncertainty(pcov)
            self.determinations[u] = self.determination(params, coords[u], psf[u])
            self.parameters[0] += params[0] / 3.0
            self.parameters[1] += params[1] / 3.0
            self.parameters[2+u] = params[2]
            self.parameters[5+u] = params[3]
            self.pcovs[u] = pcov
        if self._show == True: 
            self.plotFit1d(physic,activePath,coords)

    
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