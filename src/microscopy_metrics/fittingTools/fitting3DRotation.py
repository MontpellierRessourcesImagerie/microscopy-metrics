import math
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


class Fitting3DRotation(FittingTool):
    name = "3D Rotation"
    def __init__(self):
        super().__init__()
        self.thetas = [0,0,0]

    def gauss(self, amp, bg, muX, muY, muZ, sigmaX, sigmaY, sigmaZ, thetaX, thetaY, thetaZ):
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
            x = coords[:, 0] - muX
            y = coords[:, 1] - muY
            z = coords[:, 2] - muZ
            cx, sx = np.cos(thetaX), np.sin(thetaX)
            cy, sy = np.cos(thetaY), np.sin(thetaY)
            cz, sz = np.cos(thetaZ), np.sin(thetaZ)

            x_rot = (cy*cz) * x + (cz*sx*sy - cx*sz) * y + (cx*cz*sy + sx*sz) * z
            y_rot = (cy*sz) * x + (cx*cz + sx*sy*sz) * y + (-cz*sx + cx*sy*sz) * z
            z_rot = (-sy) * x   + (cy*sx) * y     
            
            exponent = (-(x_rot ** 2) / (2 * sigmaX ** 2) -(y_rot ** 2) / (2 * sigmaY ** 2) -(z_rot ** 2) / (2* sigmaZ **2)) 
            return bg+(amp-bg) * np.exp(exponent)
        return fun

    def evalFun(self, x, amp, bg, muX, muY, muZ, sigmaX, sigmaY, sigmaZ, thetaX, thetaY, thetaZ):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            muX,muY,muZ (float): center coordinates of the Gaussian
            sigmaX,sigmaY,sigmaZ (float): standard deviation of the Gaussian
            thetaX,thetaY,thetaZ (float): rotation angles around each axis

        Returns:
            float: Intensity value at (x,y,z) following the curve
        """
        return self.gauss(amp=amp, bg=bg, muX=muX, muY=muY, muZ=muZ, sigmaX=sigmaX, sigmaY=sigmaY, sigmaZ=sigmaZ, thetaX=thetaX, thetaY=thetaY, thetaZ=thetaZ)(x)

    def fitCurve(self, amp, bg, mu, sigma, coords, psf):
        """
        Args:
            amp (float): amplitude of the Gaussian
            bg (float): background intensity
            mu (List(float)): center of the Gaussian
            sigma (List(float)): standard deviation of the Gaussian
            coords (np.array(float)): List of X,Y,Z coordinates
            psf (np.ndarray): 1D image of the flatten 2D psf

        Returns:
            List(float),Matrix(float): List of fitted parameters and covariance matrix
        """
        params = [amp, bg, *mu, *sigma, 0, 0, 0]
        bounds = ([0,-np.inf,0,0,0,1e-6,1e-6,1e-6,-np.pi,-np.pi,-np.pi],[np.inf, np.inf, psf.shape[0], psf.shape[1],psf.shape[2], np.inf, np.inf,np.inf,np.pi,np.pi,np.pi])
        popt, pcov = curve_fit(
            self.evalFun,
            coords,
            psf.ravel(),
            p0=params,
            maxfev=5000,
            bounds=bounds,
            method="dogbox",
            #loss = 'soft_l1'
        )
        return popt, pcov

    def show2dFit(self, psf, center, outputPath,params,coords):
        fitShapeZ = max(psf.shape[0]*5,512)
        fitShapeY = max(psf.shape[1]*5,256)
        fitShapeX = max(psf.shape[2]*5,256)
        zz_fine = np.linspace(0, psf.shape[0], fitShapeZ)
        yy_fine = np.linspace(0, psf.shape[1], fitShapeY)
        xx_fine = np.linspace(0, psf.shape[2], fitShapeX)
        z_fine, y_fine, x_fine = np.meshgrid(zz_fine, yy_fine, xx_fine, indexing="ij")
        fine_coords_zyx = np.stack([z_fine.ravel(), y_fine.ravel(), x_fine.ravel()], -1)
        z0 = params[2] * 5
        y0 = params[3] * 5
        x0 = params[4] * 5
        L = min(max(psf.shape) * 5 / 4, 30)
        thetaX = params[8]
        thetaY = params[9]
        thetaZ = params[10]
        fit = self.gauss(*params)(fine_coords_zyx)
        fit = fit.reshape((fitShapeZ,fitShapeY,fitShapeX))
        if params[6] > params[7]:
            dy = L * np.cos(thetaY)
            dx = -L * np.sin(thetaY)
        else:
            dy = L * np.sin(thetaX)
            dx = L * np.cos(thetaX)  
        x1 = x0 + dx
        y1 = y0 + dy
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(psf[center[0]], cmap="viridis")
        ax1.set_title("PSF Data")
        ax2 = fig.add_subplot(1, 2, 2)
        centerFit = int(fitShapeZ * (center[0]/psf.shape[0]))
        ax2.imshow(fit[centerFit], cmap="viridis")
        ax2.set_title("Fit")
        ax2.plot([x0,x1],[y0,y1], color="red", linewidth=2)
        ax2.scatter([x0],[y0],color="red", alpha=0.7)
        ax2.axhline(y=fitShapeY / 2, color='k', alpha=0.5, linestyle='--')
        ax2.axvline(x=fitShapeX / 2, color='k', alpha=0.5, linestyle='--')
        plt.tight_layout()
        fig.savefig(os.path.join(outputPath,"2D_Gaussian_Image_YX.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        if params[5] > params[6]:
            dy = L * np.cos(thetaY)
            dx = -L * np.sin(thetaY)
        else:
            dy = L * np.sin(thetaX)
            dx = L * np.cos(thetaX)
        x1 = x0 + dx
        y1 = y0 + dy
        fig2 = plt.figure(figsize=(10, 5))
        ax1 = fig2.add_subplot(1, 2, 1)
        ax1.imshow(psf[:,center[1],:], cmap="viridis")
        ax1.set_title("PSF Data")
        ax2 = fig2.add_subplot(1, 2, 2)
        centerFit = int(fitShapeY * (center[1]/psf.shape[1]))
        ax2.imshow(fit[:,centerFit,:], cmap="viridis")
        ax2.set_title("Fit")
        ax2.plot([x0,x1],[y0,y1], color="red", linewidth=2)
        ax2.scatter([x0],[y0],color="red", alpha=0.7)
        ax2.axhline(y=fitShapeZ / 2, color='k', alpha=0.5, linestyle='--')
        ax2.axvline(x=fitShapeX / 2, color='k', alpha=0.5, linestyle='--')
        plt.tight_layout()
        fig2.savefig(os.path.join(outputPath,"2D_Gaussian_Image_XZ.png"), dpi=300, bbox_inches="tight")
        plt.close(fig2)
        if params[5] > params[7]:
            dy = L * np.cos(thetaZ)
            dx = -L * np.sin(thetaZ)
        else:
            dy = L * np.sin(thetaX)
            dx = L * np.cos(thetaX)
        x1 = x0 + dx
        y1 = y0 + dy
        fig3 = plt.figure(figsize=(10, 5))
        ax1 = fig3.add_subplot(1, 2, 1)
        ax1.imshow(psf[:,:,center[2]], cmap="viridis")
        ax1.set_title("PSF Data")
        ax2 = fig3.add_subplot(1, 2, 2)
        centerFit = int(fitShapeX * (center[2]/psf.shape[2]))
        ax2.imshow(fit[:,:,centerFit], cmap="viridis")
        ax2.set_title("Fit")
        ax2.plot([x0,x1],[y0,y1], color="red", linewidth=2)
        ax2.scatter([x0],[y0],color="red", alpha=0.7)
        ax2.axhline(y=fitShapeY / 2, color='k', alpha=0.5, linestyle='--')
        ax2.axvline(x=fitShapeX / 2, color='k', alpha=0.5, linestyle='--')
        plt.tight_layout()
        fig3.savefig(os.path.join(outputPath,"2D_Gaussian_Image_ZY.png"), dpi=300, bbox_inches="tight")
        plt.close(fig3)

    def plotSingleFit(self,coords,psf,fineCoords,fit1D,fit3D,axeStr,outputPath,params):
        yLim = [0.0, psf.max() * 1.1]
        fig1,ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(coords, psf, '-', label="PSF", color="k")
        ax1.scatter(coords, psf, color="k", alpha=0.5, label="measurement points")
        ax1.plot(fineCoords, fit1D, label="1D Curve Fit")
        halfMax = (fit1D.max() + fit1D.min()) / 2.0
        ax1.axhline(y=halfMax, color='g', linestyle='--', alpha=0.7, label='FWHM 1D fit')
        ax1.plot(fineCoords, fit3D, label="3D Curve Fit")
        halfMax = (fit3D.max() + fit3D.min()) / 2.0
        ax1.axhline(y=halfMax, color='r', linestyle='--', alpha=0.7, label='FWHM 3D fit')
        ax1.axhline(y=params[0], color='orange', linestyle='dotted', alpha=0.5, label=f"Amplitude: {params[0]:.2f}")
        ax1.axvline(x=params[2], color='purple', linestyle='dotted',alpha=0.5, label=f"Mu: {params[2]:.2f}")
        ax1.plot(params[2],params[0],'go', color='black')
        ax1.set_ylim(yLim)
        ax1.set_title(f'{axeStr} Profile')
        ax1.legend()
        fig1.savefig(os.path.join(outputPath,f"fit_curve_1D_{axeStr}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig1)
    
    def plotFit3d(self, psf, center, params, popt, outputPath, coords):
        yLim = [0.0, psf.max() * 1.1]
        psfs = [psf[:, center[1], center[2]],psf[center[0], :, center[2]],psf[center[0], center[1], :]]
        fine = [
            np.linspace(0, psf.shape[0] - 1, 500),
            np.linspace(0, psf.shape[1] - 1, 500),
            np.linspace(0, psf.shape[2] - 1, 500)
        ]
        fineCoords = [
            np.column_stack((fine[0], np.full_like(fine[0], center[1]), np.full_like(fine[0], center[2]))),
            np.column_stack((np.full_like(fine[1], center[0]), fine[1], np.full_like(fine[1], center[2]))),
            np.column_stack((np.full_like(fine[2], center[0]), np.full_like(fine[2], center[1]), fine[2]))
        ]
        Axes = ["Z","Y","X"]
        for i in range(3):
        
            self.plotSingleFit(coords[i],psfs[i],fine[i],Fitting1D().gauss(*params[i])(fine[i]),self.gauss(*popt)(fineCoords[i]),Axes[i],outputPath,[popt[0],popt[1],popt[2+i]])

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
        for _ in range(4):
            result.append([])
        for _ in range(3):
            result[1].append(0.0)
        physic = self.getLocalCentroid()
        psf = self.setNormalizedImage()
        coords = self.getCoords(psf)
        activePath = self.getActivePath(index)
        fitTool1D = FittingTool.getInstance("1D")
        fitTool1D._image = self._image
        fitTool1D._show = self._show
        fitTool1D._roi = self._roi
        fitTool1D._spacing = self._spacing
        fitTool1D._outputDir = self._outputDir
        fitTool1D._centroid = self._centroid
        results1D = fitTool1D.processSingleFit(index)
        params1D = results1D[4]
        pcovs1D = results1D[5]
        bg = (params1D[0][1])
        amp = (params1D[0][0])
        mu = [params1D[0][2], params1D[1][2], params1D[2][2]]
        sigma = [params1D[0][3], params1D[1][3], params1D[2][3]]
        params, pcov = self.fitCurve(amp, bg, mu, sigma, coords, psf)
        self.thetas = params[8:11]
        x, y, z = np.arange(psf.shape[0]), np.arange(psf.shape[1]), np.arange(psf.shape[2])
        coordsTmp = [x, y, z]
        if self._show : self.plotFit3d(psf,physic,params1D,params,activePath,coordsTmp)
        
        if self._show : 
            self.show2dFit(psf,physic,activePath,params,coords)
            for i in range(3):
                print(f"Angle {i}: {math.degrees(self.thetas[i])} ")
        result[1] = [
            pxToUm(self.fwhm(params[5]), self._spacing[0]),
            pxToUm(self.fwhm(params[6]), self._spacing[1]),
            pxToUm(self.fwhm(params[7]), self._spacing[2]),
        ]
        tmp = self.uncertainty(pcov)
        result[2] = [tmp,tmp,tmp]
        tmp = self.determination(params, coords, psf.flatten())
        result[3] = [tmp,tmp,tmp]
        result[4] = params
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