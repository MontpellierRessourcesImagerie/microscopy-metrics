import numpy as np 
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import threshold_otsu,threshold_isodata,threshold_li,threshold_minimum,threshold_triangle
from skimage.measure import regionprops, label
from skimage.exposure import adjust_sigmoid
from .utils import *
import math
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from copy import copy
from sklearn.metrics import r2_score

class Fitting(object):
    def __init__(self):
        self.images = []
        self.centroids = []
        self.spacing = [1,1,1]
        self.rois = []
        self.output_dir = ""
        self.results = []

    def get_cov_matrix(self,image,spacing,centroid):
        def cov(x,y,i):
            return np.sum(x*y*i)/np.sum(i)

        extends = [np.arange(l) * s for l,s in zip(image.shape, spacing)]
        grids = np.meshgrid(*extends,indexing="ij")
        
        if image.ndim == 1:
            x = grids[0].ravel() - centroid[0] * spacing[0]
            return cov(x,x,image.ravel())
        elif image.ndim == 2 :
            y = grids[0].ravel() - centroid[0] * spacing[0]
            x = grids[1].ravel() - centroid[1] * spacing[1]
            cxx = cov(x,x,image.ravel())
            cyy = cov(y,y,image.ravel())
            cxy = cov(x,y,image.ravel())
            return np.array([[cxx,cxy],[cxy,cyy]])
        elif image.ndim == 3 :
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
        else :
            NotImplementedError()

    def fwhm(self,sigma):
        return 2 * np.sqrt(2*np.log(2)) * sigma

    def gauss_1d(self,amp, bg, mu, sigma):
        return lambda x: amp * np.exp(-(x-mu)**2 / (2*sigma**2)) + bg

    def gauss_2d(self,amp,bg,mu_x,mu_y,cxx,cxy,cyy):
        def fun(coords):
            cov_inv = np.linalg.inv(np.array([[cxx,cxy],[cxy,cyy]]))
            exponent = -0.5 * (cov_inv[0,0] * (coords[:, 1] - mu_x) ** 2
                                    + 2 * cov_inv[0, 1] * (coords[:, 1] - mu_x) * (coords[:, 0] - mu_y)
                                    + cov_inv[1, 1] * (coords[:, 0] - mu_y) ** 2
                        )
            return amp * np.exp(exponent) + bg
        return fun

    def eval_fun(self,x,amp,bg,mu,sigma):
        return self.gauss_1d(amp=amp,bg=bg,mu=mu,sigma=sigma)(x)

    def eval_fun_2D(self,x, amp, bg, mu_x, mu_y, cxx, cxy, cyy):
        return self.gauss_2d(amp=amp, bg=bg, mu_x=mu_x, mu_y=mu_y, cxx=cxx, cxy=cxy, cyy=cyy)(x)

    def set_normalized_image(self,image):
                if image.ndim not in(2,3):
                    raise ValueError("Image have to be in 2D or 3D.")
                image_float = image.astype(np.float32)
                image_float = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float) + 1e-6)
                image_float[image_float < 0] = 0
                return image_float
                

    def get_active_path(self, index):
        """Utility function to return the current path of a given bead"""
        active_path = os.path.join(self.output_dir,f"bead_{index}")
        if not os.path.exists(active_path):
            os.makedirs(active_path)
        return active_path

    def fit_curve_1D(self,amp,bg,mu,sigma,coords_x,psf_x,y_lim):
        params = [amp,bg,mu, sigma]
        popt,pcov = curve_fit(
            self.eval_fun,
            coords_x,
            psf_x,
            p0=params,
            maxfev=2000,
            bounds=([0, 0, 0, 0], [2, 1, max(coords_x), len(coords_x)])
            )
        return popt,pcov

    def fit_curve_2D(self,amp,bg,mu,sigma,coords,psf,y_lim):
        params = [amp,bg,*mu,*sigma]

        try:
            popt, pcov = curve_fit(
                lambda x, *params: self.eval_fun_2D(x, *params),
                coords,
                psf.flatten(),
                p0=params,
                maxfev=2000000,
                bounds=(
                    [0, 0, 0, 0, 0.1, -0.5, 0.1],
                    [2, 1, psf.shape[1], psf.shape[0], psf.shape[1], psf.shape[1], psf.shape[0]]
                )
            )
        except Exception as e:
            print(f"Erreur lors de l'ajustement de la courbe: {e}")
            return None, None

        return popt, pcov


    def plot_fit_1d(self,psf1d, coords, params, prefix, ylim=None, ax=None):
        if ax is None:
            ax = plt.gca()

        if ylim is None:
            ylim = [0, psf1d.max() * 1.1]

        fine_coords = np.linspace(coords[0], coords[-1], 500)
        ax.plot(coords, psf1d, '-', label='measurement', color='k')
        ax.scatter(coords, psf1d, color='k', alpha=0.5, label='measurement points')
        ax.plot(coords, [params[1]] * len(coords), '--', label=f'{prefix} background')
        ax.plot(coords, [params[1] + params[0]] * len(coords), '--', label=f'{prefix} amplitude')
        ax.plot([params[2]] * 2, [params[1], params[1] + params[0]], '--', label=f'{prefix} location')
        ax.plot(fine_coords, self.gauss_1d(*params)(fine_coords), '--', label=f'{prefix} Gaussian')
        ax.set_ylim(ylim)
        ax.legend(loc='upper right')


    def process_single_fit(self,index):
        result = [index]
        for _ in range(4) :
            result.append([])
        image_float = self.set_normalized_image(self.images[index])
        active_path = self.get_active_path(index)
        physic = [int(self.centroids[index][0]), int(self.centroids[index][1] - self.rois[index][0][1]), int(self.centroids[index][2] - self.rois[index][0][2])]
        psf = [image_float[:,physic[1],physic[2]],image_float[physic[0],:,physic[2]],image_float[physic[0],physic[1],:]]
        axe = ["Z","Y","X"]
        coords = [np.arange(len(psf[0])),np.arange(len(psf[1])),np.arange(len(psf[2]))]
        for u in range(3):
            lim = [0,psf[u].max() * 1.1]
            bg = np.median(psf[u])
            amp = psf[u].max() - bg
            sigma = np.sqrt(self.get_cov_matrix(np.clip(psf[u] - bg, 0, psf[u].max()), [self.spacing[u]], (self.centroids[index] - self.rois[index][0])))
            mu = np.argmax(psf[u])
            params,pcov = self.fit_curve_1D(amp,bg,mu,sigma,coords[u],psf[u],lim)
            with plt.ioff():
                fig = plt.figure(figsize=(15, 5))
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.set_title(f"Fitted curve of the PSF along axis {axe[u]}")
                self.plot_fit_1d(psf[u], coords[u], params, "Fit", lim, ax=ax2)
                ax2.set_xlabel("Position in the PSF (pixels)")
                ax2.set_ylabel("Intensity level")
                output_path = os.path.join(active_path, f'fit_curve_1D_{axe[u]}.png')
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

            result[1].append(px_to_um(self.fwhm(params[3]),self.spacing[u]))
            result[2].append(self.uncertainty(pcov))
            result[3].append(self.determination(params,coords[u],psf[u]))
            result[4].append(params)
        return result

    def get_coords(self,psf,axe1,axe2):
        yy = np.arange(psf.shape[0]) * self.spacing[axe1]
        xx = np.arange(psf.shape[1]) * self.spacing[axe2]
        y, x = np.meshgrid(yy,xx,indexing="ij")
        return np.stack([y.ravel(),x.ravel()],-1)

    def process_single_fit_2D(self,index):
        result = [index]
        for _ in range(3) :
            result.append([])
        for _ in range(3):
            result[1].append(0)
        image_float = self.set_normalized_image(self.images[index])
        active_path = self.get_active_path(index)
        physic = [int(self.centroids[index][0]), int(self.centroids[index][1] - self.rois[index][0][1]), int(self.centroids[index][2] - self.rois[index][0][2])]
        psf = [image_float[:,:,physic[2]],image_float[physic[0],:,:],image_float[:,physic[1],:]]
        axe = ["ZY","YX","XZ"]
        coords = [self.get_coords(psf[0],0,1),self.get_coords(psf[1],1,2),self.get_coords(psf[2],2,0)]
        params_1D = self.process_single_fit(index)[4]
        for u in range(3):
            lim = [0,psf[u].max() * 1.1]
            bg = (params_1D[0][1] + params_1D[1][1] + params_1D[2][1])/3
            amp = (params_1D[0][0] + params_1D[1][0] + params_1D[2][0])/3
            if u+1 < 3 :
                sigma = [params_1D[u][3],0,params_1D[u+1][3]]
                u2 = u+1
            else :
                sigma = [params_1D[u][3],0,params_1D[0][3]]
                u2 = 0
            if u+1 < 3 :
                mu = [params_1D[u][2],params_1D[u+1][2]]
            else :
                mu = [params_1D[u][2],params_1D[0][2]]
            params,pcov = self.fit_curve_2D(amp,bg,mu,sigma,coords[u],psf[u],lim)
            result[1][u] += (px_to_um(self.fwhm(params[4]),self.spacing[u]))
            result[1][u2] += (px_to_um(self.fwhm(params[6]),self.spacing[u2]))
            result[2].append(self.uncertainty(pcov))
            result[3].append(self.determination_2D(params,coords[u],psf[u].flatten()))
        for i in range(len(result[1])):
            result[1][i] /=2
        return result
    

    def compute_fitting_1D(self):
        self.results = []
        with ThreadPoolExecutor() as executor : 
            futures = {executor.submit(self.process_single_fit,i) : i for i, roi in enumerate(self.rois)}

            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)


    def uncertainty(self,pcov):
        """ Measure uncertainty of parameters returned by Gaussian fit.

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


    def determination(self,params, coords, psf):
        """ Measure determination coefficient of parameters returned by Gaussian fit.

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
        psf_fit = self.eval_fun(coords,*params)
        r_squared = r2_score(psf,psf_fit)
        return r_squared

    def determination_2D(self,params, coords, psf):
        """ Measure determination coefficient of parameters returned by Gaussian fit.

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
        psf_fit = self.eval_fun_2D(coords,*params)
        r_squared = r2_score(psf,psf_fit)
        return r_squared
                

                