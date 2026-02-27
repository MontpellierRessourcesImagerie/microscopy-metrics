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

def eval_fun_2D(x, amp, bg, mu_x, mu_y, cxx, cxy, cyy):
    return gauss_2d(amp=amp, bg=bg, mu_x=mu_x, mu_y=mu_y, cxx=cxx, cxy=cxy, cyy=cyy)(x)

def fit_curve_2D(psf_yx,spacing):
    yy = np.arange(psf_yx.shape[0]) * spacing[1]
    xx = np.arange(psf_yx.shape[1]) * spacing[2]
    y, x = np.meshgrid(yy, xx, indexing="ij")
    coords_yx = np.stack([y.ravel(), x.ravel()], -1)

    yy_fine = np.linspace(0, psf_yx.shape[0], 500) * spacing[1]
    xx_fine = np.linspace(0, psf_yx.shape[1], 500) * spacing[2]
    y_fine, x_fine = np.meshgrid(yy_fine, xx_fine, indexing="ij")
    fine_coords_yx = np.stack([y_fine.ravel(), x_fine.ravel()], -1)

    bg = np.median(psf_yx)
    amp = psf_yx.max() - bg
    mu_y, mu_x = np.unravel_index(psf_yx.argmax(), psf_yx.shape)
    sigma_x = 2.0
    sigma_y = 2.0
    cxy = 0.0

    params = [amp, bg, mu_x, mu_y, sigma_x, cxy, sigma_y]

    popt,pcov = curve_fit(
        eval_fun_2D,
        coords_yx,
        psf_yx.ravel(),
        p0=params,
        maxfev=2000000,
        bounds=(
            [0, 0, 0, 0, 0.1, -0.5, 0.1],
            [2, 1, psf_yx.shape[1], psf_yx.shape[0], len(coords_yx), len(coords_yx), len(coords_yx)]
        )

    )


    cv_2d_params = copy(popt)
    cv_2d = gauss_2d(*popt)(fine_coords_yx)
    cv_2d = cv_2d.reshape((500,500))
    plot = show_2D_fit(psf_yx,cv_2d)
    return popt,pcov,plot

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
        for _ in range(3) :
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
                self.plot_fit_1d(psf[u], coords[u], params, "Fit", lim, ax=ax2)
                output_path = os.path.join(active_path, f'fit_curve_1D_{axe[u]}.png')
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

            result[1].append(self.fwhm(params[3]))
            result[2].append(self.uncertainty(pcov))
            result[3].append(self.determination(params,coords[u],psf[u]))
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
                

                