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


def get_cov_matrix(image,spacing,centroid):
    def cov(x,y,i):
        return np.sum(x*y*i)/np.sum(i)

    extends = [np.arange(l) * s for l,s in zip(get_shape(image), spacing)]
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
        cxx = cov(x, x, img.ravel())
        cyy = cov(y, y, img.ravel())
        czz = cov(z, z, img.ravel())
        cxy = cov(x, y, img.ravel())
        cxz = cov(x, z, img.ravel())
        cyz = cov(y, z, img.ravel())
        return np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]])
    else :
        NotImplementedError()
    
def fwhm(sigma):
    return 2 * np.sqrt(2*np.log(2)) * sigma

def plot_fit_1d(psf1d, coords, params, prefix, ylim=None):
    if ylim is None:
        ylim = [0,psf1d.max()*1.1]
    fine_coords = np.linspace(coords[0], coords[-1], 500)
    plt.plot(coords, psf1d, '-', label='measurement', color='k')
    plt.scatter(coords, psf1d, color='k', alpha=0.5, label='measurement points')
    plt.plot(coords, [params[1],] * len(coords), '--', label=f'{prefix} background' )
    plt.plot(coords, [params[1] + params[0],] * len(coords), '--', label=f'{prefix} amplitude')
    plt.plot([params[2],]* 2, [params[1], params[1] + params[0]], '--', label=f'{prefix} location')
    plt.plot(fine_coords, gauss_1d(*params)(fine_coords), '--', label=f'{prefix} Gaussian')
    plt.ylim(ylim)
    plt.legend(loc='upper right');

def show_2D_fit(psf,fit):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(psf)
    plt.subplot(1,2,2)
    plt.imshow(fit)
    return plt

def gauss_1d(amp, bg, mu, sigma):
    return lambda x: amp * np.exp(-(x-mu)**2 / (2*sigma**2)) + bg

def gauss_2d(amp,bg,mu_x,mu_y,cxx,cxy,cyy):
    def fun(coords):
        cov_inv = np.linalg.inv(np.array([[cxx,cxy],[cxy,cyy]]))
        exponent = -0.5 * (cov_inv[0,0] * (coords[:, 1] - mu_x) ** 2
                                + 2 * cov_inv[0, 1] * (coords[:, 1] - mu_x) * (coords[:, 0] - mu_y)
                                + cov_inv[1, 1] * (coords[:, 0] - mu_y) ** 2
                    )
        return amp * np.exp(exponent) + bg
    return fun

def eval_fun(x,amp,bg,mu,sigma):
    return gauss_1d(amp=amp,bg=bg,mu=mu,sigma=sigma)(x)

def eval_fun_2D(x, amp, bg, mu_x, mu_y, cxx, cxy, cyy):
    return gauss_2d(amp=amp, bg=bg, mu_x=mu_x, mu_y=mu_y, cxx=cxx, cxy=cxy, cyy=cyy)(x)

def fit_curve_1D(amp,bg,mu,sigma,coords_x,psf_x,y_lim):
    params = [amp,bg,mu, sigma]
    popt,pcov = curve_fit(
        eval_fun,
        coords_x,
        psf_x,
        p0=params,
        maxfev=2000,
        bounds=([0, 0, 0, 0], [2, 1, max(coords_x), len(coords_x)])
        )
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plot_fit_1d(psf_x, coords_x, params, "Est.", y_lim)
    plt.title('Estimated from data');
    plt.subplot(1,2,2)
    plot_fit_1d(psf_x, coords_x, popt, "Curve fit", y_lim)
    plt.title('Curve fit');
    return popt,pcov,plt

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