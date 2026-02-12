import numpy as np 
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import threshold_otsu,threshold_isodata,threshold_li,threshold_minimum,threshold_triangle
from skimage.measure import regionprops, label
from skimage.exposure import adjust_sigmoid
from .utils import *
import math
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


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

def plot_fit_1d(psf1d, coords, params, prefix, ylim=[0, 9200]):
    fine_coords = np.linspace(coords[0], coords[-1], 500)
    plt.plot(coords, psf1d, '-', label='measurment', color='k')
    plt.bar(coords, psf1d, width=15, color='k')
    plt.plot(coords, [params[1],] * len(coords), '--', label=f'{prefix} background' )
    plt.plot(coords, [params[1] + params[0],] * len(coords), '--', label=f'{prefix} amplitude')
    plt.plot([params[2],]* 2, [params[1], params[1] + params[0]], '--', label=f'{prefix} location')
    plt.plot(fine_coords, gauss_1d(*params)(fine_coords), '--', label=f'{prefix} Gaussian')
    plt.ylim(ylim)
    plt.legend();

def gauss_1d(amp, bg, mu, sigma):
    return lambda x: amp * np.exp(-(x-mu)**2 / (2*sigma**2)) + bg

def eval_fun(x,amp,bg,mu,sigma):
    return gauss_1d(amp=amp,bg=bg,mu=mu,sigma=sigma)(x)

def fit_curve_1D(amp,bg,mu,sigma,coords_x,psf_x,y_lim):
    params = [amp,bg,mu, sigma]
    popt,pcov = curve_fit(
        eval_fun,
        coords_x,
        psf_x,
        p0=params
    )
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plot_fit_1d(psf_x, coords_x, params, "Est.", y_lim)
    plt.title('Estimated from data');
    plt.subplot(1,2,2)
    plot_fit_1d(psf_x, coords_x, popt, "Curve fit", y_lim)
    plt.title('Curve fit');
    plt.show()
