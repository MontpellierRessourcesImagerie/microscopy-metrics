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


class Fitting(object):
    def __init__(self):
        self._images = []
        self._centroids = []
        self._spacing = [1, 1, 1]
        self._rois = []
        self._output_dir = ""
        self.results = []

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
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        if value is None or not os.path.exists(value):
            raise ValueError("The output_dir is wrong")
        self._output_dir = value

    def get_cov_matrix(self, image, spacing, centroid):
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

    def gauss_1d(self, amp, bg, mu, sigma):
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

    def gauss_2d(self, amp, bg, mu_x, mu_y, cxx, cxy, cyy):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            mu_x,mu_y (float): center coordinates of the Gaussian
            cxx,cxy,cyy (float): standard deviation of the Gaussian

        Returns:
            float: Intensity value at (x,y) following the curve
        """

        def fun(coords):
            cov_inv = np.linalg.inv(np.array([[cxx, cxy], [cxy, cyy]]))
            exponent = -0.5 * (
                cov_inv[0, 0] * (coords[:, 0] - mu_x) ** 2
                + 2 * cov_inv[0, 1] * (coords[:, 0] - mu_x) * (coords[:, 1] - mu_y)
                + cov_inv[1, 1] * (coords[:, 1] - mu_y) ** 2
            )

            return amp * np.exp(exponent) + bg

        return fun

    def eval_fun(self, x, amp, bg, mu, sigma):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            mu (float): center of the curve
            sigma (float): standard deviation of the curve

        Returns:
            float: Intensity value at x following the curve
        """
        return self.gauss_1d(amp=amp, bg=bg, mu=mu, sigma=sigma)(x)

    def eval_fun_2D(self, x, amp, bg, mu_x, mu_y, cxx, cxy, cyy):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            mu_x,mu_y (float): center coordinates of the Gaussian
            cxx,cxy,cyy (float): standard deviation of the Gaussian

        Returns:
            float: Intensity value at (x,y) following the curve
        """
        return self.gauss_2d(
            amp=amp, bg=bg, mu_x=mu_x, mu_y=mu_y, cxx=cxx, cxy=cxy, cyy=cyy
        )(x)

    def set_normalized_image(self, image):
        """Method to normalize a 2D or 3D image and erase negative values

        Args:
            image (np.ndarray): Image to be normalized

        Raises:
            ValueError: This function only operate on 2D or 3D images

        Returns:
            np.ndarray: Image normalized
        """
        if image.ndim not in (2, 3):
            raise ValueError("Image have to be in 2D or 3D.")
        image_float = image.astype(np.float32)
        image_float = (image_float - np.min(image_float)) / (
            np.max(image_float) - np.min(image_float) + 1e-6
        )
        image_float[image_float < 0] = 0
        return image_float

    def get_active_path(self, index):
        """
        Args:
            index (int): Bead ID corresping to it's position in the list

        Returns:
            Path: Folder's path found (or created) for the selected bead
        """
        active_path = os.path.join(self._output_dir, f"bead_{index}")
        if not os.path.exists(active_path):
            os.makedirs(active_path)
        return active_path

    def fit_curve_1D(self, amp, bg, mu, sigma, coords_x, psf_x):
        """
        Args:
            amp (float): amplitude of the curve
            bg (float): background intensity
            mu (float): center of the curve
            sigma (float): standard deviation of the curve
            coords_x (np.array(float)): List of X coordinates
            psf_x (np.ndarray): 1D image of the psf

        Returns:
            List(float),Matrix(float): List of fitted parameters and covariance matrix
        """
        params = [amp, bg, mu, sigma]
        popt, pcov = curve_fit(
            self.eval_fun,
            coords_x,
            psf_x,
            p0=params,
            maxfev=5000,
            bounds=([0, 0, 0, 0], [2, 1, max(coords_x), len(coords_x)]),
        )
        return popt, pcov

    def fit_curve_2D(self, amp, bg, mu, sigma, coords, psf):
        """
        Args:
            amp (float): amplitude of the Gaussian
            bg (float): background intensity
            mu (List(float)): center of the Gaussian
            sigma (List(float)): standard deviation of the Gaussian
            coords (np.array(float)): List of X,Y coordinates
            psf_x (np.ndarray): 1D image of the flatten 2D psf

        Returns:
            List(float),Matrix(float): List of fitted parameters and covariance matrix
        """
        params = [amp, bg, *mu, *sigma]
        popt, pcov = curve_fit(
            self.eval_fun_2D,
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

    def plot_fit_1d(self, psf1d, coords, params, prefix, ylim=None, ax=None):
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

        fine_coords = np.linspace(coords[0], coords[-1], 500)
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
            fine_coords,
            self.gauss_1d(*params)(fine_coords),
            "--",
            label=f"{prefix} Gaussian",
        )
        ax.set_ylim(ylim)
        ax.legend(loc="upper right")

    def show_2d_fit(self, psf, fit, output_path):
        fig = plt.figure(figsize=(10, 5))

        # Utilisez fig.add_subplot ou plt.subplot
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(psf, cmap="viridis")
        ax1.set_title("PSF Data")

        ax2 = fig.add_subplot(1, 2, 2)
        fit_reshaped = fit.reshape(
            psf.shape
        )  # Assurez-vous que fit a la même forme que psf
        ax2.imshow(fit_reshaped, cmap="viridis")
        ax2.set_title("Fit")

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Fermez la figure pour libérer la mémoire

    def process_single_fit(self, index):
        """
        Args:
            index (int): ID of the psf also position in lists

        Returns:
            List(parameters): A list containing metrics, fwhm, parameters and covariance matrix of the fit.
        """
        result = [index]
        for _ in range(5):
            result.append([])
        image_float = self.set_normalized_image(self._images[index])
        active_path = self.get_active_path(index)
        physic = [
            int(self._centroids[index][0]),
            int(self._centroids[index][1] - self._rois[index][0][1]),
            int(self._centroids[index][2] - self._rois[index][0][2]),
        ]
        psf = [
            image_float[:, physic[1], physic[2]],
            image_float[physic[0], :, physic[2]],
            image_float[physic[0], physic[1], :],
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
                self.get_cov_matrix(
                    np.clip(psf[u] - bg, 0, psf[u].max()),
                    [self._spacing[u]],
                    (self._centroids[index] - self._rois[index][0]),
                )
            )
            mu = np.argmax(psf[u])
            params, pcov = self.fit_curve_1D(amp, bg, mu, sigma, coords[u], psf[u])
            with plt.ioff():
                fig = plt.figure(figsize=(15, 5))
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.set_title(f"Fitted curve of the PSF along axis {axe[u]}")
                self.plot_fit_1d(psf[u], coords[u], params, "Fit", lim, ax=ax2)
                ax2.set_xlabel("Position in the PSF (pixels)")
                ax2.set_ylabel("Intensity level")
                output_path = os.path.join(active_path, f"fit_curve_1D_{axe[u]}.png")
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

            result[1].append(px_to_um(self.fwhm(params[3]), self._spacing[u]))
            result[2].append(self.uncertainty(pcov))
            result[3].append(self.determination(params, coords[u], psf[u]))
            result[4].append(params)
            result[5].append(pcov)
        return result

    def get_coords(self, psf, axe1, axe2):
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

    def process_single_fit_2D(self, index):
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
            result[1].append(0)
        image_float = self.set_normalized_image(self._images[index])
        physic = [
            int(self._centroids[index][0]),
            int(self._centroids[index][1] - self._rois[index][0][1]),
            int(self._centroids[index][2] - self._rois[index][0][2]),
        ]
        psf = [
            image_float[:, :, physic[2]],
            image_float[physic[0], :, :],
            image_float[:, physic[1], :],
        ]
        axe = ["ZY", "YX", "XZ"]
        coords = [
            self.get_coords(psf[0], 0, 1),
            self.get_coords(psf[1], 1, 2),
            self.get_coords(psf[2], 0, 2),
        ]
        active_path = self.get_active_path(index)
        results_1D = self.process_single_fit(index)
        params_1D = results_1D[4]
        pcovs_1D = results_1D[5]
        for u in range(3):
            lim = [0, psf[u].max() * 1.1]
            bg = params_1D[u][1]
            amp = params_1D[u][0]
            if u + 1 < 3:
                u2 = u + 1
                sigma = [params_1D[u][3], 0, params_1D[u2][3]]
                mu = [params_1D[u][2], params_1D[u2][2]]
            else:
                u2 = 0
                sigma = [params_1D[u2][3], 0, params_1D[u][3]]
                mu = [params_1D[u2][2], params_1D[u][2]]
            params, pcov = self.fit_curve_2D(amp, bg, mu, sigma, coords[u], psf[u])
            output_path = os.path.join(active_path, f"fit_curve_1D_{axe[u][0]}.png")
            psf_fit = self.eval_fun_2D(coords[u], *params)
            self.show_2d_fit(psf[u], psf_fit, output_path)
            pcov[0, 0] += pcovs_1D[u][0, 0]
            pcov[1, 1] += pcovs_1D[u][1, 1]
            pcov[2, 2] += pcovs_1D[u][2, 2]
            pcov[3, 3] += pcovs_1D[u2][2, 2]
            pcov[4, 4] += pcovs_1D[u][3, 3]
            pcov[5, 5] += pcovs_1D[u2][3, 3]
            result[1][u] += px_to_um(self.fwhm(params[4]), self._spacing[u])
            result[1][u2] += px_to_um(self.fwhm(params[5]), self._spacing[u2])
            result[2].append(self.uncertainty(pcov))
            result[3].append(self.determination_2D(params, coords[u], psf[u].flatten()))
        for i in range(len(result[1])):
            result[1][i] /= 2
        return result

    def compute_fitting_1D(self):
        self.results = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.process_single_fit_2D, i): i
                for i, roi in enumerate(self._rois)
            }

            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)

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
        psf_fit = self.eval_fun(coords, *params)
        r_squared = r2_score(psf, psf_fit)
        return r_squared

    def determination_2D(self, params, coords, psf):
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
        psf_fit = self.eval_fun_2D(coords, *params)

        mean_intensity = np.mean(psf)
        Var_data = np.sum((psf - mean_intensity) ** 2)
        Var_residual = np.sum((psf - psf_fit) ** 2)

        r_squared = 1 - (Var_residual / Var_data)
        return r_squared
