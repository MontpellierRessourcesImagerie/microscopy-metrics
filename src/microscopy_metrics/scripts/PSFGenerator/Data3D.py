# This file is a part of the PSFGenerator toolbox (https://github.com/Biomedical-Imaging-Group/PSFGenerator) by Biomedical-Imaging-Group, which implements 3D point spread function generation for microscopy.
# The code has been adapted to Python and structured as a class for easier integration into the microscopy metrics framework.

import numpy as np

from microscopy_metrics.scripts.PSFGenerator.Point3D import Point3D


class Data3D(object):
    """Class for storing and manipulating 3D data for point spread function (PSF) generation in microscopy.
    This class provides methods for creating data planes, calculating histograms, determining maximum values and energy, and estimating the full width at half maximum (FWHM) of the PSF.
    The data is stored in a 3D numpy array, and the class includes attributes for tracking the maximum point, FWHM, and energy of the PSF.
    """

    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nxy = nx * ny
        self.data = np.zeros((nz, ny, nx), dtype=np.float64)
        self.max = Point3D()
        self.fwhm = Point3D()
        self.energy = 0.0

    def createAsByte(self, z):
        """Creates a byte representation of the data for a specific plane (z) by scaling the values to the range of 0-255 and converting them to unsigned 8-bit integers.
        Args:
            z (int): The index of the plane for which to create the byte representation.
        Returns:
            np.ndarray: A 1D array containing the byte representation of the data for the specified plane.
        """
        scaled = np.clip(self.data[z] * 255.0, 0, 255)
        return scaled.astype(np.uint8).flatten()

    def createAsShort(self, z):
        """Creates a short integer representation of the data for a specific plane (z) by scaling the values to the range of 0-65535 and converting them to unsigned 16-bit integers.
        Args:
            z (int): The index of the plane for which to create the short integer representation.
        Returns:
            np.ndarray: A 1D array containing the short integer representation of the data for the specified plane.
        """
        scaled = np.clip(self.data[z] * 65535.0, 0, 65535)
        return scaled.astype(np.uint16).flatten()

    def createAsFloat(self, z):
        """Creates a float representation of the data for a specific plane (z) by converting the values to 32-bit floating-point numbers.
        Args:
            z (int): The index of the plane for which to create the float representation.
        Returns:
            np.ndarray: A 1D array containing the float representation of the data for the specified plane.
        """
        return self.data[z].astype(np.float32).flatten()

    def createAsDouble(self, z):
        """Creates a double representation of the data for a specific plane (z) by converting the values to 64-bit floating-point numbers.
        Args:
            z (int): The index of the plane for which to create the double representation.
        Returns:
            np.ndarray: A 1D array containing the double representation of the data for the specified plane.
        """
        return self.data[z].copy().flatten()

    def getHistogram(self, nbins):
        """Calculates the histogram of the data for all planes by scaling the values to the specified number of bins and counting the occurrences of each bin value.
        Args:
            nbins (int): The number of bins to use for the histogram calculation.
        Returns:
            np.ndarray: A 1D array containing the histogram counts for each bin value.
        """
        scaled_data = (self.data * nbins).astype(np.int32)
        valid_data = scaled_data[scaled_data >= 0]
        histo = np.bincount(valid_data, minlength=nbins)
        if len(histo) > nbins:
            histo = histo[:nbins]
        return histo

    def getPlane(self, z):
        """Retrieves the data for a specific plane (z) and returns it as a flattened 1D array.
        Args:
            z (int): The index of the plane to retrieve.
        Returns:
            np.ndarray: A 1D array containing the data for the specified plane.
        """
        return self.data[z].flatten()

    def setPlane(self, z, plane):
        """Sets the data for a specific plane (z) by reshaping the input plane to match the dimensions of the data and assigning it to the corresponding index.
        Args:
            z (int): The index of the plane where the data should be set.
            plane (np.ndarray): The 1D array containing the data to be set for the specified plane, which will be reshaped to fit the dimensions of the data array.
        """
        self.data[z] = np.array(plane).reshape((self.ny, self.nx))

    def putXY(self, z, plane):
        """Stores the provided plane data into the specified plane (z) of the data array by reshaping the input plane to match the dimensions of the data and assigning it to the corresponding index.
        Args:
            z (int): The index of the plane where the data should be stored.
            plane (np.ndarray): The 1D array containing the data to be stored in the specified plane, which will be reshaped to fit the dimensions of the data array.
        """
        self.setPlane(z, plane)

    def getXY(self, z, plane=None):
        """Retrieves the data for a specific plane (z) and optionally stores it in the provided plane array.
        If the plane array is not provided, the method returns a flattened version of the data for the specified plane.
        Args:
            z (int): The index of the plane to retrieve.
            plane (np.ndarray, optional): An optional array to store the retrieved plane data. If None, the method returns a flattened version of the data for the specified plane.
        Returns:
            np.ndarray: A 1D array containing the data for the specified plane, either stored in the provided plane array or returned as a flattened version of the data.
        """
        if plane is not None:
            plane[:] = self.data[z].flatten()
        return self.data[z].flatten()

    def determineMaximumAndEnergy(self):
        """Determines the maximum value and energy of the data by iterating through all planes and calculating the sum of squares for the energy and finding the maximum value across all planes.
        The method updates the max attribute with the coordinates and value of the maximum point, and the energy attribute with the calculated energy of the data.
        """
        self.energy = float(np.sum(self.data**2))
        max_idx = np.unravel_index(np.argmax(self.data), self.data.shape)
        self.max.z = max_idx[0]
        self.max.y = max_idx[1]
        self.max.x = max_idx[2]
        self.max.value = float(self.data[max_idx])

    def getMaximum(self, z):
        """Returns the maximum value of the data for a specific plane (z) by finding the maximum value in the specified plane.
        Args:
            z (int): The index of the plane for which to get the maximum value.
        Returns:
            float: The maximum value of the data for the specified plane.
        """
        return float(np.max(self.data[z]))

    def multiply(self, num):
        """Multiplies the data by a specified number by applying element-wise multiplication to the data array.
        Args:
            num (float): The number by which to multiply the data values.
        """
        self.data *= num

    def clip(self, lower, upper):
        """Clips the data values to the specified lower and upper bounds by applying the numpy clip function to the data array.
        Args:
            lower (float): The lower bound for clipping the data values.
            upper (float): The upper bound for clipping the data values.
        """
        np.clip(self.data, lower, upper, out=self.data)

    def rescale(self, scale, max_val):
        """Rescales the data based on the specified scale type and maximum value. The method supports different scaling options, including linear scaling, logarithmic scaling, square root scaling, and decibel scaling. The rescaled data is stored back in the data attribute for further use.
        Args:
            scale (int): The type of scaling to apply (0 for linear, 1 for logarithmic, 2 for square root, 3 for decibel).
            max_val (float): The maximum value used for scaling the data, which is typically the maximum intensity value in the data.
        """
        if scale == 0:
            self.data /= max_val
        elif scale == 1:
            self.data = np.log(np.maximum(self.data / max_val, 1e-6))
        elif scale == 2:
            self.data = np.sqrt(np.maximum(self.data / max_val, 1e-6))
        elif scale == 3:
            self.data = 20 * np.log10(np.maximum(self.data / max_val, 1e-6))

    def getNorm2(self, z=None):
        """Calculates the L2 norm (energy) of the data for a specific plane (z) or for all planes if z is None. The method computes the sum of squares of the data values and returns the calculated energy.
        Args:
            z (int, optional): The index of the plane for which to calculate the energy. If None, the energy is calculated for all planes.
        Returns:
            float: The calculated energy of the data for the specified plane or for all planes.
        """
        if z is None:
            return float(np.sum(self.data**2))
        return float(np.sum(self.data[z] ** 2))

    def getPlaneInformation(self):
        """Calculates and returns information about each plane of the data, including the maximum value, energy, and standard deviation for each plane. The method iterates through all planes, computes the necessary statistics, and returns a 2D array containing the calculated information for each plane.
        Returns:
            np.ndarray: A 2D array where each row corresponds to a plane and contains the plane index, normalized maximum value, normalized energy, and standard deviation for that plane.
        """
        p = np.zeros((self.nz, 4), dtype=np.float64)
        x0 = (self.nx - 1) / 2.0
        y0 = (self.ny - 1) / 2.0
        y_indices, x_indices = np.indices((self.ny, self.nx))
        distances_sq = (x_indices - x0) ** 2 + (y_indices - y0) ** 2
        for z in range(self.nz):
            slice_max_val = self.getMaximum(z)
            slice_energy = self.getNorm2(z)
            p[z, 0] = z
            p[z, 1] = slice_max_val / self.max.value if self.max.value != 0 else 0
            p[z, 2] = slice_energy / self.energy if self.energy != 0 else 0
            sum_val = np.sum(self.data[z])
            sigma2 = np.sum(self.data[z] * distances_sq)
            p[z, 3] = np.sqrt(sigma2 / sum_val) if sum_val != 0 else 0
        return p

    def estimateFWHM(self):
        """Estimates the full width at half maximum (FWHM) of the data by analyzing the intensity profiles along the x, y, and z axes at the location of the maximum point.
        The method calculates the half-maximum value and determines the points where the intensity crosses this value along each axis to estimate the FWHM in each direction.
        The calculated FWHM values are stored in the fwhm attribute for further analysis and evaluation.
        """
        mz, my, mx = self.max.z, self.max.y, self.max.x
        half_max = self.max.value * 0.5
        x_profile = self.data[mz, my, :]
        x2 = mx
        for x in range(mx, self.nx):
            if x_profile[x] < half_max:
                break
            x2 = x
        x1 = mx
        for x in range(mx, -1, -1):
            if x_profile[x] < half_max:
                break
            x1 = x
        y_profile = self.data[mz, :, mx]
        y2 = my
        for y in range(my, self.ny):
            if y_profile[y] < half_max:
                break
            y2 = y
        y1 = my
        for y in range(my, -1, -1):
            if y_profile[y] < half_max:
                break
            y1 = y
        z_profile = self.data[:, my, mx]
        z2 = mz
        for z in range(mz, self.nz):
            if z_profile[z] < half_max:
                break
            z2 = z
        z1 = mz
        for z in range(mz, -1, -1):
            if z_profile[z] < half_max:
                break
            z1 = z
        self.fwhm.x = x2 - x1
        self.fwhm.y = y2 - y1
        self.fwhm.z = z2 - z1
        self.fwhm.value = float(
            np.sum(
                self.data[
                    z1 : min(z2 + 1, self.nz),
                    y1 : min(y2 + 1, self.ny),
                    x1 : min(x2 + 1, self.nx),
                ]
            )
        )
