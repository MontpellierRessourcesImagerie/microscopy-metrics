import numpy as np
from microscopy_metrics.scripts.PSFGenerator.Point3D import Point3D

class Data3D(object):
    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nxy = nx * ny
        
        # On utilise numpy pour stocker les données selon [Z, Y, X]
        self.data = np.zeros((nz, ny, nx), dtype=np.float64)
        
        self.max = Point3D()
        self.fwhm = Point3D()
        self.energy = 0.0

    def createAsByte(self, z):
        scaled = np.clip(self.data[z] * 255.0, 0, 255)
        return scaled.astype(np.uint8).flatten()

    def createAsShort(self, z):
        scaled = np.clip(self.data[z] * 65535.0, 0, 65535)
        return scaled.astype(np.uint16).flatten()

    def createAsFloat(self, z):
        return self.data[z].astype(np.float32).flatten()

    def createAsDouble(self, z):
        return self.data[z].copy().flatten()

    def getHistogram(self, nbins):
        # Transpose/flatten calculation based on nbins
        scaled_data = (self.data * nbins).astype(np.int32)
        valid_data = scaled_data[scaled_data >= 0]
        histo = np.bincount(valid_data, minlength=nbins)
        if len(histo) > nbins:
            histo = histo[:nbins]
        return histo

    def getPlane(self, z):
        return self.data[z].flatten()

    def setPlane(self, z, plane):
        self.data[z] = np.array(plane).reshape((self.ny, self.nx))

    def putXY(self, z, plane):
        self.setPlane(z, plane)

    def getXY(self, z, plane=None):
        if plane is not None:
            plane[:] = self.data[z].flatten()
        return self.data[z].flatten()

    def determineMaximumAndEnergy(self):
        # Compute squared energy
        self.energy = float(np.sum(self.data ** 2))
        
        # Find global max
        max_idx = np.unravel_index(np.argmax(self.data), self.data.shape)
        self.max.z = max_idx[0]
        self.max.y = max_idx[1]
        self.max.x = max_idx[2]
        self.max.value = float(self.data[max_idx])

    def getMaximum(self, z):
        return float(np.max(self.data[z]))

    def multiply(self, num):
        self.data *= num

    def clip(self, lower, upper):
        np.clip(self.data, lower, upper, out=self.data)

    def rescale(self, scale, max_val):
        """
        Scale the intensity PSF. Scale (scale==0, linear scale, do nothing)
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
        if z is None:
            return float(np.sum(self.data ** 2))
        return float(np.sum(self.data[z] ** 2))

    def getPlaneInformation(self):
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
        
        self.fwhm.value = float(np.sum(
            self.data[
                z1 : min(z2 + 1, self.nz),
                y1 : min(y2 + 1, self.ny),
                x1 : min(x2 + 1, self.nx)
            ]
        ))
