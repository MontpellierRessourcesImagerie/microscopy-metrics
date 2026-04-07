import math
import numpy as np
from microscopy_metrics.scripts.PSFGenerator.KirchhoffDiffractionSimpson import KirchhoffDiffractionSimpson
from microscopy_metrics.scripts.PSFGenerator.Data3D import Data3D

class BornWolfPSF(object):
    def __init__(self):
        self.fullname = "Born & Wolf 3D Optical Model"
        self.shortname = "BW"
        self.niDefault = 1.5
        self.ni = self.niDefault
        self.accuracy = 0

        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.resLateral = 0.0
        self.resAxial = 0.0
        self.NA = 0.0
        self.lmbda = 0.0
        self.data = Data3D(0, 0, 0)
        self.live = True

    def setParameters(self, ni, accuracy):
        self.ni = ni
        self.accuracy = accuracy

    def generate(self):
        for z in range(self.nz):
            if not self.live:
                break
            defocus = self.resAxial * 1e-9 * (z - (self.nz - 1.0) / 2.0)
            self._process_plane(z, defocus)

    def _process_plane(self, z, defocus):
        OVERSAMPLING = 1
        x0 = (self.nx - 1) / 2.0
        y0 = (self.ny - 1) / 2.0
        maxRadius = int(round(math.sqrt((self.nx - x0) ** 2 + (self.ny - y0) ** 2))) + 1
        
        r_length = maxRadius * OVERSAMPLING + 1
        r = np.zeros(r_length)
        h = np.zeros(r_length)

        I_simpson = KirchhoffDiffractionSimpson(defocus, self.ni, self.accuracy, self.NA, self.lmbda)

        for n in range(maxRadius * OVERSAMPLING):
            r[n] = float(n) / float(OVERSAMPLING)
            h[n] = I_simpson.calculate(r[n] * self.resLateral * 1e-9)
            if not self.live:
                return
                
        r[-1] = r[-2] + (r[-2] - r[-3])
        h[-1] = h[-2]

        y_indices, x_indices = np.indices((self.ny, self.nx))
        rPixel = np.sqrt((x_indices - x0) ** 2 + (y_indices - y0) ** 2)
        
        index = np.floor(rPixel * OVERSAMPLING).astype(int)
        
        index = np.clip(index, 0, r_length - 2)
        
        slice_data = h[index] + (h[index + 1] - h[index]) * (rPixel - r[index]) * OVERSAMPLING
        
        if not self.live:
            return 
            
        if self.data is not None:
            self.data.setPlane(z, slice_data)
            