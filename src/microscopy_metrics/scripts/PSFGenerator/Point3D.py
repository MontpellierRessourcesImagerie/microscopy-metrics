# This file is a part of the PSFGenerator toolbox (https://github.com/Biomedical-Imaging-Group/PSFGenerator) by Biomedical-Imaging-Group, which implements the Born & Wolf 3D optical model for generating point spread functions (PSFs) in microscopy.
# The code has been adapted to Python and structured as a class for easier integration into the microscopy metrics framework.


class Point3D :
    """Class representing a point in 3D space with associated intensity value.
    This class provides a simple structure to store the coordinates (x, y, z) and intensity value of a point in 3D space, which can be used for various applications such as representing the position and intensity of a point spread function (PSF) in microscopy.
    """
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.value = 0.0

    def toString(self):
        return f"{self.x} {self.y} {self.z} {self.value}"
