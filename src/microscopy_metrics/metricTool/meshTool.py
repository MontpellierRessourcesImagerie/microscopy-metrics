import math

import numpy as np
import SimpleITK as sitk
from skimage import measure
from scipy.spatial import ConvexHull
import open3d as o3d
import pyvista as pv
from scipy.ndimage import gaussian_filter



class MeshBuilder(object):
    """Class for building a 3D mesh from a microscopy image using segmentation and surface extraction techniques.
    This class provides methods for constructing a mesh representation of the largest region in a 3D microscopy image, calculating concavity and curvature metrics, and saving the mesh to a file.
    Attributes:
        _image (np.ndarray): The input 3D microscopy image for mesh construction.
        _pixelSize (list): The pixel size in micrometers for each dimension of the image.
        _RMSMaxError (float): The maximum root mean square error for the Chan-Vese segmentation algorithm.
        _numIterations (int): The number of iterations for the Chan-Vese segmentation algorithm.
        _curvatureScaling (float): The scaling factor for curvature in the Chan-Vese segmentation algorithm.
        _vertices (np.ndarray): The vertices of the constructed mesh.
        _verticesResized (np.ndarray): The resized vertices of the constructed mesh, scaled by the pixel size.
        _faces (np.ndarray): The faces of the constructed mesh.
        _concavity (float): The concavity metric of the constructed mesh.
        _curvature (list): The curvature values of the constructed mesh.
        _sphericity (float): The sphericity metric of the constructed mesh.
        _largestRegionMask (np.ndarray): The binary mask of the largest region in the image.
        _segArray (np.ndarray): The segmented array of the image after applying the Chan-Vese algorithm.  
        
    """
    def __init__(self, image=None):
        self._image = image
        self._pixelSize = [1.0, 1.0, 1.0]
        self._RMSMaxError = 0.02
        self._numIterations = 30
        self._curvatureScaling = 5.0
        self._vertices = None
        self._verticesResized = None
        self._faces = None
        self._concavity = None
        self._curvature = []
        self._sphericity = None
        self._largestRegionMask = None
        self._segArray = None


    def BuildMesh(self):
        """Builds a 3D mesh from the input microscopy image using segmentation and surface extraction techniques.
        The method normalizes the input image, applies Gaussian filtering, and performs Otsu thresholding to create a binary mask. It then applies the Chan-Vese segmentation algorithm to extract the largest region in the image. 
        The method uses the marching cubes algorithm to generate a mesh representation of the largest region, and computes the vertices and faces of the mesh. 
        The vertices are resized based on the pixel size, and the mesh is smoothed and cleaned to remove duplicates. 
        The resulting mesh can be used for further analysis and visualization.
        Raises:
            ValueError: If the input image is not set or no regions are found in the image.
            ValueError: If the largest region cannot be extracted.
        Returns:
            o3d.geometry.TriangleMesh: The constructed 3D mesh.
        """
        if self._image is None:
            raise ValueError(
                "Image is not set. Please set the image before building the mesh."
            )
        imgArray = self._image.astype(np.float32)
        imgBlurred = gaussian_filter(imgArray, sigma=0.5)
        maxImg = np.max(imgBlurred)
        NormImg = imgBlurred / maxImg if maxImg > 0 else imgBlurred
        enhancedImg = (NormImg**1.25) * maxImg
        imageVolume = sitk.GetImageFromArray(enhancedImg)

        otsuFilter = sitk.OtsuThresholdImageFilter()
        otsuFilter.SetInsideValue(0)
        otsuFilter.SetOutsideValue(1)
        thesholdImage = otsuFilter.Execute(imageVolume)
        binaryImage = sitk.Cast(thesholdImage, sitk.sitkFloat32)
        binaryImage = binaryImage * (-2.0) + 1.0

        chanVese = sitk.ScalarChanAndVeseDenseLevelSetImageFilter()
        chanVese.SetNumberOfIterations(self._numIterations)
        chanVese.SetMaximumRMSError(self._RMSMaxError)
        chanVese.SetCurvatureWeight(self._curvatureScaling)
        segmentation = chanVese.Execute(binaryImage, imageVolume)

        self._segArray = sitk.GetArrayFromImage(segmentation)
        binaryImage = self._segArray.astype(bool)

        LabelledImage = measure.label(binaryImage)
        regions = measure.regionprops(LabelledImage)
        if len(regions) == 0:
            raise ValueError(
                "No regions found in the image. Please check the input image."
            )
        largestRegion = max(regions, key=lambda r: r.area)
        self._largestRegionMask = (LabelledImage == largestRegion.label).astype(float)
        self._largestRegionMask = gaussian_filter(self._largestRegionMask, sigma=0.5)

        self._vertices, self._faces, _, _ = measure.marching_cubes(
            self._largestRegionMask, level=0.5
        )

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self._vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self._faces)
        mesh = mesh.filter_smooth_simple(number_of_iterations=5)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.compute_vertex_normals()
        self._vertices = np.asarray(mesh.vertices)
        self._verticesResized = np.asarray(mesh.vertices) * np.array(self._pixelSize)
        self._faces = np.asarray(mesh.triangles)
        return self._verticesResized, self._faces

    def saveMesh(self, filename):
        """Saves the constructed 3D mesh to a file in the specified format.
        Arguments:
            filename (str): The path to the file where the mesh will be saved. The file format is determined by the file extension (e.g., .ply, .obj, .stl).
        Raises:
            ValueError: If the mesh has not been built before saving.
        """
        if self._verticesResized is None or self._faces is None:
            raise ValueError(
                "Mesh has not been built. Please build the mesh before saving."
            )
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self._verticesResized)
        mesh.triangles = o3d.utility.Vector3iVector(self._faces)
        o3d.io.write_triangle_mesh(filename, mesh)

    def concavity(self):
        """Calculates the concavity metric of the constructed 3D mesh.
        The concavity is defined as the ratio of the difference between the volume of the convex hull and the volume of the mesh to the volume of the convex hull.
        A higher concavity value indicates a more concave shape, while a lower value indicates a more convex shape.
        Raises:
            ValueError: If the mesh has not been built before calculating concavity.

        Returns:
            float: The calculated concavity metric of the mesh.
        """
        if self._verticesResized is None or self._faces is None:
            raise ValueError(
                "Mesh has not been built. Please build the mesh before calculating concavity."
            )
        p1 = self._verticesResized[self._faces[:, 0]]
        p2 = self._verticesResized[self._faces[:, 1]]
        p3 = self._verticesResized[self._faces[:, 2]]
        meshVolume = abs(np.sum(p1 * np.cross(p2, p3, axis=1)) / 6.0)
        hull = ConvexHull(self._verticesResized)
        hullVolume = hull.volume
        self._concavity = (hullVolume - meshVolume) / hullVolume
        return self._concavity

    def curvature(self):
        """Calculates the curvature metric of the constructed 3D mesh.
        Raises:
            ValueError: If the mesh has not been built before calculating curvature.
        Returns:
            np.ndarray: The calculated curvature metric of the mesh.
        """
        if self._verticesResized is None or self._faces is None:
            raise ValueError(
                "Mesh has not been built. Please build the mesh before calculating curvature."
            )
        facesPV = np.insert(self._faces, 0, 3, axis=1).flatten()
        meshPV = pv.PolyData(self._verticesResized, facesPV)
        meshPV.flip_faces()
        self._curvature = meshPV.curvature()
        return self._curvature

    def computeMeshMetrics(self):
        """Computes the mesh metrics, including construction, concavity and curvature."""
        self.BuildMesh()
        self.concavity()
        self.curvature()
