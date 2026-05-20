import math

import numpy as np
import SimpleITK as sitk
from skimage import measure
from scipy.spatial import ConvexHull
import open3d as o3d
import pyvista as pv
from scipy.ndimage import gaussian_filter


class MeshBuilder(object):
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
        if self._verticesResized is None or self._faces is None:
            raise ValueError(
                "Mesh has not been built. Please build the mesh before saving."
            )
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self._verticesResized)
        mesh.triangles = o3d.utility.Vector3iVector(self._faces)
        o3d.io.write_triangle_mesh(filename, mesh)

    def concavity(self):
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
        self.BuildMesh()
        self.concavity()
        self.curvature()
