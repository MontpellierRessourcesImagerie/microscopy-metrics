import numpy as np
import SimpleITK as sitk
from skimage import measure
from scipy.spatial import ConvexHull
import open3d as o3d
import pyvista as pv
from skimage.segmentation import clear_border
from scipy.ndimage import gaussian_filter
from microscopy_metrics.thresholdTools.legacy import ThresholdLegacy

class MeshBuilder(object):
    def __init__(self, image=None):
        self._image = image
        self._pixelSize = [1.0, 1.0, 1.0]
        self._RMSMaxError = 0.02
        self._numIterations = 100
        self._curvatureScaling = 5.0
        self._vertices = None
        self._faces = None
        self._concavity = None
        self._curvature = None
        self._sphericity = None
        self._largestRegionMask = None

    def BuildMesh(self):
        if self._image is None : 
            raise ValueError("Image is not set. Please set the image before building the mesh.")
        imgArray = self._image.astype(np.float32)
        imgBlurred = gaussian_filter(imgArray, sigma=0.5)
        maxImg = np.max(imgBlurred)
        NormImg = imgBlurred / maxImg if maxImg > 0 else imgBlurred
        enhancedImg = (NormImg ** 1.25) * maxImg
        imageVolume = sitk.GetImageFromArray(enhancedImg)

        otsuFilter = sitk.TriangleThresholdImageFilter()
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

        segArray = sitk.GetArrayFromImage(segmentation)
        binaryImage = clear_border(segArray.astype(bool))

        LabelledImage = measure.label(binaryImage)
        regions = measure.regionprops(LabelledImage)
        if len(regions) == 0:
            raise ValueError("No regions found in the image. Please check the input image.")
        largestRegion = max(regions, key=lambda r: r.area)
        self._largestRegionMask = (LabelledImage == largestRegion.label).astype(float)
        self._largestRegionMask = gaussian_filter(self._largestRegionMask, sigma=0.5)

        self._vertices, self._faces, _, _ = measure.marching_cubes(self._largestRegionMask, level=0.5)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self._vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self._faces)
        mesh = mesh.filter_smooth_simple(number_of_iterations=5)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.compute_vertex_normals()
        self._vertices = np.asarray(mesh.vertices) * np.array(self._pixelSize)
        self._faces = np.asarray(mesh.triangles)
        return self._vertices, self._faces
    
    def concavity(self):
        if self._vertices is None or self._faces is None:
            raise ValueError("Mesh has not been built. Please build the mesh before calculating concavity.")
        p1 = self._vertices[self._faces[:, 0]]
        p2 = self._vertices[self._faces[:, 1]]
        p3 = self._vertices[self._faces[:, 2]]
        meshVolume = abs(np.sum(p1 * np.cross(p2, p3, axis=1)) / 6.0)
        hull = ConvexHull(self._vertices)
        hullVolume = hull.volume
        print(f"Mesh Volume: {meshVolume}, Hull Volume: {hullVolume}")
        self._concavity = (hullVolume - meshVolume) / hullVolume
        print(f"Concavity: {self._concavity}")
        return self._concavity
    
    def curvature(self):
        if self._vertices is None or self._faces is None:
            raise ValueError("Mesh has not been built. Please build the mesh before calculating curvature.")
        facesPV = np.insert(self._faces, 0, 3, axis=1).flatten()
        meshPV = pv.PolyData(self._vertices, facesPV)
        curvature = meshPV.curvature(curv_type='mean')
        self._curvature = np.nanmean(np.abs(curvature))
        print(f"Curvature: {self._curvature}")
        return self._curvature
    
    def computeMeshMetrics(self):
        self.BuildMesh()
        self.concavity()
        self.curvature()        
