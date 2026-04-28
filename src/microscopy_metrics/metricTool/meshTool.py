import numpy as np
import SimpleITK as sitk
from skimage import measure
from scipy.spatial import ConvexHull
import open3d as o3d
import pyvista as pv
from scipy.ndimage import zoom

class MeshBuilder(object):
    def __init__(self, image=None):
        self._image = image
        self._pixelSize = [1.0, 1.0, 1.0]
        self._RMSMaxError = 0.02
        self._numIterations = 500
        self._curvatureScaling = 0.1
        self._vertices = None
        self._faces = None
        self._concavity = None
        self._curvature = None
        self._sphericity = None

#TODO : delete border vertices of the image to avoid artifacts in the mesh
    def BuildMesh(self):
        if self._image is None : 
            raise ValueError("Image is not set. Please set the image before building the mesh.")
        imageVolume = sitk.SmoothingRecursiveGaussian(sitk.GetImageFromArray(self._image.astype(np.float32)), sigma=1.0)

        otsuFilter = sitk.OtsuThresholdImageFilter()
        otsuFilter.SetInsideValue(0)
        otsuFilter.SetOutsideValue(1)
        binaryImage = sitk.Cast(otsuFilter.Execute(imageVolume), sitk.sitkFloat32)
        binaryImage = binaryImage * (-2.0) + 1.0

        chanVese = sitk.ScalarChanAndVeseDenseLevelSetImageFilter()
        chanVese.SetNumberOfIterations(self._numIterations)
        chanVese.SetMaximumRMSError(self._RMSMaxError)
        chanVese.SetCurvatureWeight(self._curvatureScaling)
        segmentation = chanVese.Execute(binaryImage, imageVolume)
        segArray = sitk.GetArrayFromImage(segmentation)

        self._vertices, self._faces, _, _ = measure.marching_cubes(segArray, level=0.5)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self._vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self._faces)
        mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
        mesh.compute_vertex_normals()
        self._vertices = np.asarray(mesh.vertices)
        self._faces = np.asarray(mesh.triangles)
        return self._vertices, self._faces
    
    def concavity(self):
        if self._vertices is None or self._faces is None:
            raise ValueError("Mesh has not been built. Please build the mesh before calculating concavity.")
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self._vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self._faces)
        meshVolume = mesh.get_volume()
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
