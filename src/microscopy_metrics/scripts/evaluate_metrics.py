from microscopy_metrics.scripts.PSFGenerator.PSF import PSFRandomParameter
from microscopy_metrics.metricTool.metricTool import MetricTool
from microscopy_metrics.fittingTools.fitting1D import Fitting1D
import time
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np

PSF_SIZE = 100
NOISE = True
PSFSHOW = False

def showPSF(psf, title="PSF"):
    """Displays the provided PSF image using matplotlib.
    Args:
        psf (np.ndarray): The PSF image to be displayed.
        title (str, optional): The title for the displayed image. Defaults to "PSF".
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('QtAgg') 
    if not PSFSHOW:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = np.arange(psf.shape[0])
        y = np.arange(psf.shape[1])
        X, Y = np.meshgrid(x, y)
        ax.imshow(psf[:, :, psf.shape[2] // 2], cmap='viridis')
        ax.set_title(title)
        plt.show()

def showPSF2(psf, title="PSF", ax=None):
    """Displays the provided PSF image using matplotlib on the specified axis.
    Args:
        psf (np.ndarray): The PSF image to be displayed.
        title (str, optional): The title for the displayed image. Defaults to "PSF".
        ax (matplotlib.axes.Axes, optional): The axis on which to display the image. If None, a new figure and axis are created. Defaults to None.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('QtAgg')

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    x = np.arange(psf.shape[0])
    y = np.arange(psf.shape[1])
    X, Y = np.meshgrid(x, y)
    ax.imshow(psf[:, psf.shape[2] // 2, :], cmap='viridis')
    ax.set_title(title)

def showPSFvsNoisy():
    """Generates a PSF with comatic aberration, adds noise to it, and displays both the original and noisy PSF images side by side using matplotlib.""" 
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('QtAgg')

    psfGen = PSFRandomParameter(size=PSF_SIZE, aberrationType="spherical")
    psf = psfGen.psf
    noisy_psf = addMicroscopyNoise(psf)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    showPSF2(psf, title="Original PSF", ax=axes[0])    
    showPSF2(noisy_psf, title="Noisy PSF", ax=axes[1])
    plt.tight_layout()
    plt.show()



def addMicroscopyNoise(image, maxPhotons=500, readoutNoiseStd=5.0):
    """Adds Poisson and Gaussian noise to the provided image to simulate microscopy noise.
    Args:
        image (np.ndarray): The input image to which noise will be added.
        maxPhotons (int, optional): The maximum number of photons for scaling the image. Defaults to 500.
        readoutNoiseStd (float, optional): The standard deviation of the Gaussian readout noise. Defaults to 5.0.
    Returns:
        np.ndarray: The noisy image with added Poisson and Gaussian noise, clipped to the range [0, maxPhotons].
    """
    maxPhotons = np.max(image)
    scaledImage = (image / np.max(image)) * maxPhotons
    noisyPoisson = np.random.poisson(scaledImage).astype(np.float32)
    gaussianNoise = np.random.normal(0, readoutNoiseStd, size=image.shape)
    finalNoisyImage = noisyPoisson + gaussianNoise
    finalNoisyImage = np.minimum(np.maximum(finalNoisyImage, 0), maxPhotons)
    return finalNoisyImage

def test_comatic_aberration_with_aberration_single():
    """Tests the detection of comatic aberration in a PSF generated with comatic aberration.
    This function generates a PSF with comatic aberration, applies various methods, and evaluates the results using the defined metrics.
    Returns:
        tuple: A tuple containing the detection results for comaticity, mesh, skeleton, and curvature, along with the durations for each metric calculation.
    """
    psfGen = PSFRandomParameter(size=PSF_SIZE, aberrationType="comatic")
    psf = psfGen.psf
    metricTool = MetricTool()
    metricTool._pixelSize = (0.05, 0.05, 0.05)
    if NOISE:
        psf = addMicroscopyNoise(psf)
    metricTool._image = psf

    Mean_Duration_Mesh = 0.0
    mesh_detected = 0
    Mean_Duration_Skeleton = 0.0
    skeleton_detected = 0
    Mean_Duration_Concavity = 0.0
    curvature_detected = 0

    start_time = time.time()
    metricTool.comaticity()
    duration_comaticity = time.time() - start_time
    comaticity_detected = metricTool._comaticity > 0.0

    start_time = time.time()
    metricTool.meshMetrics()
    Mean_Duration_Mesh += time.time() - start_time
    mesh_detected = metricTool.meshBuilder._concavity > 0.15

    start_time = time.time()
    metricTool.skeletonizePath()
    Mean_Duration_Skeleton += time.time() - start_time
    skeleton_detected = metricTool._skeleton2Extremities > 1.1
    
    start_time = time.time()
    metricTool.curvaturePath()
    Mean_Duration_Concavity += time.time() - start_time
    curvature_detected = metricTool._RMin > 0.25
    
    return comaticity_detected, mesh_detected, skeleton_detected, curvature_detected, duration_comaticity, Mean_Duration_Mesh, Mean_Duration_Skeleton, Mean_Duration_Concavity


def test_comatic_aberration_without_aberration_single():
    """Tests the detection of comatic aberration in a PSF generated without comatic aberration.
    This function generates a PSF without comatic aberration, applies various methods, and evaluates the results using the defined metrics.
    Returns:
        tuple: A tuple containing the detection results for comaticity, mesh, skeleton, and curvature, along with the durations for each metric calculation.
    """
    psfGen = PSFRandomParameter(size=PSF_SIZE)
    psf = psfGen.psf
    metricTool = MetricTool()
    metricTool._pixelSize = (0.05, 0.05, 0.05)
    if NOISE:
        psf = addMicroscopyNoise(psf)
    metricTool._image = psf

    Mean_Duration_Mesh = 0.0
    mesh_detected = 0
    Mean_Duration_Skeleton = 0.0
    skeleton_detected = 0
    Mean_Duration_Concavity = 0.0
    curvature_detected = 0

    start_time = time.time()
    metricTool.comaticity() 
    duration_comaticity = time.time() - start_time
    comaticity_detected = np.isclose(metricTool._comaticity, 0.0, atol=0.05)

    start_time = time.time()
    metricTool.meshMetrics()
    Mean_Duration_Mesh += time.time() - start_time
    mesh_detected = np.isclose(metricTool.meshBuilder._concavity, 0.0, atol=0.15)

    start_time = time.time()
    metricTool.skeletonizePath()
    Mean_Duration_Skeleton += time.time() - start_time
    skeleton_detected = np.isclose(metricTool._skeleton2Extremities, 1.0, atol=0.1)

    start_time = time.time()
    metricTool.curvaturePath()
    Mean_Duration_Concavity += time.time() - start_time
    curvature_detected = metricTool._RMin < 0.25
    
    return comaticity_detected, mesh_detected, skeleton_detected, curvature_detected, duration_comaticity, Mean_Duration_Mesh, Mean_Duration_Skeleton, Mean_Duration_Concavity


def test_comatic_aberration():
    """Test function for evaluating metrics of comatic aberration detection.
    This function generates a PSF with comatic aberration, applies various methods, and evaluates the results using the defined metrics.
    The test assesses the accuracy of the methods in estimating the PSF parameters in the presence of comatic aberration.
    """
    tdqm = tqdm(total=200, desc="Evaluating Comatic Aberration Detection", unit="test")
    
    Mean_Duration_Comaticity = 0.0
    Mean_Duration_Skeleton = 0.0
    Mean_Duration_Curvature = 0.0
    Mean_Duration_Mesh = 0.0
    tp_comatic = 0
    tp_mesh = 0
    tp_skeleton = 0
    tp_curvature = 0
    
    with ProcessPoolExecutor(max_workers=int(os.cpu_count() * 0.75), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(test_comatic_aberration_with_aberration_single) for _ in range(100)]
        for future in as_completed(futures):
            c_det, m_det, s_det, curv_det, duration_comaticity, md_mesh, md_skel, md_conc = future.result()
            Mean_Duration_Comaticity += duration_comaticity
            Mean_Duration_Mesh += md_mesh
            Mean_Duration_Skeleton += md_skel
            Mean_Duration_Curvature += md_conc
            if c_det:
                tp_comatic += 1
            if m_det:
                tp_mesh += 1
            if s_det:
                tp_skeleton += 1
            if curv_det:
                tp_curvature += 1
            tdqm.update(1)
    
    Rappel_comaticity = tp_comatic / 100
    Rappel_skeleton = tp_skeleton / 100
    Rappel_curvature = tp_curvature / 100
    Rappel_mesh = tp_mesh / 100
    
    tn_comatic = 0
    tn_mesh = 0
    tn_skeleton = 0
    tn_curvature = 0
    
    with ProcessPoolExecutor(max_workers=int(os.cpu_count() * 0.75), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(test_comatic_aberration_without_aberration_single) for _ in range(100)]
        for future in as_completed(futures):
            c_det, m_det, s_det, curv_det, duration_comaticity, md_mesh, md_skel, md_conc = future.result()
            Mean_Duration_Comaticity += duration_comaticity
            Mean_Duration_Mesh += md_mesh
            Mean_Duration_Skeleton += md_skel
            Mean_Duration_Curvature += md_conc
            if c_det:
                tn_comatic += 1
            if m_det:
                tn_mesh += 1
            if s_det:
                tn_skeleton += 1
            if curv_det:
                tn_curvature += 1
            tdqm.update(1)
    
    Precision_comaticity = tp_comatic / (tp_comatic + (100 - tn_comatic)) if (tp_comatic + (100-tn_comatic)) > 0 else 0.0
    Precision_skeleton = tp_skeleton / (tp_skeleton + (100 - tn_skeleton)) if (tp_skeleton + (100-tn_skeleton)) > 0 else 0.0
    Precision_curvature = tp_curvature / (tp_curvature + (100 - tn_curvature)) if (tp_curvature + (100-tn_curvature)) > 0 else 0.0
    Precision_mesh = tp_mesh / (tp_mesh + (100 - tn_mesh)) if (tp_mesh + (100-tn_mesh)) > 0 else 0.0
    
    Accuracy_comaticity = (tp_comatic + tn_comatic) / 200
    Accuracy_skeleton = (tp_skeleton + tn_skeleton) / 200
    Accuracy_curvature = (tp_curvature + tn_curvature) / 200
    Accuracy_mesh = (tp_mesh + tn_mesh) / 200
    
    F1_comaticity = 2 * (Precision_comaticity * Rappel_comaticity) / (Precision_comaticity + Rappel_comaticity) if (Precision_comaticity + Rappel_comaticity) > 0 else 0.0
    F1_skeleton = 2 * (Precision_skeleton * Rappel_skeleton) / (Precision_skeleton + Rappel_skeleton) if (Precision_skeleton + Rappel_skeleton) > 0 else 0.0
    F1_curvature = 2 * (Precision_curvature * Rappel_curvature) / (Precision_curvature + Rappel_curvature) if (Precision_curvature + Rappel_curvature) > 0 else 0.0
    F1_mesh = 2 * (Precision_mesh * Rappel_mesh) / (Precision_mesh + Rappel_mesh) if (Precision_mesh + Rappel_mesh) > 0 else 0.0
    
    print(f"Comaticity - Precision: {Precision_comaticity:.2f}, Recall: {Rappel_comaticity:.2f}, Accuracy: {Accuracy_comaticity:.2f}, F1 Score: {F1_comaticity:.2f}, Mean Duration: {Mean_Duration_Comaticity/200:.2f} seconds")
    print(f"Skeleton - Precision: {Precision_skeleton:.2f}, Recall: {Rappel_skeleton:.2f}, Accuracy: {Accuracy_skeleton:.2f}, F1 Score: {F1_skeleton:.2f}, Mean Duration: {Mean_Duration_Skeleton/200:.2f} seconds")
    print(f"Curvature - Precision: {Precision_curvature:.2f}, Recall: {Rappel_curvature:.2f}, Accuracy: {Accuracy_curvature:.2f}, F1 Score: {F1_curvature:.2f}, Mean Duration: {Mean_Duration_Curvature/200:.2f} seconds")
    print(f"Mesh - Precision: {Precision_mesh:.2f}, Recall: {Rappel_mesh:.2f}, Accuracy: {Accuracy_mesh:.2f}, F1 Score: {F1_mesh:.2f}, Mean Duration: {Mean_Duration_Mesh/200:.2f} seconds")


def test_spherical_aberration_with_aberration_single():
    """Tests the detection of spherical aberration in a PSF generated with spherical aberration.    
    This function generates a PSF with spherical aberration, applies various fitting methods, and evaluates the results using the defined metrics.
    Returns:
        tuple: A tuple containing the detection result for sphericality and the duration for the metric calculation.
    """
    psfGen = PSFRandomParameter(size=PSF_SIZE, aberrationType="spherical")
    psf = psfGen.psf
    metricTool = MetricTool()
    metricTool._pixelSize = (0.05, 0.05, 0.05)
    if NOISE:
        psf = addMicroscopyNoise(psf)
    metricTool._image = psf
    
    start_time = time.time()
    metricTool.sphericalAberration()
    duration_sphericality = time.time() - start_time
    sphericality_detected = metricTool._sphericalAberration > 0.05
    
    return sphericality_detected, duration_sphericality


def test_spherical_aberration_without_aberration_single():
    """Tests the detection of spherical aberration in a PSF generated without spherical aberration.
    This function generates a PSF without spherical aberration, applies various fitting methods, and evaluates the results using the defined metrics.
    Returns:
        tuple: A tuple containing the detection result for sphericality and the duration for the metric calculation.
    """
    psfGen = PSFRandomParameter(size=PSF_SIZE)
    psf = psfGen.psf
    metricTool = MetricTool()
    metricTool._pixelSize = (0.05, 0.05, 0.05)
    if NOISE:
        psf = addMicroscopyNoise(psf)
    metricTool._image = psf
    
    start_time = time.time()
    metricTool.sphericalAberration() 
    duration_sphericality = time.time() - start_time
    sphericality_detected = metricTool._sphericalAberration <= 0.05
    
    return sphericality_detected, duration_sphericality


def test_spherical_aberration():
    """Test function for evaluation metrics of spherical aberration.
    This function generates a PSF with spherical aberration, applies various methods, and evaluates the results using the defined metrics.
    The test assesses the accuracy of the methods in estimating the PSF parameters in the presence of spherical aberration.
    """
    Mean_Duration_Sphericality = 0.0
    tp_spherical = 0
    
    tdqm = tqdm(total=200, desc="Evaluating Spherical Aberration Detection", unit="test")
   
    with ProcessPoolExecutor(max_workers=int(os.cpu_count() * 0.75), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(test_spherical_aberration_with_aberration_single) for _ in range(100)]
        for future in as_completed(futures):
            detected, duration_sphericality = future.result()
            Mean_Duration_Sphericality += duration_sphericality
            if detected:
                tp_spherical += 1
            tdqm.update(1)
   
    Rappel_sphericality = tp_spherical / 100

    tn_spherical = 0
    
    with ProcessPoolExecutor(max_workers=int(os.cpu_count() * 0.75), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(test_spherical_aberration_without_aberration_single) for _ in range(100)]
        for future in as_completed(futures):
            detected, duration_sphericality = future.result()
            Mean_Duration_Sphericality += duration_sphericality
            if detected:
                tn_spherical += 1
            tdqm.update(1)
    
    Precision_sphericality = tp_spherical / (tp_spherical + (100 - tn_spherical)) if (tp_spherical + (100-tn_spherical)) > 0 else 0.0
    
    Accuracy_sphericality = (tp_spherical + tn_spherical) / 200
    
    F1_sphericality = 2 * (Precision_sphericality * Rappel_sphericality) / (Precision_sphericality + Rappel_sphericality) if (Precision_sphericality + Rappel_sphericality) > 0 else 0.0
    
    print(f"Spherical Aberration - Precision: {Precision_sphericality:.2f}, Recall: {Rappel_sphericality:.2f}, Accuracy: {Accuracy_sphericality:.2f}, F1 Score: {F1_sphericality:.2f}, Mean Duration: {Mean_Duration_Sphericality/200:.2f} seconds")


def test_astigmatism_aberration_with_aberration_single():
    """Tests the detection of astigmatism aberration in a PSF generated with astigmatism aberration.
    This function generates a PSF with astigmatism aberration, applies various methods, and evaluates the results using the defined metrics.
    Returns:
        tuple: A tuple containing the detection result for astigmatism and the duration for the metric calculation.
    """
    psfGen = PSFRandomParameter(size=PSF_SIZE, aberrationType="astigmatism")
    psf = psfGen.psf
    metricTool = MetricTool()
    metricTool._pixelSize = (0.05, 0.05, 0.05)
    if NOISE:
        psf = addMicroscopyNoise(psf)
    metricTool._image = psf
    
    fitTool1D = Fitting1D()
    fitTool1D._image = psf
    fitTool1D._show = False
    fitTool1D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool1D._roi = [np.array([0, 0, 0])]
    
    start_time = time.time()
    fitTool1D.processSingleFit(0)
    metricTool.astigmatism(mu=fitTool1D.parameters[2:5], sigma=fitTool1D.parameters[5:8])
    duration_astigmatism = time.time() - start_time
    
    astigmatism_detected = metricTool._astigmatism > 0.05 or metricTool._astigmatism < -0.05
    
    return astigmatism_detected, duration_astigmatism

def test_astigmatism_aberration_without_aberration_single():
    """Tests the detection of astigmatism aberration in a PSF generated without astigmatism aberration.
    This function generates a PSF without astigmatism aberration, applies various methods, and evaluates the results using the defined metrics.
    Returns:
        tuple: A tuple containing the detection result for astigmatism and the duration for the metric calculation.
    """
    psfGen = PSFRandomParameter(size=PSF_SIZE)
    psf = psfGen.psf
    metricTool = MetricTool()
    metricTool._pixelSize = (0.05, 0.05, 0.05)
    if NOISE:
        psf = addMicroscopyNoise(psf)
    metricTool._image = psf
    
    fitTool1D = Fitting1D()
    fitTool1D._image = psf
    fitTool1D._show = False
    fitTool1D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool1D._roi = [np.array([0, 0, 0])]
    
    start_time = time.time()
    fitTool1D.processSingleFit(0)
    metricTool.astigmatism(mu=fitTool1D.parameters[2:5], sigma=fitTool1D.parameters[5:8])
    duration_astigmatism = time.time() - start_time
    
    astigmatism_detected = metricTool._astigmatism <= 0.05 and metricTool._astigmatism >= -0.05
    
    return astigmatism_detected, duration_astigmatism

def test_astigmatism_aberration():
    """Test function for evaluating metrics of astigmatism aberration detection.
    This function generates a PSF with astigmatism aberration, applies various methods, and evaluates the results using the defined metrics.
    The test assesses the accuracy of the methods in estimating the PSF parameters in the presence of astigmatism aberration.
    """
    Mean_Duration_Astigmatism = 0.0
    tp_astig = 0
    
    tdqm = tqdm(total=200, desc="Evaluating Astigmatism Aberration Detection", unit="test")
    
    with ProcessPoolExecutor(max_workers=int(os.cpu_count() * 0.75), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(test_astigmatism_aberration_with_aberration_single) for _ in range(100)]
        for future in as_completed(futures):
            detected, duration_astigmatism = future.result()
            Mean_Duration_Astigmatism += duration_astigmatism
            if detected:
                tp_astig += 1
            tdqm.update(1)
    
    Rappel_astigmatism = tp_astig / 100
    
    tn_astig = 0
    
    with ProcessPoolExecutor(max_workers=int(os.cpu_count() * 0.75), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(test_astigmatism_aberration_without_aberration_single) for _ in range(100)]
        for future in as_completed(futures):
            detected, duration_astigmatism = future.result()
            Mean_Duration_Astigmatism += duration_astigmatism
            if detected:
                tn_astig += 1
            tdqm.update(1)
    
    Precision_astigmatism = tp_astig / (tp_astig + (100 - tn_astig)) if (tp_astig + (100-tn_astig)) > 0 else 0.0
    
    Accuracy_astigmatism = (tp_astig + tn_astig) / 200
    
    F1_astigmatism = 2 * (Precision_astigmatism * Rappel_astigmatism) / (Precision_astigmatism + Rappel_astigmatism) if (Precision_astigmatism + Rappel_astigmatism) > 0 else 0.0
    
    print(f"Astigmatism Aberration - Precision: {Precision_astigmatism:.2f}, Recall: {Rappel_astigmatism:.2f}, Accuracy: {Accuracy_astigmatism:.2f}, F1 Score: {F1_astigmatism:.2f}, Mean Duration: {Mean_Duration_Astigmatism/200:.2f} seconds")
    
def BenchMetrics():
    """Runs the benchmark tests for comatic aberration, spherical aberration, and astigmatism aberration detection metrics."""
    #test_comatic_aberration()
    test_spherical_aberration()
    #test_astigmatism_aberration()

if __name__ == "__main__":
    #showPSFvsNoisy()
    BenchMetrics()