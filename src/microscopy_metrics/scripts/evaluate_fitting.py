import os
import time
import random
import numpy as np
import matplotlib

import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from microscopy_metrics.fittingTools.fitting3D import Fitting3D
from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.scripts.PSFGenerator.PSF import PSFRandomParameter
from microscopy_metrics.fittingTools.fitting3DRotation import Fitting3DRotation

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

GIBSONLANNI = False
ORIENTED = True
PSF_SIZE = 80
NOISE = False
FIT_METHODS = ["1D", "2D", "3D", "2DRotation", "3DRotation", "2DEllips", "Prominence"]

TRUE_AMP = 255.0
TRUE_BG = 0.0
TRUE_MU_X = PSF_SIZE / 2
TRUE_MU_Y = PSF_SIZE / 2
TRUE_MU_Z = PSF_SIZE / 2
TRUE_SIGMA_X = PSF_SIZE / 10
TRUE_SIGMA_Y = PSF_SIZE / 10
TRUE_SIGMA_Z = PSF_SIZE / 10


def show2DPsf(psf, center):
    """Displays a 2D slice of the Point Spread Function (PSF) image at the specified center index along the Z-axis.
    
    Args:
        psf (np.ndarray): The 3D PSF image to be visualized.
        center (int): The index along the Z-axis at which to take the 2D slice for visualization.
    """
    plt.imshow(psf[:, center, :], cmap="viridis")
    plt.colorbar()
    plt.title("2D Image of a Gaussian")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def computeMape(estimations, truths):
    """Computes the Mean Absolute Percentage Error (MAPE) between the estimated values and the true values.
    
    Args:
        estimations (list): A list of estimated values.
        truths (list): A list of true values corresponding to the estimations.
    
    Returns:
        float: The calculated MAPE value, expressed as a percentage. If there are no valid comparisons, returns infinity.
    """
    errors = []
    for est, truth in zip(estimations, truths):
        if truth == 0:
            continue
        error = abs(est - truth) / truth
        errors.append(error)
    return (sum(errors) / len(errors)) * 100 if errors else float("inf")


def get_rotation_matrix(thetaX, thetaY, thetaZ):
    cx, sx = np.cos(thetaX), np.sin(thetaX)
    cy, sy = np.cos(thetaY), np.sin(thetaY)
    cz, sz = np.cos(thetaZ), np.sin(thetaZ)
    R = np.array(
        [
            [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
            [cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz],
            [-sy, cy * sx, cy * cx],
        ]
    )
    return R


def computeAngularError(estimations, truths):
    """Computes the angular error between the estimated rotation angles and the true rotation angles.
    
    Args:
        estimations (list): A list of estimated rotation angles [thetaX, thetaY, thetaZ].
        truths (list): A list of true rotation angles [thetaX, thetaY, thetaZ].
    
    Returns:
        float: The calculated angular error in radians. If the fit method is not "2D" or "3D", returns infinity.
    """
    R_est = get_rotation_matrix(estimations[0], estimations[1], estimations[2])
    R_true = get_rotation_matrix(truths[0], truths[1], truths[2])
    R_diff = R_est @ R_true.T
    trace = np.trace(R_diff)
    err_x = np.arccos((trace - 1) / 2)
    return err_x


def computeMSE(estimations, truths):
    """Computes the Mean Squared Error (MSE) between the estimated values and the true values.
    
    Args:
        estimations (list): A list of estimated values.
        truths (list): A list of true values corresponding to the estimations.
    
    Returns:
        float: The calculated MSE value. If there are no valid comparisons, returns infinity.
    """
    return np.mean((np.array(estimations) - np.array(truths)) ** 2)


def computePSNR(estimations, truths):
    """Computes the Peak Signal-to-Noise Ratio (PSNR) between the estimated values and the true values.
    
    Args:
        estimations (list): A list of estimated values.
        truths (list): A list of true values corresponding to the estimations.
    
    Returns:
        float: The calculated PSNR value in decibels (dB). If the MSE is zero, returns infinity.
    """
    mse = computeMSE(estimations, truths)
    if mse == 0:
        return float("inf")
    psnr = 10 * np.log10((TRUE_AMP**2) / mse)
    return psnr


def computeBhattacharyyaDistance(mu1, mu2, sigma1, sigma2):
    """Computes the Bhattacharyya distance between two Gaussian distributions defined by their means and standard deviations.
    
    Args:
        mu1 (float): Mean of the first Gaussian distribution.
        mu2 (float): Mean of the second Gaussian distribution.
        sigma1 (float): Standard deviation of the first Gaussian distribution.
        sigma2 (float): Standard deviation of the second Gaussian distribution.
    
    Returns:
        float: The calculated Bhattacharyya distance between the two distributions.
    """
    return 0.25 * ((mu1 - mu2) ** 2 / (sigma1**2 + sigma2**2)) + 0.5 * np.log(
        (sigma1**2 + sigma2**2) / (2 * sigma1 * sigma2)
    )


def generateRandomPSFParams(psf_size=10):
    """Generates random parameters for a Point Spread Function (PSF) based on the specified PSF size.
    
    Args:
        psf_size (int): The size of the PSF, used to determine the range for generating random parameters.
    
    Returns:
        list: A list containing the generated parameters [amp, bg, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z].
    """
    mu_x = psf_size * 0.5
    mu_y = psf_size * 0.5
    mu_z = psf_size * 0.5
    sigmaDefault = psf_size / 10.0
    sigma_x = np.random.uniform(0.95 * psf_size / 7.0, 1.05 * sigmaDefault)
    sigma_y = np.random.uniform(0.95 * sigmaDefault, 1.05 * sigmaDefault)
    sigma_z = np.random.uniform(0.95 * sigmaDefault, 1.05 * sigmaDefault)
    amp = TRUE_AMP
    bg = 0.0
    return [amp, bg, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z]


def addMicroscopyNoise(image, maxPhotons=1000, readoutNoiseStd=5.0):
    """Adds realistic microscopy noise to the input image, simulating both Poisson noise (photon counting noise) and Gaussian readout noise.
    
    Args:
        image (np.ndarray): The input image to which noise will be added.
        maxPhotons (int): The maximum number of photons for scaling the image before adding noise.
        readoutNoiseStd (float): The standard deviation of the Gaussian readout noise.
    
    Returns:
        np.ndarray: The noisy image with added Poisson and Gaussian noise, clipped to the range [0, maxPhotons].
    """
    scaledImage = (image / np.max(image)) * maxPhotons
    noisyPoisson = np.random.poisson(scaledImage).astype(np.float32)
    gaussianNoise = np.random.normal(0, readoutNoiseStd, size=image.shape)
    finalNoisyImage = noisyPoisson + gaussianNoise
    finalNoisyImage = np.minimum(np.maximum(finalNoisyImage, 0), maxPhotons)
    return finalNoisyImage


def generatePSF(params):
    """Generates a Point Spread Function (PSF) based on the provided parameters using the Fitting3D class.
    
    Args:
        params (list): A list containing the parameters [amp, bg, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z].
    
    Returns:
        tuple: A tuple containing the generated PSF, coordinates, and FWHM values.
    """
    fitTool = Fitting3D()
    fitTool._show = False
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    psf = fitTool.gauss(*params)(coords)
    fwhm = [fitTool.fwhm(params[5]), fitTool.fwhm(params[6]), fitTool.fwhm(params[7])]
    return psf, coords, fwhm


def generateRandomGibsonLanniPSF(seed=None):
    """Generates a random Point Spread Function (PSF) based on the Gibson-Lanni model using the PSFRandomParameter class.
    
    Args:
        seed (int, optional): The random seed for reproducibility.
    
    Returns:
        tuple: A tuple containing the generated PSF, coordinates, and FWHM values.
    """
    if seed is not None:
        np.random.seed(seed)
    gl = PSFRandomParameter(size=PSF_SIZE)
    fwhm = gl.fwhm
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    psf = gl.psf
    return psf, coords, fwhm


def generateOrientedPSF(params, seed=None):
    """Generates an oriented Point Spread Function (PSF) based on the provided parameters and applies a random rotation to it.
    
    Args:
        params (list): A list containing the parameters [amp, bg, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z].
        seed (int, optional): The random seed for reproducibility.
    
    Returns:
        tuple: A tuple containing the generated PSF, coordinates, and FWHM values.
    """
    if seed is not None:
        np.random.seed(seed)
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    index = random.randint(0, 2)
    thetas = [0.0, 0.0, 0.0]
    thetas[index] = np.deg2rad(45)
    fitTool = Fitting3DRotation()
    params[5] *= 1.5
    psf = fitTool.gauss(*params, *thetas)(coords)
    fwhm = [fitTool.fwhm(params[7]), fitTool.fwhm(params[6]), fitTool.fwhm(params[5])]
    return psf, coords, fwhm, thetas


def fitPSF(fitName, image):
    """Fits the provided Point Spread Function (PSF) image using the specified fitting method and returns the fitting results along with the elapsed time.
    
    Args:
        fitName (str): The name of the fitting method to be used (e.g., "1D", "2D", "3D").
        image (np.ndarray): The PSF image to be fitted.
    
    Returns:
        tuple: A tuple containing the fitting results and the elapsed time for the fitting process.
    """
    fitTool = FittingTool.getInstance(fitName)
    fitTool._image = image
    fitTool._show = False
    fitTool._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool._roi = [np.array([0, 0, 0])]
    if GIBSONLANNI:
        fitTool._pixelSize = [0.05, 0.05, 0.05]
    else:
        fitTool._pixelSize = [0.1, 0.1, 0.1]
    fitTool._outputDir = Path(__file__).parent / "EvalFitResult" / fitName
    start = time.time()
    fitTool.processSingleFit(0)
    end = time.time()
    elapsed = end - start
    result = [
        0,
        fitTool.fwhms,
        fitTool.uncertainties,
        fitTool.determinations,
        fitTool.parameters,
        fitTool.pcovs,
        fitTool.thetas,
    ]
    return result, elapsed


def getBhattacharyyaFit(params, mu, sigma):
    """Calculates the average Bhattacharyya distance between the fitted parameters and the true parameters for a Point Spread Function (PSF).
    
    Args:
        params (list): A list containing the fitted parameters [amp, bg, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z].
        mu (list): A list containing the true means [mu_x, mu_y, mu_z].
        sigma (list): A list containing the true standard deviations [sigma_x, sigma_y, sigma_z].
    
    Returns:
        float: The average Bhattacharyya distance between the fitted and true parameters.
    """
    DistBat = 0.0
    DistBat += computeBhattacharyyaDistance(params[2], mu[0], params[5], sigma[0])
    DistBat += computeBhattacharyyaDistance(params[3], mu[1], params[6], sigma[1])
    DistBat += computeBhattacharyyaDistance(params[4], mu[2], params[7], sigma[2])
    DistBat /= 3.0
    return DistBat


def evaluateXDPsf(
    psf, psfReshape, FWHM, coords, params=None, fitMethod="2D", thetas=None
):
    """Evaluates the fitting of a Point Spread Function (PSF) using the specified fitting method and computes various metrics such as correlation, PSNR, Bhattacharyya distance, determination coefficient, elapsed time, and orientation correlation.
    
    Args:
        psf (np.ndarray): The original PSF image.
        psfReshape (np.ndarray): The reshaped PSF image for fitting.
        FWHM (list): A list containing the true Full Width at Half Maximum (FWHM) values.
        coords (np.ndarray): The coordinates corresponding to the PSF image.
        params (list, optional): A list containing the true parameters [amp, bg, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z].
        fitMethod (str): The name of the fitting method to be used (e.g., "1D", "2D", "3D").
        thetas (list, optional): A list containing the true rotation angles [thetaX, thetaY, thetaZ].
    
    Returns:
        tuple: A tuple containing the computed metrics (correlation, PSNR, Bhattacharyya distance, determination coefficient, elapsed time, orientation correlation).
    """
    result, elapsed = fitPSF(fitMethod, psfReshape)
    corr = computeMape(result[1], FWHM)
    fit = Fitting3D().gauss(*result[4])(coords)

    psnr = computePSNR(fit, psf.flatten())
    mu = [result[4][2], result[4][3], result[4][4]]
    sigma = [result[4][5], result[4][6], result[4][7]]
    if params is not None:
        DistBat = getBhattacharyyaFit(params, mu, sigma)
    else:
        DistBat = 0
    determination = (result[3][0] + result[3][1] + result[3][2]) / 3.0
    if thetas is not None:
        corrOrientation = computeAngularError(result[6], thetas)
    else:
        corrOrientation = np.inf
    return corr, psnr, DistBat, determination, elapsed, corrOrientation


def evaluatePsf(seed=None):
    """Evaluates the fitting of a Point Spread Function (PSF) by generating a random PSF, optionally adding noise, and then fitting it using various fitting methods. Computes metrics such as correlation, PSNR, Bhattacharyya distance, determination coefficient, elapsed time, and orientation correlation for each fitting method.
    
    Args:
        seed (int, optional): The random seed for reproducibility.
    
    Returns:
        tuple: A tuple containing lists of computed metrics for each fitting method (correlation, PSNR, Bhattacharyya distance, determination coefficient, elapsed time, orientation correlation).
    """
    thetas = None
    params = generateRandomPSFParams(PSF_SIZE)
    if GIBSONLANNI and not ORIENTED:
        psf, coords, FWHM = generateRandomGibsonLanniPSF(seed=seed)
        params = None
    elif ORIENTED:
        psf, coords, FWHM, thetas = generateOrientedPSF(params, seed=seed)
        params = None
    else:
        psf, coords, FWHM = generatePSF(params)
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    if NOISE:
        psfReshape = addMicroscopyNoise(psfReshape, maxPhotons=TRUE_AMP)
    corr1D, psnr1D, DistBat1D, determination1D, elapsed1D, corrOrientation1D = (
        evaluateXDPsf(psf, psfReshape, FWHM, coords, params, "1D")
    )
    corr2D, psnr2D, DistBat2D, determination2D, elapsed2D, corrOrientation2D = (
        evaluateXDPsf(psf, psfReshape, FWHM, coords, params, "2D")
    )
    corr3D, psnr3D, DistBat3D, determination3D, elapsed3D, corrOrientation3D = (
        evaluateXDPsf(psf, psfReshape, FWHM, coords, params, "3D")
    )

    result, elapsedProminence = fitPSF("Prominence", psfReshape)
    corrProminence = computeMape(result[1], FWHM)
    corrOrientationProminence = np.inf

    (
        corr2DEllips,
        psnr2DEllips,
        DistBat2DEllips,
        determination2DEllips,
        elapsed2DEllips,
        corrOrientation2DEllips,
    ) = evaluateXDPsf(psf, psfReshape, FWHM, coords, params, "2D Ellipse", thetas)
    (
        corr2DRotation,
        psnr2DRotation,
        DistBat2DRotation,
        determination2DRotation,
        elapsed2DRotation,
        corrOrientation2DRotation,
    ) = evaluateXDPsf(psf, psfReshape, FWHM, coords, params, "2D rotation", thetas)
    (
        corr3DRotation,
        psnr3DRotation,
        DistBat3DRotation,
        determination3DRotation,
        elapsed3DRotation,
        corrOrientation3DRotation,
    ) = evaluateXDPsf(psf, psfReshape, FWHM, coords, params, "3D Rotation", thetas)
    return (
        [
            corr1D,
            corr2D,
            corr3D,
            corr2DRotation,
            corr3DRotation,
            corr2DEllips,
            corrProminence,
        ],
        [psnr1D, psnr2D, psnr3D, psnr2DRotation, psnr3DRotation, psnr2DEllips],
        [
            DistBat1D,
            DistBat2D,
            DistBat3D,
            DistBat2DRotation,
            DistBat3DRotation,
            DistBat2DEllips,
        ],
        [
            determination1D,
            determination2D,
            determination3D,
            determination2DRotation,
            determination3DRotation,
            determination2DEllips,
        ],
        [
            elapsed1D,
            elapsed2D,
            elapsed3D,
            elapsed2DRotation,
            elapsed3DRotation,
            elapsed2DEllips,
            elapsedProminence,
        ],
        [
            corrOrientation1D,
            corrOrientation2D,
            corrOrientation3D,
            corrOrientation2DRotation,
            corrOrientation3DRotation,
            corrOrientation2DEllips,
            corrOrientationProminence,
        ],
    )


def addTab(tabA, tabB):
    """Adds two lists element-wise, returning a new list containing the sums of corresponding elements.
    
    Args:
        tabA (list): The first list of values.
        tabB (list): The second list of values.
    
    Returns:
        list: A new list containing the element-wise sums of tabA and tabB. The length of the returned list is the minimum of the lengths of tabA and tabB.
    """
    minRange = min(len(tabA), len(tabB))
    tabC = []
    for i in range(minRange):
        tabC.append(tabA[i] + tabB[i])
    return tabC


def divTab(tab, div):
    """Divides each element of a list by a specified divisor, returning a new list containing the results.
    
    Args:
        tab (list): The list of values to be divided.
        div (float): The divisor by which each element of the list will be divided.
    
    Returns:
        list: A new list containing the results of dividing each element of tab by div.
    """
    return [x / div for x in tab]


def superior(a, b):
    """Checks if the first argument is strictly greater than the second.
    
    Args:
        a (float): The first value.
        b (float): The second value.
    
    Returns:
        bool: True if a is strictly greater than b, False otherwise.
    """
    return a > b


def inferior(a, b):
    """Checks if the first argument is strictly less than the second.
    
    Args:
        a (float): The first value.
        b (float): The second value.
    
    Returns:
        bool: True if a is strictly less than b, False otherwise.
    """
    return a < b


def printResults(metricStr, metric, unitStr, comparison=superior):
    """Prints the results of the evaluation for a specific metric, including the mean values for each fitting method and identifies the best fitting method based on the specified comparison function.
    
    Args:
        metricStr (str): The name of the metric being evaluated (e.g., "MAPE", "PSNR").
        metric (list): A list of mean values for the specified metric corresponding to each fitting method.
        unitStr (str): The unit of measurement for the metric (e.g., "%", "dB").
        comparison (function): A function that defines how to compare two values to determine which is better. Defaults to the 'superior' function.
    """
    print("=" * 20, metricStr.upper(), "=" * 20)
    best_fit = FIT_METHODS[0]
    best_value = metric[0]
    print(f"Mean {metricStr} {FIT_METHODS[0]}: {metric[0]:.8f} {unitStr}")
    for i in range(1, len(metric)):
        if comparison(metric[i], best_value):
            best_fit = FIT_METHODS[i]
            best_value = metric[i]
        elif metric[i] == best_value:
            best_fit += f" and {FIT_METHODS[i]}"
        print(f"Mean {metricStr} {FIT_METHODS[i]}: {metric[i]:.8f} {unitStr}")
    print("-" * 50)
    print(
        f"The best fitting method is: {best_fit} | with a {metricStr} of {best_value:.8f} {unitStr}",
        "\n",
    )


if __name__ == "__main__":
    params = generateRandomPSFParams(PSF_SIZE)
    if GIBSONLANNI and not ORIENTED:
        psf, coords, FWHM = generateRandomGibsonLanniPSF(seed=None)
    elif ORIENTED:
        psf, coords, FWHM, _ = generateOrientedPSF(params)
    else:
        psf, coords, FWHM = generatePSF(params)
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    if NOISE:
        psfReshape = addMicroscopyNoise(psfReshape)
    show2DPsf(psfReshape, int(PSF_SIZE / 2))

    meanCorr = [0 for _ in range(len(FIT_METHODS))]
    meanPSNR = [0, 0, 0, 0, 0, 0, 0]
    meanBat = [0, 0, 0, 0, 0, 0, 0]
    meanDetermination = [0, 0, 0, 0, 0, 0, 0]
    meanDuration = [0 for _ in range(len(FIT_METHODS))]
    meanCorrOrientation = [0.0 for _ in range(len(FIT_METHODS))]

    n_tests = 100

    pbar = tqdm(total=n_tests, desc="Evaluating PSF Fitting", unit="test")
    workers = int(os.cpu_count() * 0.75)

    randomSeeds = np.random.randint(0, 2**32, size=n_tests)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = {executor.submit(evaluatePsf, seed=seed) for seed in randomSeeds}
        for future in as_completed(futures):
            corr, psnr, bat, determination, duration, corrOrientation = future.result()
            meanCorr = addTab(meanCorr, corr)
            meanPSNR = addTab(meanPSNR, psnr)
            meanBat = addTab(meanBat, bat)
            meanDetermination = addTab(meanDetermination, determination)
            meanDuration = addTab(meanDuration, duration)
            meanCorrOrientation = addTab(meanCorrOrientation, corrOrientation)
            pbar.update(1)
            pbar.set_postfix(
                {
                    "MAPE 1D": f"{corr[0]:.8f}%",
                    "MAPE 2D": f"{corr[1]:.8f}%",
                    "MAPE 3D": f"{corr[2]:.8f}%",
                    "PSNR 1D": f"{psnr[0]:.2f}dB",
                    "PSNR 2D": f"{psnr[1]:.2f}dB",
                    "PSNR 3D": f"{psnr[2]:.2f}dB",
                }
            )
    meanCorr = divTab(meanCorr, n_tests)
    meanPSNR = divTab(meanPSNR, n_tests)
    meanBat = divTab(meanBat, n_tests)
    meanDetermination = divTab(meanDetermination, n_tests)
    meanDuration = divTab(meanDuration, n_tests)
    meanCorrOrientation = divTab(meanCorrOrientation, n_tests)

    print("\n\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50, "\n")

    printResults("MAPE", meanCorr, "%", inferior)
    printResults("PSNR", meanPSNR, "dB", superior)
    printResults("distance", meanBat, "", inferior)
    printResults("R2 score", meanDetermination, "", superior)
    printResults("duration", meanDuration, "seconds", inferior)
    printResults("orientation", meanCorrOrientation, "°", inferior)

    print("=" * 50)
    print("END SUMMARY")
    print("=" * 50, "\n")
