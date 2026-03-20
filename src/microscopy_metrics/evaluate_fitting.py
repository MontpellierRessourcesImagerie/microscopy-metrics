from microscopy_metrics.fitting import FittingTool, Fitting3D, Fitting1D, Fitting2D
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from pathlib import Path
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

PSF_SIZE = 80

TRUE_AMP = 1.0
TRUE_BG = 0.0
TRUE_MU_X = PSF_SIZE / 2
TRUE_MU_Y = PSF_SIZE / 2
TRUE_MU_Z = PSF_SIZE / 2
TRUE_SIGMA_X = PSF_SIZE / 10
TRUE_SIGMA_Y = PSF_SIZE / 10
TRUE_SIGMA_Z = PSF_SIZE / 10


def show3DPsf(X, Y, Z, psf_3d):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    alphas = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)

    ax.scatter(X, Y, Z, c=psf_3d, cmap="viridis", s=1, alpha=alphas)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("PSF 3D (Gaussienne)")
    mappable = plt.cm.ScalarMappable(cmap="viridis")
    mappable.set_array(psf_3d)
    plt.colorbar(mappable, ax=ax, label="Intensité")
    plt.show()


def show2DPsf(psf, center):
    plt.imshow(psf[center], cmap="viridis")
    plt.colorbar()
    plt.title("2D Image of a Gaussian")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def computeMape(estimations, truths):
    errors = []
    for est, truth in zip(estimations, truths):
        if truth == 0:
            continue 
        error = abs(est - truth) / truth
        errors.append(error)
    return (sum(errors) / len(errors)) * 100 if errors else float("inf")


def computeMSE(estimations, truths):
    return np.mean((np.array(estimations) - np.array(truths)) ** 2)


def computePSNR(estimations, truths, maxI=1.0):
    mse = computeMSE(estimations, truths)
    if mse == 0:
        return float("inf")
    psnr = 10 * np.log10((maxI**2) / mse)
    return psnr


def computeBhattacharyyaDistance(mu1,mu2,sigma1,sigma2):
    return 0.25 * ((mu1 - mu2)**2 / (sigma1**2 + sigma2 **2)) + 0.5*np.log((sigma1**2 + sigma2**2)/(2*sigma1*sigma2))


def generateRandomPSFParams(psf_size=10):
    mu_x = psf_size * 0.5
    mu_y = psf_size * 0.5
    mu_z = psf_size * 0.5

    sigmaDefault = psf_size / 10.0
    sigma_x = np.random.uniform(0.95 * sigmaDefault, 1.05 * sigmaDefault)
    sigma_y = np.random.uniform(0.95 * sigmaDefault, 1.05 * sigmaDefault)
    sigma_z = np.random.uniform(0.95 * sigmaDefault, 1.05 * sigmaDefault)

    amp = 1.0
    bg = 0.0
    return [amp, bg, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z]


def addMicroscopyNoise(image, maxPhotons=1000, readoutNoiseStd=5.0):
    scaledImage = (image / np.max(image)) * maxPhotons
    noisyPoisson = np.random.poisson(scaledImage).astype(np.float32)
    gaussianNoise = np.random.normal(0, readoutNoiseStd, size=image.shape)
    finalNoisyImage = noisyPoisson + gaussianNoise
    finalNoisyImage = np.maximum(finalNoisyImage, 0)
    return finalNoisyImage


def evaluatePsf():
    fitTool = Fitting3D()
    fitTool._show = False
    params = [
        TRUE_AMP,
        TRUE_BG,
        TRUE_MU_X,
        TRUE_MU_Y,
        TRUE_MU_Z,
        TRUE_SIGMA_X,
        TRUE_SIGMA_Y,
        TRUE_SIGMA_Z,
    ]
    params = generateRandomPSFParams(PSF_SIZE)
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    psf = fitTool.gauss(*params)(coords)
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    psfReshapeTest = psfReshape
    psfReshape = addMicroscopyNoise(psfReshape)
    FWHM = [fitTool.fwhm(params[5]), fitTool.fwhm(params[6]), fitTool.fwhm(params[7])]
    fitTool1D = Fitting1D()
    fitTool1D._image = psfReshape
    fitTool1D._show = False
    fitTool1D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool1D._roi = [np.array([0, 0, 0])]
    fitTool1D._outputDir = Path(__file__).parent / "EvalFitResult" / "1D"
    start = time.time()
    result = fitTool1D.processSingleFit(0)
    end = time.time()
    elapsed1D = end - start
    corr1D = computeMape(result[1], FWHM)
    amp = (result[4][0][0] + result[4][1][0] + result[4][2][0]) / 3.0
    bg = (result[4][0][1] + result[4][1][1] + result[4][2][1]) / 3.0
    mu = [result[4][0][2], result[4][1][2], result[4][2][2]]
    sigma = [result[4][0][3], result[4][1][3], result[4][2][3]]
    params1D = [amp, bg, *mu, *sigma]
    fit = Fitting3D().gauss(*params1D)(coords)
    psnr1D = computePSNR(fit, psf)
    center = int(PSF_SIZE / 2)
    fit = fit.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    DistBat1D = 0.0
    DistBat1D += computeBhattacharyyaDistance(params[2],mu[0],params[5],sigma[0])
    DistBat1D += computeBhattacharyyaDistance(params[3],mu[1],params[6],sigma[1])
    DistBat1D += computeBhattacharyyaDistance(params[4],mu[2],params[7],sigma[2])
    DistBat1D /= 3.0
    determination1D = (result[3][0] + result[3][1] + result[3][2]) / 3.0

    fitTool2D = Fitting2D()
    fitTool2D._image = psfReshape
    fitTool2D._show = False
    fitTool2D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool2D._roi = [np.array([0, 0, 0])]
    fitTool2D._outputDir = Path(__file__).parent / "EvalFitResult" / "2D"
    start = time.time()
    result = fitTool2D.processSingleFit(0)
    end = time.time()
    elapsed2D = end - start
    corr2D = computeMape(result[1], FWHM)
    fit = Fitting3D().gauss(*result[4])(coords)
    psnr2D = computePSNR(fit, psf)
    fit = fit.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    mu = [result[4][2], result[4][3], result[4][4]]
    sigma = [result[4][5], result[4][6], result[4][7]]
    DistBat2D = 0.0
    DistBat2D += computeBhattacharyyaDistance(params[2],mu[0],params[5],sigma[0])
    DistBat2D += computeBhattacharyyaDistance(params[3],mu[1],params[6],sigma[1])
    DistBat2D += computeBhattacharyyaDistance(params[4],mu[2],params[7],sigma[2])
    DistBat2D /= 3.0
    determination2D = (result[3][0] + result[3][1] + result[3][2]) / 3.0

    fitTool3D = Fitting3D()
    fitTool3D._image = psfReshape
    fitTool3D._show = False
    fitTool3D._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool3D._roi = [np.array([0, 0, 0])]
    fitTool3D._outputDir = Path(__file__).parent / "EvalFitResult" / "3D"
    start = time.time()
    result = fitTool3D.processSingleFit(0)
    end = time.time()
    elapsed3D = end - start
    corr3D = computeMape(result[1], FWHM)
    fit = Fitting3D().gauss(*result[4])(coords)
    psnr3D = computePSNR(fit, psf)
    fit = fit.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    mu = [result[4][2], result[4][3], result[4][4]]
    sigma = [result[4][5], result[4][6], result[4][7]]
    DistBat3D = 0.0
    DistBat3D += computeBhattacharyyaDistance(params[2],mu[0],params[5],sigma[0])
    DistBat3D += computeBhattacharyyaDistance(params[3],mu[1],params[6],sigma[1])
    DistBat3D += computeBhattacharyyaDistance(params[4],mu[2],params[7],sigma[2])
    DistBat3D /= 3.0
    determination3D = (result[3][0] + result[3][1] + result[3][2]) / 3.0
    return (
        corr1D,
        corr2D,
        corr3D,
        psnr1D,
        psnr2D,
        psnr3D,
        DistBat1D,
        DistBat2D,
        DistBat3D,
        determination1D,
        determination2D,
        determination3D,
        elapsed1D,
        elapsed2D,
        elapsed3D,
    )


fitTool = Fitting3D()
fitTool._show = False
params = [
    TRUE_AMP,
    TRUE_BG,
    TRUE_MU_X,
    TRUE_MU_Y,
    TRUE_MU_Z,
    TRUE_SIGMA_X,
    TRUE_SIGMA_Y,
    TRUE_SIGMA_Z,
]
params = generateRandomPSFParams(PSF_SIZE)
zz = np.arange(PSF_SIZE)
yy = np.arange(PSF_SIZE)
xx = np.arange(PSF_SIZE)
x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
psf = fitTool.gauss(*params)(coords)
psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
psfReshape = addMicroscopyNoise(psfReshape)
show2DPsf(psfReshape, int(PSF_SIZE / 2))
meanCorr1D, meanCorr2D, meanCorr3D = 0, 0, 0
meanPSNR1D, meanPSNR2D, meanPSNR3D = 0, 0, 0
meanBat1D, meanBat2D, meanBat3D = 0, 0, 0
meanDetermination1D, meanDetermination2D, meanDetermination3D = 0, 0, 0
meanDuration1D, meanDuration2D, meanDuration3D = 0, 0, 0
n_tests = 100
pbar = tqdm(total=n_tests, desc="Evaluating PSF Fitting", unit="test")
workers = int(os.cpu_count() * 0.75)
with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {executor.submit(evaluatePsf) for _ in range(n_tests)}
    for future in as_completed(futures):
        (
            corr1D,
            corr2D,
            corr3D,
            psnr1D,
            psnr2D,
            psnr3D,
            DistBat1D,
            DistBat2D,
            DistBat3D,
            determination1D,
            determination2D,
            determination3D,
            duration1D,
            duration2D,
            duration3D,
        ) = future.result()
        meanCorr1D += corr1D
        meanCorr2D += corr2D
        meanCorr3D += corr3D
        meanPSNR1D += psnr1D
        meanPSNR2D += psnr2D
        meanPSNR3D += psnr3D
        meanBat1D += DistBat1D
        meanBat2D += DistBat2D
        meanBat3D += DistBat3D
        meanDetermination1D += determination1D
        meanDetermination2D += determination2D
        meanDetermination3D += determination3D
        meanDuration1D += duration1D
        meanDuration2D += duration2D
        meanDuration3D += duration3D
        pbar.update(1)
        pbar.set_postfix(
            {
                "MAPE 1D": f"{corr1D:.8f}%",
                "MAPE 2D": f"{corr2D:.8f}%",
                "MAPE 3D": f"{corr3D:.8f}%",
                "PSNR 1D": f"{psnr1D:.2f}dB",
                "PSNR 2D": f"{psnr2D:.2f}dB",
                "PSNR 3D": f"{psnr3D:.2f}dB",
            }
        )
meanCorr1D /= n_tests
meanCorr2D /= n_tests
meanCorr3D /= n_tests
meanPSNR1D /= n_tests
meanPSNR2D /= n_tests
meanPSNR3D /= n_tests
meanBat1D /= n_tests
meanBat2D /= n_tests
meanBat3D /= n_tests
meanDetermination1D /= n_tests
meanDetermination2D /= n_tests
meanDetermination3D /= n_tests
meanDuration1D /= n_tests
meanDuration2D /= n_tests
meanDuration3D /= n_tests

best_fit = "1D"
best_value = meanCorr1D
if meanCorr2D < best_value:
    best_fit = "2D"
    best_value = meanCorr2D
if meanCorr3D < best_value:
    best_fit = "3D"
    best_value = meanCorr3D

print("\n" + "=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"Mean MAPE 1D: {meanCorr1D:.8f}%")
print(f"Mean MAPE 2D: {meanCorr2D:.8f}%")
print(f"Mean MAPE 3D: {meanCorr3D:.8f}%")
print("-" * 50)
print(f"The best fitting method is: {best_fit} with a MAPE of {best_value:.8f}%")
print("=" * 50)

best_fit = "1D"
best_value = meanPSNR1D
if meanPSNR2D > best_value:
    best_fit = "2D"
    best_value = meanPSNR2D
if meanPSNR3D > best_value:
    best_fit = "3D"
    best_value = meanPSNR3D

print(f"Mean PSNR 1D: {meanPSNR1D:.2f}dB")
print(f"Mean PSNR 2D: {meanPSNR2D:.2f}dB")
print(f"Mean PSNR 3D: {meanPSNR3D:.2f}dB")
print("-" * 50)
print(f"The best fitting method is: {best_fit} with a PSNR of {best_value:.2f}dB")
print("=" * 50)

best_fit = "1D"
best_value = meanBat1D
if meanBat2D < best_value:
    best_fit = "2D"
    best_value = meanBat2D
if meanBat3D < best_value:
    best_fit = "3D"
    best_value = meanBat3D

print(f"Mean distance 1D: {meanBat1D:.10f}")
print(f"Mean distance 2D: {meanBat2D:.10f}")
print(f"Mean distance 3D: {meanBat3D:.10f}")
print("-" * 50)
print(f"The best fitting method is: {best_fit} with a distance of {best_value:.10f}")
print("=" * 50)

best_fit = "1D"
best_value = meanDetermination1D
if meanDetermination2D > best_value:
    best_fit = "2D"
    best_value = meanDetermination2D
if meanDetermination3D > best_value:
    best_fit = "3D"
    best_value = meanDetermination3D

print(f"Mean determination (R^2) 1D: {meanDetermination1D}")
print(f"Mean determination (R^2) 2D: {meanDetermination2D}")
print(f"Mean determination (R^2) 3D: {meanDetermination3D}")
print("-" * 50)
print(f"The best fitting method is: {best_fit} with a determination of {best_value}")
print("=" * 50)

best_fit = "1D"
best_value = meanDuration1D
if meanDuration2D < best_value:
    best_fit = "2D"
    best_value = meanDuration2D
if meanDuration3D < best_value:
    best_fit = "3D"
    best_value = meanDuration3D

print(f"Mean duration 1D: {meanDuration1D} seconds")
print(f"Mean duration 2D: {meanDuration2D} seconds")
print(f"Mean duration 3D: {meanDuration3D} seconds")
print("-" * 50)
print(f"The best fitting method is: {best_fit} with a duration of {best_value} seconds")
print("=" * 50)
