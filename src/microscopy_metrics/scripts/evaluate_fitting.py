from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.fittingTools.fitting1D import Fitting1D
from microscopy_metrics.fittingTools.fitting2D import Fitting2D
from microscopy_metrics.fittingTools.fitting2DEllips import Fitting2DEllips
from microscopy_metrics.fittingTools.fitting2DRotate import Fitting2DRotation
from microscopy_metrics.fittingTools.fitting3D import Fitting3D
from microscopy_metrics.fittingTools.prominence import Prominence
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

PSF_SIZE = 80
NOISE = True
FIT_METHODS = ["1D","2D","3D","Prominence","2DEllips","2DRotation"]

TRUE_AMP = 255.0
TRUE_BG = 0.0
TRUE_MU_X = PSF_SIZE / 2
TRUE_MU_Y = PSF_SIZE / 2
TRUE_MU_Z = PSF_SIZE / 2
TRUE_SIGMA_X = PSF_SIZE / 10
TRUE_SIGMA_Y = PSF_SIZE / 10
TRUE_SIGMA_Z = PSF_SIZE / 10


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


def computePSNR(estimations, truths):
    mse = computeMSE(estimations, truths)
    if mse == 0:
        return float("inf")
    psnr = 10 * np.log10((TRUE_AMP**2) / mse)
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
    amp = TRUE_AMP
    bg = 0.0
    return [amp, bg, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z]


def addMicroscopyNoise(image, maxPhotons=1000, readoutNoiseStd=5.0):
    scaledImage = (image / np.max(image)) * maxPhotons
    noisyPoisson = np.random.poisson(scaledImage).astype(np.float32)
    gaussianNoise = np.random.normal(0, readoutNoiseStd, size=image.shape)
    finalNoisyImage = noisyPoisson + gaussianNoise
    finalNoisyImage = np.minimum(np.maximum(finalNoisyImage, 0),maxPhotons)
    return finalNoisyImage

def generatePSF(params):
    fitTool = Fitting3D()
    fitTool._show = False
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    psf = fitTool.gauss(*params)(coords)
    return psf,coords

def fitPSF(fitName, image):
    fitTool = FittingTool.getInstance(fitName)
    fitTool._image = image
    fitTool._show = False
    fitTool._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool._roi = [np.array([0, 0, 0])]
    fitTool._outputDir = Path(__file__).parent / "EvalFitResult" / fitName
    start = time.time()
    result = fitTool.processSingleFit(0)
    end = time.time()
    elapsed = end - start
    return result,elapsed

def getBhattacharyyaFit(params, mu, sigma):
    DistBat = 0.0
    DistBat += computeBhattacharyyaDistance(params[2],mu[0],params[5],sigma[0])
    DistBat += computeBhattacharyyaDistance(params[3],mu[1],params[6],sigma[1])
    DistBat += computeBhattacharyyaDistance(params[4],mu[2],params[7],sigma[2])
    DistBat /= 3.0
    return DistBat

def evaluate1DPsf(psf,psfReshape,FWHM,coords,params):
    result,elapsed1D = fitPSF("1D", psfReshape)
    corr1D = computeMape(result[1], FWHM)
    amp = (result[4][0][0] + result[4][1][0] + result[4][2][0]) / 3.0
    bg = (result[4][0][1] + result[4][1][1] + result[4][2][1]) / 3.0
    mu = [result[4][0][2], result[4][1][2], result[4][2][2]]
    sigma = [result[4][0][3], result[4][1][3], result[4][2][3]]
    params1D = [amp, bg, *mu, *sigma]
    fit = Fitting3D().gauss(*params1D)(coords)
    psnr1D = computePSNR(fit, psf)
    DistBat1D = getBhattacharyyaFit(params, mu, sigma)
    determination1D = (result[3][0] + result[3][1] + result[3][2]) / 3.0
    return corr1D,psnr1D,DistBat1D,determination1D,elapsed1D

def evaluateXDPsf(psf,psfReshape,FWHM,coords,params,fitMethod = "2D"):
    result, elapsed = fitPSF(fitMethod, psfReshape)
    corr = computeMape(result[1], FWHM)
    fit = Fitting3D().gauss(*result[4])(coords)
    psnr = computePSNR(fit, psf)
    mu = [result[4][2], result[4][3], result[4][4]]
    sigma = [result[4][5], result[4][6], result[4][7]]
    DistBat = getBhattacharyyaFit(params, mu, sigma)
    determination = (result[3][0] + result[3][1] + result[3][2]) / 3.0
    return corr,psnr,DistBat,determination,elapsed


def evaluatePsf():
    params = generateRandomPSFParams(PSF_SIZE)
    psf,coords = generatePSF(params)
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    if NOISE :
        psfReshape = addMicroscopyNoise(psfReshape,maxPhotons=TRUE_AMP)
    FWHM = [FittingTool().fwhm(params[5]), FittingTool().fwhm(params[6]), FittingTool().fwhm(params[7])]
    
    corr1D,psnr1D,DistBat1D,determination1D,elapsed1D = evaluate1DPsf(psf,psfReshape,FWHM,coords,params)
    corr2D,psnr2D,DistBat2D,determination2D,elapsed2D = evaluateXDPsf(psf,psfReshape,FWHM,coords,params,"2D")
    corr3D,psnr3D,DistBat3D,determination3D,elapsed3D = evaluateXDPsf(psf,psfReshape,FWHM,coords,params,"3D")

    result, elapsedProminence = fitPSF("Prominence", psfReshape)
    corrProminence = computeMape(result[1],FWHM)

    result, elapsed2DEllips = fitPSF("2D Ellipse",psfReshape)
    corr2DEllips = computeMape(result[1],FWHM)

    result, elapsed2DRotation = fitPSF("2D rotation", psfReshape)
    corr2DRotation = computeMape(result[1],FWHM)
    return (
        [corr1D,corr2D,corr3D,corrProminence,corr2DEllips,corr2DRotation],
        [psnr1D,psnr2D,psnr3D],
        [DistBat1D,DistBat2D,DistBat3D],
        [determination1D,determination2D,determination3D],
        [elapsed1D,elapsed2D,elapsed3D,elapsedProminence,elapsed2DEllips,elapsed2DRotation],
    )

def addTab(tabA, tabB) :
    minRange = min(len(tabA), len(tabB))
    tabC = []
    for i in range(minRange):
        tabC.append(tabA[i] + tabB[i])
    return tabC

def divTab(tab, div):
    return [x / div for x in tab]

def superior(a,b):
    return a > b

def inferior(a,b):
    return a < b

def printResults(metricStr,metric,unitStr,comparison=superior):
    print("=" * 20,metricStr.upper(), "="*20)
    best_fit = FIT_METHODS[0]
    best_value = metric[0]
    print(f"Mean {metricStr} {FIT_METHODS[0]}: {metric[0]:.8f}{unitStr}")
    for i in range(1,len(metric)):
        if comparison(metric[i],best_value) :
            best_fit = FIT_METHODS[i]
            best_value = metric[i]
        elif metric[i] == best_value:
            best_fit += f" and {FIT_METHODS[i]}"
        print(f"Mean {metricStr} {FIT_METHODS[i]}: {metric[i]:.8f}{unitStr}")
    print("-" * 50)
    print(f"The best fitting method is: {best_fit} | with a {metricStr} of {best_value:.8f}{unitStr}", "\n")
    
        
if __name__ == "__main__":
    params = generateRandomPSFParams(PSF_SIZE)
    psf,coords = generatePSF(params)
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    if NOISE : 
        psfReshape = addMicroscopyNoise(psfReshape)
    show2DPsf(psfReshape, int(PSF_SIZE / 2))

    meanCorr = [0 for _ in range(len(FIT_METHODS))]
    meanPSNR = [0, 0, 0]
    meanBat = [0, 0, 0]
    meanDetermination = [0, 0, 0]
    meanDuration = [0 for _ in range(len(FIT_METHODS))]

    n_tests = 100

    pbar = tqdm(total=n_tests, desc="Evaluating PSF Fitting", unit="test")
    workers = int(os.cpu_count() * 0.75)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluatePsf) for _ in range(n_tests)}
        for future in as_completed(futures):
            (corr,psnr,bat,determination,duration) = future.result()
            meanCorr = addTab(meanCorr, corr)
            meanPSNR = addTab(meanPSNR,psnr)
            meanBat = addTab(meanBat,bat)
            meanDetermination = addTab(meanDetermination, determination)
            meanDuration = addTab(meanDuration, duration)
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
    meanCorr = divTab(meanCorr,n_tests)
    meanPSNR = divTab(meanPSNR,n_tests)
    meanBat = divTab(meanBat,n_tests)
    meanDetermination = divTab(meanDetermination,n_tests)
    meanDuration = divTab(meanDuration,n_tests)

    print("\n\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50,"\n")

    printResults("MAPE",meanCorr,"%",inferior)
    printResults("PSNR",meanPSNR,"dB",superior)
    printResults("distance",meanBat,"",inferior)
    printResults("R2 score", meanDetermination,"",superior)
    printResults("duration",meanDuration,"seconds",inferior)

    print("=" * 50)
    print("END SUMMARY")
    print("=" * 50,"\n")
