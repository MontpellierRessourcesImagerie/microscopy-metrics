import os
import time 
import random
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from microscopy_metrics.utils import pxToUm
from microscopy_metrics.fittingTools.fitting3D import Fitting3D
from microscopy_metrics.scripts.PSFGenerator.Data3D import Data3D
from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.scripts.PSFGenerator.BornWolfPSF import BornWolfPSF
from microscopy_metrics.fittingTools.fitting3DRotation import Fitting3DRotation

BORNOWOLF_PSF = False
ORIENTED = False
PSF_SIZE = 80
NOISE = False
FIT_METHODS = ["1D","2D","3D","2DRotation","3DRotation","2DEllips","Prominence"]

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


def get_rotation_matrix(thetaX, thetaY, thetaZ):
    """Calcule la matrice de rotation 3D à partir des angles d'Euler"""
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

def computeAngularError(estimations, truths, fitMethod="2D"):
    if fitMethod == "3D Rotation":
        R_est = get_rotation_matrix(estimations[0], estimations[1], estimations[2])
        R_true = get_rotation_matrix(truths[0], truths[1], truths[2])
        dot_x = abs(np.dot(R_est[:, 0], R_true[:, 0]))
        dot_y = abs(np.dot(R_est[:, 1], R_true[:, 1]))
        dot_z = abs(np.dot(R_est[:, 2], R_true[:, 2]))        
        err_x = np.rad2deg(np.arccos(min(dot_x, 1.0)))
        err_y = np.rad2deg(np.arccos(min(dot_y, 1.0)))
        err_z = np.rad2deg(np.arccos(min(dot_z, 1.0)))        
        return np.mean([err_x, err_y, err_z])
    errors = []
    for est, truth in zip(estimations, truths):
        truth_deg = np.rad2deg(truth) if abs(truth) <= 2 * np.pi else truth
        diff = abs(est - truth_deg)
        diff = diff % 180.0
        error = min(diff, 180.0 - diff)
        errors.append(error)
    return np.mean(errors) if errors else 0.0


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
    sigma_x = np.random.uniform(0.95 * psf_size/7.0, 1.05 * sigmaDefault)
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
    fwhm = [fitTool.fwhm(params[5]), fitTool.fwhm(params[6]), fitTool.fwhm(params[7])]
    return psf,coords,fwhm

def generateBornWolfPSF():
    bw = BornWolfPSF()
    bw.nx = PSF_SIZE
    bw.ny = PSF_SIZE
    bw.nz = PSF_SIZE
    bw.resLateral = 50.0
    bw.resAxial = 25.0
    bw.NA = 1.4
    bw.lmbda = 500.0 * 1e-9
    bw.setParameters(ni=1.5, accuracy=0)
    bw.data = Data3D(bw.nx, bw.ny, bw.nz)
    
    bw.generate()
    bw.data.determineMaximumAndEnergy()
    bw.data.estimateFWHM()
    
    fwhm = [bw.data.fwhm.x, bw.data.fwhm.y, bw.data.fwhm.z]
    
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    
    psf = bw.data.data.flatten()
    for i in range(3):
        fwhm[i] = pxToUm(fwhm[i], bw.resLateral / 1000.0 if i < 2 else bw.resAxial / 1000.0)

    return psf, coords, fwhm

def generateRandomBornoWolfPSF(seed=None):
    if seed is not None:
        np.random.seed(seed)
    bw = BornWolfPSF()
    bw.nx = PSF_SIZE
    bw.ny = PSF_SIZE
    bw.nz = PSF_SIZE
    bw.resLateral = 50.0
    bw.resAxial = 50.0
    bw.NA = 1.4
    bw.lmbda = np.random.uniform(400.0, 700.0) * 1e-9
    ni = np.random.uniform(1.4, 1.6)
    accuracy = np.random.randint(0, 3)
    bw.setParameters(ni=ni, accuracy=accuracy)
    bw.data = Data3D(bw.nx, bw.ny, bw.nz)
    
    bw.generate()
    bw.data.determineMaximumAndEnergy()
    bw.data.estimateFWHM()
    
    fwhm = [bw.data.fwhm.x, bw.data.fwhm.y, bw.data.fwhm.z]
    fwhm.reverse()
    
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    
    psf = bw.data.data.flatten()
    
    return psf, coords, fwhm

def generateOrientedPSF(params,seed=None):
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
    psf = fitTool.gauss(*params,*thetas)(coords)
    fwhm = [fitTool.fwhm(params[7]), fitTool.fwhm(params[6]), fitTool.fwhm(params[5])]
    return psf, coords, fwhm, thetas

def fitPSF(fitName, image):
    fitTool = FittingTool.getInstance(fitName)
    fitTool._image = image
    fitTool._show = False
    fitTool._centroid = [int(PSF_SIZE / 2), int(PSF_SIZE / 2), int(PSF_SIZE / 2)]
    fitTool._roi = [np.array([0, 0, 0])]
    fitTool._pixelSize = [0.1,0.1,0.1]
    fitTool._outputDir = Path(__file__).parent / "EvalFitResult" / fitName
    start = time.time()
    fitTool.processSingleFit(0)
    end = time.time()
    elapsed = end - start
    result = [0, fitTool.fwhms, fitTool.uncertainties, fitTool.determinations, fitTool.parameters, fitTool.pcovs, fitTool.thetas]
    return result,elapsed

def getBhattacharyyaFit(params, mu, sigma):
    DistBat = 0.0
    DistBat += computeBhattacharyyaDistance(params[2],mu[0],params[5],sigma[0])
    DistBat += computeBhattacharyyaDistance(params[3],mu[1],params[6],sigma[1])
    DistBat += computeBhattacharyyaDistance(params[4],mu[2],params[7],sigma[2])
    DistBat /= 3.0
    return DistBat

def evaluateXDPsf(psf,psfReshape,FWHM,coords,params = None,fitMethod = "2D",thetas = None):
    result, elapsed = fitPSF(fitMethod, psfReshape)
    corr = computeMape(result[1], FWHM)
    fit = Fitting3D().gauss(*result[4])(coords)
    psnr = computePSNR(fit, psf)
    mu = [result[4][2], result[4][3], result[4][4]]
    sigma = [result[4][5], result[4][6], result[4][7]]
    if params is not None:
        DistBat = getBhattacharyyaFit(params, mu, sigma)
    else:
        DistBat = 0
    determination = (result[3][0] + result[3][1] + result[3][2]) / 3.0
    if thetas is not None :
        corrOrientation = computeAngularError(result[6], thetas, fitMethod)
    else : 
        corrOrientation = np.inf
    return corr,psnr,DistBat,determination,elapsed,corrOrientation


def evaluatePsf(seed=None):
    thetas = None
    params = generateRandomPSFParams(PSF_SIZE)
    if BORNOWOLF_PSF and not ORIENTED:
        psf, coords, FWHM = generateRandomBornoWolfPSF(seed=seed)
        params = None
    elif ORIENTED:
        psf, coords, FWHM, thetas = generateOrientedPSF(params,seed=seed)
        params = None
    else:
        psf,coords,FWHM = generatePSF(params)
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    if NOISE :
        psfReshape = addMicroscopyNoise(psfReshape,maxPhotons=TRUE_AMP)
    corr1D,psnr1D,DistBat1D,determination1D,elapsed1D,corrOrientation1D = evaluateXDPsf(psf,psfReshape,FWHM,coords,params,"1D")
    corr2D,psnr2D,DistBat2D,determination2D,elapsed2D,corrOrientation2D = evaluateXDPsf(psf,psfReshape,FWHM,coords,params,"2D")
    corr3D,psnr3D,DistBat3D,determination3D,elapsed3D,corrOrientation3D = evaluateXDPsf(psf,psfReshape,FWHM,coords,params,"3D")

    result, elapsedProminence = fitPSF("Prominence", psfReshape)
    corrProminence = computeMape(result[1],FWHM)
    corrOrientationProminence = np.inf

    corr2DEllips,psnr2DEllips,DistBat2DEllips,determination2DEllips,elapsed2DEllips,corrOrientation2DEllips = evaluateXDPsf(psf,psfReshape,FWHM,coords,params,"2D Ellipse",thetas)
    corr2DRotation,psnr2DRotation,DistBat2DRotation,determination2DRotation,elapsed2DRotation,corrOrientation2DRotation = evaluateXDPsf(psf,psfReshape,FWHM,coords,params,"2D rotation",thetas)
    corr3DRotation,psnr3DRotation,DistBat3DRotation,determination3DRotation,elapsed3DRotation,corrOrientation3DRotation = evaluateXDPsf(psf,psfReshape,FWHM,coords,params,"3D Rotation",thetas)
    return (
        [corr1D,corr2D,corr3D,corr2DRotation,corr3DRotation,corr2DEllips,corrProminence],
        [psnr1D,psnr2D,psnr3D,psnr2DRotation,psnr3DRotation,psnr2DEllips],
        [DistBat1D,DistBat2D,DistBat3D,DistBat2DRotation,DistBat3DRotation,DistBat2DEllips],
        [determination1D,determination2D,determination3D,determination2DRotation,determination3DRotation,determination2DEllips],
        [elapsed1D,elapsed2D,elapsed3D,elapsed2DRotation,elapsed3DRotation,elapsed2DEllips,elapsedProminence],
        [corrOrientation1D,corrOrientation2D,corrOrientation3D,corrOrientation2DRotation,corrOrientation3DRotation,corrOrientation2DEllips,corrOrientationProminence]
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
    print(f"Mean {metricStr} {FIT_METHODS[0]}: {metric[0]:.8f} {unitStr}")
    for i in range(1,len(metric)):
        if comparison(metric[i],best_value) :
            best_fit = FIT_METHODS[i]
            best_value = metric[i]
        elif metric[i] == best_value:
            best_fit += f" and {FIT_METHODS[i]}"
        print(f"Mean {metricStr} {FIT_METHODS[i]}: {metric[i]:.8f} {unitStr}")
    print("-" * 50)
    print(f"The best fitting method is: {best_fit} | with a {metricStr} of {best_value:.8f} {unitStr}", "\n")
    
        
if __name__ == "__main__":
    params = generateRandomPSFParams(PSF_SIZE)
    if BORNOWOLF_PSF and not ORIENTED:
        psf, coords, FWHM = generateBornWolfPSF()
    elif ORIENTED:
        psf, coords, FWHM, _ = generateOrientedPSF(params)
    else:
        psf,coords,FWHM = generatePSF(params)
    psfReshape = psf.reshape((PSF_SIZE, PSF_SIZE, PSF_SIZE))
    if NOISE : 
        psfReshape = addMicroscopyNoise(psfReshape)
    show2DPsf(psfReshape, int(PSF_SIZE / 2))

    meanCorr = [0 for _ in range(len(FIT_METHODS))]
    meanPSNR = [0, 0, 0, 0, 0, 0, 0]
    meanBat = [0, 0, 0, 0, 0, 0, 0]
    meanDetermination = [0, 0, 0, 0, 0, 0, 0]
    meanDuration = [0 for _ in range(len(FIT_METHODS))]
    meanCorrOrientation = [0.0 for _ in range(len(FIT_METHODS))]

    n_tests = 30

    pbar = tqdm(total=n_tests, desc="Evaluating PSF Fitting", unit="test")
    workers = int(os.cpu_count() * 0.75)

    randomSeeds = np.random.randint(0, 2**32, size=n_tests)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(evaluatePsf, seed=seed) for seed in randomSeeds}
        for future in as_completed(futures):
            (corr,psnr,bat,determination,duration,corrOrientation) = future.result()
            meanCorr = addTab(meanCorr, corr)
            meanPSNR = addTab(meanPSNR,psnr)
            meanBat = addTab(meanBat,bat)
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
    meanCorr = divTab(meanCorr,n_tests)
    meanPSNR = divTab(meanPSNR,n_tests)
    meanBat = divTab(meanBat,n_tests)
    meanDetermination = divTab(meanDetermination,n_tests)
    meanDuration = divTab(meanDuration,n_tests)
    meanCorrOrientation = divTab(meanCorrOrientation,n_tests)

    print("\n\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50,"\n")

    printResults("MAPE",meanCorr,"%",inferior)
    printResults("PSNR",meanPSNR,"dB",superior)
    printResults("distance",meanBat,"",inferior)
    printResults("R2 score", meanDetermination,"",superior)
    printResults("duration",meanDuration,"seconds",inferior)
    printResults("orientation",meanCorrOrientation,"°",inferior)

    print("=" * 50)
    print("END SUMMARY")
    print("=" * 50,"\n")
