from microscopy_metrics.fitting import FittingTool,Fitting3D,Fitting1D,Fitting2D
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from pathlib import Path
from tqdm import tqdm


PSF_SIZE = 50

TRUE_AMP = 1.0
TRUE_BG = 0.0
TRUE_MU_X = PSF_SIZE / 2
TRUE_MU_Y = PSF_SIZE / 2
TRUE_MU_Z = PSF_SIZE / 2
TRUE_SIGMA_X = PSF_SIZE / 10
TRUE_SIGMA_Y = PSF_SIZE / 10
TRUE_SIGMA_Z = PSF_SIZE / 10

def show3DPsf(X,Y,Z,psf_3d):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    alphas = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)


    ax.scatter(X, Y, Z, c=psf_3d, cmap='viridis', s=1, alpha=alphas)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("PSF 3D (Gaussienne)")
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(psf_3d)
    plt.colorbar(mappable, ax=ax, label='Intensité')
    plt.show()

def show2DPsf(psf,center):
    plt.imshow(psf[center], cmap='viridis')
    plt.colorbar()
    plt.title("2D Image of a Gaussian")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def computeMape(estimations, truths):
    error = 0.0
    for est, truth in zip(estimations, truths):
        error += abs(est - truth) / truth
    return (error / len(truths)) * 100

def generateRandomPSFParams(psf_size=10):
    mu_x = np.random.uniform(psf_size * 0.4, psf_size * 0.6)
    mu_y = np.random.uniform(psf_size * 0.4, psf_size * 0.6)
    mu_z = np.random.uniform(psf_size * 0.4, psf_size * 0.6)
    
    sigma_x = np.random.uniform(5.0, 10.0)
    sigma_y = np.random.uniform(5.0, 15.0)
    sigma_z = np.random.uniform(8.0, 20.0) 
    
    amp = np.random.uniform(0.5, 2.0)
    bg = np.random.uniform(0.0, 0.2)
    
    return [amp, bg, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z]

def evaluatePsf():
    fitTool = Fitting3D()
    params = [TRUE_AMP, TRUE_BG, TRUE_MU_X, TRUE_MU_Y, TRUE_MU_Z, TRUE_SIGMA_X, TRUE_SIGMA_Y, TRUE_SIGMA_Z]    
    #params = generateRandomPSFParams(PSF_SIZE)
    zz = np.arange(PSF_SIZE)
    yy = np.arange(PSF_SIZE)
    xx = np.arange(PSF_SIZE)
    x, y, z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    psf = fitTool.gauss(*params)(coords)
    psfReshape = psf.reshape((PSF_SIZE,PSF_SIZE,PSF_SIZE))
    #show2DPsf(psfReshape,int(PSF_SIZE/2))
    FWHM = [fitTool.fwhm(TRUE_SIGMA_X), fitTool.fwhm(TRUE_SIGMA_Y), fitTool.fwhm(TRUE_SIGMA_Z)]
    fitTool1D = Fitting1D()
    fitTool1D._image = psfReshape
    fitTool1D._centroid = [int(PSF_SIZE/2),int(PSF_SIZE/2),int(PSF_SIZE/2)]
    fitTool1D._roi = [np.array([0,0,0])]
    fitTool1D._outputDir = Path(__file__).parent / "EvalFitResult" / "1D"
    result = fitTool1D.processSingleFit(0)
    corr1D = computeMape(result[1], FWHM)
    fitTool2D = Fitting2D()
    fitTool2D._image = psfReshape
    fitTool2D._centroid = [int(PSF_SIZE/2),int(PSF_SIZE/2),int(PSF_SIZE/2)]
    fitTool2D._roi = [np.array([0,0,0])]
    fitTool2D._outputDir = Path(__file__).parent / "EvalFitResult" / "2D"
    result = fitTool2D.processSingleFit(0)
    corr2D = computeMape(result[1], FWHM)
    fitTool3D = Fitting3D()
    fitTool3D._image = psfReshape
    fitTool3D._centroid = [int(PSF_SIZE/2),int(PSF_SIZE/2),int(PSF_SIZE/2)]
    fitTool3D._roi = [np.array([0,0,0])]
    fitTool3D._outputDir = Path(__file__).parent / "EvalFitResult" / "3D"
    result = fitTool3D.processSingleFit(0)
    corr3D = computeMape(result[1], FWHM)
    return corr1D,corr2D,corr3D



corr1D,corr2D,corr3D = 0,0,0
meanCorr1D, meanCorr2D, meanCorr3D = 0,0,0
n_tests = 10
pbar = tqdm(total=n_tests, desc="Evaluating PSF Fitting", unit="test")
for i in range(10):
    corr1D,corr2D,corr3D = evaluatePsf()
    meanCorr1D += corr1D
    meanCorr2D += corr2D
    meanCorr3D += corr3D
    pbar.update(1)
    pbar.set_postfix({"MAPE 1D": f"{corr1D:.2f}%", "MAPE 2D": f"{corr2D:.2f}%", "MAPE 3D": f"{corr3D:.2f}%"})
print(f"Final Results: MAPE 1D = {meanCorr1D/10:.2f}%, MAPE 2D = {meanCorr2D/10:.2f}%, MAPE 3D = {meanCorr3D/10:.2f}%")