import math
from scipy.integrate import quad
from scipy.special import j0

class KirchhoffDiffractionSimpson(object):
    TOL = 1e-1
    K = 0
    NA = 1.4
    lmbda = 610
    defocus = 1
    ni = 1.5

    def __init__(self, defocus, ni, accuracy, NA, lmbda):
        self.NA = NA
        self.lmbda = lmbda
        self.defocus = defocus
        self.ni = ni 
        if accuracy == 0:
            self.K = 5
        elif accuracy == 1:
            self.K = 7
        elif accuracy == 2:
            self.K = 9
        else:
            self.K = 3

    def calculate(self, r):
        # Séparation car quad ne gère pas les listes [real, imag]
        def real_integrand(rho):
            return self.integrand(rho, r)[0]
            
        def imag_integrand(rho):
            return self.integrand(rho, r)[1]

        # quad calcule l'intégrale de 0 à 1 très rapidement en C
        real_val, _ = quad(real_integrand, 0.0, 1.0, limit=500, epsabs=1e-4)
        imag_val, _ = quad(imag_integrand, 0.0, 1.0, limit=500, epsabs=1e-4)

        curI = real_val**2 + imag_val**2
        return curI

    
    def integrand(self, rho, r):
        k0 = 2.0 * math.pi / self.lmbda
        BesselValue = j0(k0 * self.NA * r * rho)
        OPD = 0.0
        W = 0.0
        I = [0.0,0.0]
        OPD = self.NA **2 * self.defocus * rho **2 / (2.0 * self.ni)
        W = k0 * OPD
        I[0] = BesselValue * math.cos(W) * rho
        I[1] = -BesselValue * math.sin(W) * rho
        return I
