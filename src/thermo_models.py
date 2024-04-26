
import sys
sys.path.insert(0, '../py_scripts')

import numpy as np
from scipy import optimize
import joblib

from numba import njit

@njit
def predict_nonpplatable(ST):
        
    return np.zeros_like(len(ST), np.float64)


@njit
def predict_substrate_only(ST, vbgp):

    SpT = ST*vbgp / (vbgp + 1)
        
    return SpT

@njit
def predict_push(WT, ST, vbgp, vWSp, alphaWS):
  
    Ncells = len(WT)
        
    poly_coeffs = np.zeros((Ncells, 3), np.float64)
    poly_coeffs[:, 0] = 1.0 # x^2
    poly_coeffs[:, 1] = 1.0 + (WT-ST)/alphaWS # x^1
    poly_coeffs[:, 2] = -ST/alphaWS # x^0
    
    
    Spf = np.zeros(Ncells, np.float64)
    for i in range(Ncells):
        
        roots = np.roots(poly_coeffs[i].astype(np.complex128))
        
        pos_roots = roots[np.real(roots) >= 0.0]
        if len(pos_roots) == 0:
            print(i, roots)
            print(roots[np.real(roots) >= 0.0])
        root = np.real(pos_roots[np.argmin(np.abs(np.imag(pos_roots)))])

        Spf[i] = root * alphaWS
        
    Wf = WT/(1+Spf/alphaWS)
        
    pWSu = Wf/alphaWS/(1+Wf/alphaWS)

    SpT = ST*(vWSp*pWSu + vbgp)/ (vWSp*pWSu + vbgp + 1)
        
        
    return SpT


@njit
def predict_pushpull(WT, ET, ST, vbgp, vWSp, alphaWS, vESu, alphaES):
      
    Ncells = len(WT)
        
    poly_coeffs = np.zeros((Ncells, 4), np.float64)
    poly_coeffs[:, 0] = 1.0 # x^3
    poly_coeffs[:, 1] = (alphaWS+alphaES + WT + ET - ST)/np.sqrt(alphaWS*alphaES) # x^2
    poly_coeffs[:, 2] = 1.0 + (WT-ST)/alphaWS + (ET-ST)/alphaES # x^1
    poly_coeffs[:, 3] = -ST/np.sqrt(alphaWS*alphaES) # x^0
        
    Spf = np.zeros(Ncells, np.float64)
    for i in range(Ncells):
        
        roots = np.roots(poly_coeffs[i].astype(np.complex128))
        
        pos_roots = roots[np.real(roots) >= 0.0]
        if len(pos_roots) == 0:
            print(i, roots)
            print(roots[np.real(roots) >= 0.0])
        root = np.real(pos_roots[np.argmin(np.abs(np.imag(pos_roots)))])
                                
        Spf[i] = root * np.sqrt(alphaWS*alphaES)
        
    Wf = WT/(1+Spf/alphaWS)
    Ef = ET/(1+Spf/alphaES)
        
    pWSu = Wf/alphaWS/(1+Wf/alphaWS+Ef/alphaES)
    pESp = Ef/alphaES/(1+Wf/alphaWS+Ef/alphaES)

    SpT = ST*(vWSp*pWSu + vbgp)/ (vWSp*pWSu + vESu*pESp + vbgp + 1)
        
        
    return SpT

def predict_twolayer(WT, ET, S1T, S2T, vpS1bg, vpS2bg, vpWS1, alphaWS1, vuES1, alphaES1, vpS1S2, alphaS1S2, n_jobs=1):
    
    Ncells = len(WT)
    
#     @njit
    def components(S1p, S1u, i):
        
        S1f = S1p + S1u

        a = S1p/np.sqrt(alphaWS1*alphaES1)
        b = 1 + S1p/alphaS1S2
        c1 = np.sqrt(alphaES1/alphaWS1) + S1f/np.sqrt(alphaWS1*alphaES1)
        c2 = np.sqrt(alphaWS1/alphaES1) + S1f/np.sqrt(alphaWS1*alphaES1)
        d = S2T[i] / alphaS1S2

        poly_coeffs = np.zeros(4, np.float64)
        poly_coeffs[0] = a**2 * b # x^3
        poly_coeffs[1] = a * (a * (WT[i] + ET[i] - S2T[i])/alphaS1S2 + b * (c1 + c2))# x^2
        poly_coeffs[2] = (a * WT[i]*ET[i]/np.sqrt(alphaWS1*alphaES1)/alphaS1S2 +b)*c1*c2 - a * d * (c1 + c2) # x^1
        poly_coeffs[3] =  -d * c1 * c2# x^0
        roots = np.roots(poly_coeffs.astype(np.complex128))
        
        pos_roots = roots[np.real(roots) >= 0.0]
        if len(pos_roots) == 0:
            print(i, roots)
            print(roots[np.real(roots) >= 0.0])
        
        root = np.real(pos_roots[np.argmin(np.abs(np.imag(pos_roots)))])
        
        S2f = root * alphaS1S2
        
        W = WT[i] / (1 + S1f/alphaWS1 + S1p*S2f/alphaWS1/alphaS1S2)
        E = ET[i] / (1 + S1f/alphaES1 + S1p*S2f/alphaES1/alphaS1S2)

        pWS1u = W/alphaWS1 / (1 + W/alphaWS1 + E/alphaES1)
        pES1p = E/alphaES1 / (1 + W/alphaWS1  + E/alphaES1)
        
        S1pT = S1T[i] * (vpWS1*pWS1u + vpS1bg) / (vpWS1*pWS1u + vuES1*pES1p + vpS1bg + 1)

        return (W, E, S2f, S1pT)
    
#     @njit    
    def costs(S1p, S1u, i):
        
        (W, E, S2f, S1pT) = components(S1p, S1u, i)
        S1uT = S1T[i] - S1pT

        f = np.zeros(2, np.float64)
        
        ZS1p = (1 + S2f/alphaS1S2) * (1 + W/alphaWS1 + E/alphaES1)
        ZS1u = (1 + W/alphaWS1 + E/alphaES1)

        if S1T[i] == 0:
            print(i, "a", S1p, S1u, S1T[i])
            
        if ZS1p == 0:
            print(ZS1p, "b")
        if ZS1u == 0:
            print(ZS1u, "c")
        
        f[0] = S1p/S1T[i] - S1pT/S1T[i]/ZS1p
        f[1] = S1u/S1T[i] - S1uT/S1T[i]/ZS1u
        
        return f
        
#     @njit   
    def pplation(S1p, S1u, i):
        
        (W, E, S2f, S1pT) = components(S1p, S1u, i)
                
        pS1pS2u = S1p/alphaS1S2 * (1 + W/alphaWS1 + E/alphaES1) / (1 + S1p/alphaS1S2 * (1 + W/alphaWS1 + E/alphaES1))

        S2pT = S2T[i] * (vpS1S2*pS1pS2u + vpS2bg) / (vpS1S2*pS1pS2u + vpS2bg + 1)
        
        return S1pT, S2pT
    
    def func(x, i):

        (fS1p, fS1u) = x.tolist()            
        
        S1p = fS1p * S1T[i]
        S1u = fS1u * S1T[i]
        return costs(S1p, S1u, i)
    

    def loop(i):

#         if i % 1000 == 0:
#             print(i, "/", Ncells)
            
        if S1T[i] == 0.0:
            S1p = 0.0
            S1u = 0.0
        else:
        
            x0 = np.zeros(2, np.float64)
            x0[0] = 1.0 # S1p/S1T
            x0[1] = 1.0 # S1u/S1T

            bounds = (0, 1)
            res = optimize.least_squares(func, x0=x0, args=(i,), bounds=bounds, jac='2-point', ftol=1e-8, xtol=1e-4, gtol=1e-8, verbose=0, 
                                            method='dogbox', max_nfev=10000)

            if not res.success:
                print(i)
                print(res)


            (fS1p, fS1u) = res.x.tolist()

            S1p = fS1p * S1T[i]
            S1u = fS1u * S1T[i]
        
        return pplation(S1p, S1u, i)
    
    res = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(loop)(i) for i in range(Ncells))
        
    S1pT_array, S2pT_array = zip(*res)
    S1pT_array = np.array(S1pT_array)
    S2pT_array = np.array(S2pT_array)
        
    return S1pT_array, S2pT_array
    
    
def predict_twolayer_nowriter(ET, S1T, S2T, vpS1bg, vpS2bg, vuES1, alphaES1, vpS1S2, alphaS1S2, n_jobs=1):
    
    Ncells = len(ET)
    
#     @njit     
    def components(S1p, S1u, i):
        
        S1f = S1p + S1u

        a = S1p/alphaES1
        b = 1 + S1p/alphaS1S2
        c = 1 + S1f/alphaES1
        d = S2T[i] / alphaS1S2

        poly_coeffs = np.zeros(3, np.float64)
        poly_coeffs[0] = a * b # x^2
        poly_coeffs[1] = a * (ET[i] - S2T[i])/alphaS1S2 + b * c# x^1
        poly_coeffs[2] = - d * c # x^0
        roots = np.roots(poly_coeffs.astype(np.complex128))
        pos_roots = roots[np.real(roots) >= 0.0]
        if len(pos_roots) == 0:
            print(i, roots)
            print(roots[np.real(roots) >= 0.0])
        root = np.real(pos_roots[np.argmin(np.abs(np.imag(pos_roots)))])
        
        S2f = root * alphaS1S2

        E = ET[i] / (1 + S1f/alphaES1 + S1p*S2f/alphaES1/alphaS1S2)

        pES1p = E/alphaES1 / (1 + E/alphaES1)
        
        S1pT = S1T[i] * vpS1bg / (vuES1*pES1p + vpS1bg + 1)

        return (E, S2f, S1pT)
       
#     @njit    
    def costs(S1p, S1u, i):
        
        (E, S2f, S1pT) = components(S1p, S1u, i)
        S1uT = S1T[i] - S1pT

        f = np.zeros(2, np.float64)
        
        ZS1p = (1 + S2f/alphaS1S2) * (1 + E/alphaES1)
        ZS1u = (1 + E/alphaES1)

#         f[0] = S1p/S1T[i] - S1pT/S1T[i]/ZS1p
#         f[1] = S1u/S1T[i] - S1uT/S1T[i]/ZS1u
        f[0] = S1p - S1pT/ZS1p
        f[1] = S1u - S1uT/ZS1u
        
        return f
        
#     @njit   
    def pplation(S1p, S1u, i):
        
        (E, S2f, S1pT) = components(S1p, S1u, i)
                
        pS1pS2u = S1p/alphaS1S2 * (1 + E/alphaES1) / (1 + S1p/alphaS1S2 * (1 + E/alphaES1))

        S2pT = S2T[i] * (vpS1S2*pS1pS2u + vpS2bg) / (vpS1S2*pS1pS2u + vpS2bg + 1)
        
        return S1pT, S2pT
    
    
    def func(x, i):

        (fS1p, fS1u) = x.tolist()

#         S1p = fS1p * S1T[i]
#         S1u = fS1u * S1T[i]
        S1p = fS1p
        S1u = fS1u
        return costs(S1p, S1u, i)
    

    def loop(i):

#         if i % 1000 == 0:
#             print(i, "/", Ncells)


        x0 = np.zeros(2, np.float64)
        x0[0] = 0.5*S1T[i] # S1p
        x0[1] = 0.5*S1T[i] # S1u

        bounds = (0, S1T[i])
        res = optimize.least_squares(func, x0=x0, args=(i,), bounds=bounds, jac='2-point', ftol=1e-8, xtol=1e-4, gtol=1e-8, verbose=0, 
                                        method='dogbox', max_nfev=1000)


        if not res.success:
            print(res)


        (fS1p, fS1u) = res.x.tolist()

        S1p = fS1p
        S1u = fS1u

        return pplation(S1p, S1u, i)

        
    res = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(loop)(i) for i in range(Ncells))
        
    S1pT_array, S2pT_array = zip(*res)
    S1pT_array = np.array(S1pT_array)
    S2pT_array = np.array(S2pT_array)
        
    return S1pT_array, S2pT_array

def predict_twolayer_noeraser(WT, S1T, S2T, vpS1bg, vpS2bg, vpWS1, alphaWS1, vpS1S2, alphaS1S2, n_jobs=1):
    
    Ncells = len(WT)
    
#     @njit 
    def components(S1p, S1u, i):
        
        S1f = S1p + S1u

        a = S1p/alphaWS1
        b = 1 + S1p/alphaS1S2
        c = 1 + S1f/alphaWS1
        d = S2T[i] / alphaS1S2

        poly_coeffs = np.zeros(3, np.float64)
        poly_coeffs[0] = a * b # x^2
        poly_coeffs[1] = a * (WT[i] - S2T[i])/alphaS1S2 + b * c# x^1
        poly_coeffs[2] = - d * c # x^0
        roots = np.roots(poly_coeffs.astype(np.complex128))
        pos_roots = roots[np.real(roots) >= 0.0]
        if len(pos_roots) == 0:
            print(i, roots)
            print(roots[np.real(roots) >= 0.0])
        root = np.real(pos_roots[np.argmin(np.abs(np.imag(pos_roots)))])
        
        S2f = root * alphaS1S2

        W = WT[i] / (1 + S1f/alphaWS1 + S1p*S2f/alphaWS1/alphaS1S2)

        pWS1u = W/alphaWS1 / (1 + W/alphaWS1)
        
        S1pT = S1T[i] * (vpWS1*pWS1u + vpS1bg) / (vpWS1*pWS1u + vpS1bg + 1)

        return (W, S2f, S1pT)
    
#     @njit    
    def costs(S1p, S1u, i):
        
        (W, S2f, S1pT) = components(S1p, S1u, i)
        S1uT = S1T[i] - S1pT

        f = np.zeros(2, np.float64)
        
        ZS1p = (1 + S2f/alphaS1S2) * (1 + W/alphaWS1)
        ZS1u = (1 + W/alphaWS1)

        f[0] = S1p/S1T[i] - S1pT/S1T[i]/ZS1p
        f[1] = S1u/S1T[i] - S1uT/S1T[i]/ZS1u
        
        return f
        
#     @njit   
    def pplation(S1p, S1u, i):
        
        (W, S2f, S1pT) = components(S1p, S1u, i)
                
        pS1pS2u = S1p/alphaS1S2 * (1 + W/alphaWS1) / (1 + S1p/alphaS1S2 * (1 + W/alphaWS1))

        S2pT = S2T[i] * (vpS1S2*pS1pS2u + vpS2bg) / (vpS1S2*pS1pS2u + vpS2bg + 1)
        
        return S1pT, S2pT
        
    
    def func(x, i):

        (fS1p, fS1u) = x.tolist()

        S1p = fS1p * S1T[i]
        S1u = fS1u * S1T[i]
        return costs(S1p, S1u, i)
    

    def loop(i):

#         if i % 1000 == 0:
#             print(i, "/", Ncells)
            
        
        
        if S1T[i] == 0.0:
            S1p = 0.0
            S1u = 0.0
        else:
        
            x0 = np.zeros(2, np.float64)
            x0[0] = 1.0 # S1p/S1T
            x0[1] = 1.0 # S1u/S1T

            bounds = (0, 1)
            res = optimize.least_squares(func, x0=x0, args=(i,), bounds=bounds, jac='2-point', ftol=1e-8, xtol=1e-4, gtol=1e-8, verbose=0, 
                                            method='dogbox', max_nfev=1000)


            if not res.success:
                print(res)


            (fS1p, fS1u) = res.x.tolist()

            S1p = fS1p * S1T[i]
            S1u = fS1u * S1T[i]
        
        return pplation(S1p, S1u, i)
    
    res = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(loop)(i) for i in range(Ncells))
        
    S1pT_array, S2pT_array = zip(*res)
    S1pT_array = np.array(S1pT_array)
    S2pT_array = np.array(S2pT_array)
        
    return S1pT_array, S2pT_array
