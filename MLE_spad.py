# -*- coding: utf-8 -*-
"""
Maximum likelihood estimation for SPAD

Performs maximum likelihood estimation using Levenberg Marquardt

Input: 
ROI: intensity matrix in ROI
sigma: emitter psf [sigma_x,sigma_y]
theta: emitter parameters: [loc_x,loc_y,intensity,background]
N: aggregated binary frame per image
te_SPAD: exposure time SPAD
    
Output:
Centre of mass
Optimized theta for binomial image model
Optimized theta for poissonian image model
Chi-square value and threshold for ROI
"""

#Levenberg-Marquardt optimization for SPAD

from image_model import pixel_values, likelihoods, poiss_likelihoods
import numpy as np

#Centre of mass calculation
def CoM(ROI):
    Mx = 0
    My = 0
    for y in range(ROI.shape[0]):
        for x in range(ROI.shape[1]):
            Mx += x*ROI[x,y]
            My += y*ROI[x,y]
    return Mx/np.sum(ROI), My/np.sum(ROI)

# Levenberg Marquardt optimization
def LM_SPAD_MLE(ROI,theta,sigma,N,te,dcr,stop=0.01,max_its=20):
    dll = stop+1
    it=0 #iteration
    ll = likelihoods(ROI,theta,sigma,N,te,dcr)
    LL_new = ll.calc()
    lamb = 0 #Relaxation parameter

    while it<max_its and abs(dll)>stop:
        H = ll.hess() - lamb*np.eye(4)
        try:
            H_ = np.linalg.inv(H)
            theta = theta-np.matmul(H_,ll.grad()).reshape(4)
            if theta[3] <0:
                theta[3] = 0
            if theta[2] <0:
                theta[2] = 0
            if theta[0] <2:
                theta[0] = 2
            if theta[0] >6:
                theta[0] = 6
            if theta[1] <2:
                theta[1] = 2
            if theta[1] >6:
                theta[1] = 6
                
            LL_old = LL_new
            ll = likelihoods(ROI,theta,sigma,N,te,dcr)
            LL_new = ll.calc()
            dll = LL_old - LL_new
        except: 
            LL_new = float('NaN')
            break

        it+=1

    return theta, LL_new, it-1

# Poissonian maximum likelihood estimation using Levenberg Marquardt
def poiss_LM_SPAD_MLE(ROI,theta,sigma,N,te,dcr,stop=0.01,max_its=20):
    dll = stop+1
    it=0
    ll = poiss_likelihoods(ROI,theta,sigma,N,te,dcr)
    LL_new = ll.calc()
    lamb = 0

    while it<max_its and abs(dll)>stop:
        H = ll.hess() - lamb*np.eye(4)
        try:
            H_ = np.linalg.inv(H)
            theta = theta-np.matmul(H_,ll.grad()).reshape(4)
            if theta[3] <0:
                theta[3] = 0
            if theta[2] <0:
                theta[2] = 0
                
            LL_old = LL_new
            ll = poiss_likelihoods(ROI,theta,sigma,N,te,dcr)
            LL_new = ll.calc()
            dll = LL_old - LL_new
        except: 
            LL_new = float('NaN')
            break

        it+=1

    return theta, LL_new

# Calculate chi-square value and threshold
# Derivation provided in supplement of DOI: 10.1364/OE.439340
def chi2filter(ROI,theta,sigma,N,te,dcr):
    chi2 = 0
    E_chi2 = 0
    avar_chi2 = 0
    var_chi2 = 0
    for i in range(ROI.shape[0]):
        for j in range(ROI.shape[1]):
            mu = pixel_values(theta,sigma,i,j)
            p = 1-np.exp(-te*(mu.k()+dcr[i,j]))
            chi2 += (ROI[i,j]-N*p)**2/(N*p)
            E_chi2 += 1-p
            var_chi2 += (1-p)*(2*N*p - 6*p - 2*N*p**2 + 6*p**2 + 1)/(N*p)
            
            avar_chi2 += 2+1/(N*p)

    return chi2, E_chi2 + 2*np.sqrt(var_chi2), E_chi2