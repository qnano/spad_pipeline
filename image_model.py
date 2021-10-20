# -*- coding: utf-8 -*-
"""
Binary image model

Contains the expressions for the binomial and Poissonian image model
Also contains first and second derivatives of the likelihood
Expressions of binomial image model are derived in the supplement of DOI: 10.1364/OE.439340
Expression of poissonian image model are derived in the supplement of DOI: 10.1038/nmeth.1449

Input: 
ROI: intensity matrix in ROI
sigma: emitter psf [sigma_x,sigma_y]
theta: emitter parameters: [loc_x,loc_y,intensity,background]
N: aggregated binary frame per image
te_SPAD: exposure time SPAD
    
Output:
binomial image model
binomial image model, first derivative
binomial image model, second derivative
binomial likelihood
binomial likelihood, first derivative
binomial likelihood, second derivative
binomial likelihood, Cramér-Rao lower bound (CRLB)

poissonian likelihood
poissonian likelihood, first derivative
poissonian likelihood, second derivative
poissonian likelihood, Cramér-Rao lower bound (CRLB)

"""

#mu and derivatives
import numpy as np
import scipy.special as sps
        


class pixel_values:
    def __init__(self,theta,sigma,x,y):

        self.theta = theta
        self.sigma = sigma
        self.x     = x
        self.y     = y

    def DEx(self):
        return 0.5*sps.erf((self.x-self.theta[0]+0.5)/(self.sigma[0]*np.sqrt(2)))-0.5*sps.erf((self.x-self.theta[0]-0.5)/(self.sigma[0]*np.sqrt(2)))
        
    def DEy(self):
        return 0.5*sps.erf((self.y-self.theta[1]+0.5)/(self.sigma[1]*np.sqrt(2)))-0.5*sps.erf((self.y-self.theta[1]-0.5)/(self.sigma[1]*np.sqrt(2)))
    
    # binomial image model
    def k(self):
        k  = self.theta[2]*self.DEx()*self.DEy()+self.theta[3]
        if k<1e-9:
            k=1e-9
        return k

    # binomial image model, first derivative
    def grad(self):
        expx1 = np.exp(-(self.x-self.theta[0]-0.5)**2/(2*self.sigma[0]**2))
        expx2 = np.exp(-(self.x-self.theta[0]+0.5)**2/(2*self.sigma[0]**2))
        expy1 = np.exp(-(self.y-self.theta[1]-0.5)**2/(2*self.sigma[1]**2))
        expy2 = np.exp(-(self.y-self.theta[1]+0.5)**2/(2*self.sigma[1]**2))
                
        dmu_dx = self.theta[2]/(np.sqrt(2*np.pi)*self.sigma[0])*(expx1-expx2)*self.DEy()
        dmu_dy = self.theta[2]/(np.sqrt(2*np.pi)*self.sigma[1])*(expy1-expy2)*self.DEx()
        dmu_dI = self.DEx()*self.DEy()
        dmu_db = 1
        mu_grad = np.array([[dmu_dx], [dmu_dy], [dmu_dI], [dmu_db]])
        return mu_grad
    
    # binomial image model, second derivative
    def hess(self):
        expx1 = np.exp(-(self.x-self.theta[0]-0.5)**2/(2*self.sigma[0]**2))
        expx2 = np.exp(-(self.x-self.theta[0]+0.5)**2/(2*self.sigma[0]**2))
        expy1 = np.exp(-(self.y-self.theta[1]-0.5)**2/(2*self.sigma[1]**2))
        expy2 = np.exp(-(self.y-self.theta[1]+0.5)**2/(2*self.sigma[1]**2))
        
        dmu_dx_dx = self.theta[2]/(np.sqrt(2*np.pi)*self.sigma[0]**3)*((self.x-self.theta[0]-0.5)*expx1-(self.x-self.theta[0]+0.5)*expx2)*self.DEy()
        dmu_dx_dy = self.theta[2]/(2*np.pi*self.sigma[0]*self.sigma[1])*(expx1-expx2)*(expy1-expy2)
        dmu_dx_dI = 1/(np.sqrt(2*np.pi)*self.sigma[0])*(expx1-expx2)*self.DEy()
        dmu_dx_db = 0
        
        dmu_dy_dx = self.theta[2]/(2*np.pi*self.sigma[0]*self.sigma[1])*(expx1-expx2)*(expy1-expy2)
        dmu_dy_dy = self.theta[2]/(np.sqrt(2*np.pi)*self.sigma[1]**3)*((self.y-self.theta[1]-0.5)*expy1-(self.y-self.theta[1]+0.5)*expy2)*self.DEx()
        dmu_dy_dI = 1/(np.sqrt(2*np.pi)*self.sigma[0])*(expy1-expy2)*self.DEx()
        dmu_dy_db = 0

        dmu_dI_dx = 1/(np.sqrt(2*np.pi)*self.sigma[0])*(expx1-expx2)*self.DEy()
        dmu_dI_dy = 1/(np.sqrt(2*np.pi)*self.sigma[1])*(expy1-expy2)*self.DEx()
        dmu_dI_dI = 0
        dmu_dI_db = 0
        
        dmu_db_dx = 0
        dmu_db_dy = 0
        dmu_db_dI = 0
        dmu_db_db = 0
        
        mu_hess = np.array([[dmu_dx_dx,dmu_dx_dy,dmu_dx_dI,dmu_dx_db],
                [dmu_dy_dx,dmu_dy_dy,dmu_dy_dI,dmu_dy_db],
                [dmu_dI_dx,dmu_dI_dy,dmu_dI_dI,dmu_dI_db],
                [dmu_db_dx,dmu_db_dy,dmu_db_dI,dmu_db_db]])
        return mu_hess
    
class likelihoods:
    def __init__(self,ROI,theta,sigma,N,te,dcr):
        self.ROI    = ROI
        self.theta  = theta
        self.sigma  = sigma
        self.N      = N
        self.te     = te
        self.dcr    = dcr
        
    # binomial likelihood
    def calc(self):
        LL = 0  
        for i in range(self.ROI.shape[0]):
            for j in range(self.ROI.shape[1]):
                mu = pixel_values(self.theta,self.sigma,i,j)  
                if self.ROI[i,j]==0:
                    self.ROI[i,j]=1e-9
                if self.N-self.ROI[i,j]==0:
                    self.ROI[i,j]=self.N-1e-9                
                # print(self.theta,i,j)
                LL += self.N*np.log(self.N)-self.ROI[i,j]*np.log(self.ROI[i,j])-(self.N-self.ROI[i,j])*np.log(self.N-self.ROI[i,j])
                # print('1',LL)
                LL += self.ROI[i,j]*np.log(1-np.exp(-self.te*(mu.k()+self.dcr[i,j])))
                # print('2',LL)
                LL -= self.te*(mu.k()+self.dcr[i,j])*(self.N-self.ROI[i,j])
                # print('3',LL)
        return LL
    
    # binomial likelihood, first derivative 
    def grad(self):
        LL_grad = np.zeros([4,1])
        for i in range(self.ROI.shape[0]):
            for j in range(self.ROI.shape[1]):
                mu = pixel_values(self.theta,self.sigma,i,j)
                E = 1-np.exp(-self.te*(mu.k()+self.dcr[i,j]))
                LL_grad += self.te*(self.ROI[i,j]-self.N*E)/E*mu.grad()
        return LL_grad
    
    # binomial likelihood, second derivative    
    def hess(self):
        LL_hess = np.zeros([4,4])
        for i in range(self.ROI.shape[0]):
            for j in range(self.ROI.shape[1]):
                mu = pixel_values(self.theta,self.sigma,i,j)
                frac1=(self.ROI[i,j]-self.N*(1-np.exp(-self.te*(mu.k()+self.dcr[i,j]))))/(1-np.exp(-self.te*(mu.k()+self.dcr[i,j])))
                frac2=(self.te*np.exp(-self.te*(mu.k()+self.dcr[i,j]))*self.ROI[i,j])/((1-np.exp(-self.te*(mu.k()+self.dcr[i,j])))**2)
                # for pm1 in range(4):
                #     for pm2 in range(4):
                LL_hess += self.te*(mu.hess()*frac1-mu.grad()*mu.grad().transpose()*frac2)
        return LL_hess

    
class results:
    def __init__(self,roishape,thetas,sigma,N,te,dcr):
        self.roishape   = roishape
        self.thetas     = thetas
        self.sigma      = sigma
        self.N          = N
        self.te         = te
        self.dcr        = dcr

    # binomial likelihood, Cramér-Rao lower bound (CRLB)      
    def CRLB(self):
        I = np.zeros([4,4])
        norm = self.N*self.te**2
        for x in range(self.roishape[0]):
            for y in range(self.roishape[1]):
                mu = pixel_values(self.thetas,self.sigma,x,y)
                If = np.exp(-self.te*(mu.k()+self.dcr[x,y]))/(1-np.exp(-self.te*(mu.k()+self.dcr[x,y])))
                I += norm*If*mu.grad()*mu.grad().transpose()
        return np.sqrt(np.linalg.inv(I).diagonal())[0], np.sqrt(np.linalg.inv(I).diagonal())[1]
    
    def poiss_CRLB(self):
        I = np.zeros([4,4])
        norm = self.N*self.te
        for x in range(self.roishape[0]):
            for y in range(self.roishape[1]):
                mu = pixel_values(self.thetas,self.sigma,x,y)
                If = 1/(mu.k()+self.dcr[x,y])
                I += norm*If*mu.grad()*mu.grad().transpose()
        return np.sqrt(np.linalg.inv(I).diagonal())[0], np.sqrt(np.linalg.inv(I).diagonal())[1]
 
    
class poiss_likelihoods:
    def __init__(self,ROI,theta,sigma,N,te,dcr):
        self.ROI    = ROI
        self.theta  = theta
        self.sigma  = sigma
        self.N      = N
        self.te     = te
        self.dcr    = dcr
    
    # poissonian likelihood
    def calc(self):
        LL = 0  
        for i in range(self.ROI.shape[0]):
            for j in range(self.ROI.shape[1]):
                mu = pixel_values(self.theta,self.sigma,i,j)
                if self.ROI[i,j] == 0:
                    self.ROI[i,j] = 1e-9
                LL += np.log(self.N*self.te*(mu.k()+self.dcr[i,j]))*self.ROI[i,j]
                LL -= self.N*self.te*(mu.k()+self.dcr[i,j])
                LL -= self.ROI[i,j]*np.log(self.ROI[i,j])-self.ROI[i,j]
        return LL

    # poissonian likelihood, first derivative    
    def grad(self):
        LL_grad = np.zeros([4,1])
        for i in range(self.ROI.shape[0]):
            for j in range(self.ROI.shape[1]):
                mu = pixel_values(self.theta,self.sigma,i,j)
                E = (self.ROI[i,j]/(self.N*self.te*(mu.k()+self.dcr[i,j]))-1)*self.N*self.te
                for pm in range(4):
                    LL_grad += E*mu.grad()
        return LL_grad
     

    # poissonian likelihood, second derivative         
    def hess(self):
        LL_hess = np.zeros([4,4])
        for i in range(self.ROI.shape[0]):
            for j in range(self.ROI.shape[1]):
                mu = pixel_values(self.theta,self.sigma,i,j)
                frac1=(self.ROI[i,j]/(self.N*self.te*(mu.k()+self.dcr[i,j]))-1)*self.N*self.te
                frac2=(self.ROI[i,j]/(self.N*self.te*(mu.k()+self.dcr[i,j])**2))*(self.N*self.te)
                LL_hess += mu.hess()*frac1-mu.grad()*mu.grad().transpose()*frac2
        return LL_hess


# poissonian likelihood, Cramér-Rao lower bound (CRLB)        
class poiss_results:
    def __init__(self,roishape,thetas,sigma,te,dcr):
        self.roishape   = roishape
        self.thetas     = thetas
        self.sigma      = sigma
        self.te         = te
        self.dcr        = dcr
        
    def CRLB(self):

        I = np.zeros([4,4])
        norm = self.te
        for x in range(self.roishape[0]):
            for y in range(self.roishape[1]):
                mu = pixel_values(self.thetas,self.sigma,x,y)
                If = 1/(mu.k()+self.dcr[x,y])
                I += norm*If*mu.grad()*mu.grad().transpose()

        return np.sqrt(np.linalg.inv(I).diagonal())[0], np.sqrt(np.linalg.inv(I).diagonal())[1]
    
    
