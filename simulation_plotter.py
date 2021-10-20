# -*- coding: utf-8 -*-
"""
Simulation plotter

Takes emitter estimates from simulation and clusters them into emitter locations
Then plots achieved and theorized uncertainty as well as histograms of emitter estimates

Input:
.hdf5 files of emitter estimates from "binary_simulation.py"
.npy files of emitter data from "binary_simulation.py"
Output:
Plots of theorized and achieved uncertainty
Plot of fitted Gaussian distribution to emitter estimates
"""


import scipy.stats as sps
import h5py
import numpy as np
import spadtools as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
text_kwargs = dict(ha='center', va='center', fontsize=10)
plt.rcParams["font.family"] = "Arial"

blue = (57/255, 106/255, 177/255) 
yellow = (218/255,124/255,48/255) 
green = (62/255,150/255,81/255)
red = (204/255,37/255,41/255)


#%% plot simulated uncertainties

data_simulation = np.load('data_simulation\\data_simulation.npy')

stdlst              = data_simulation[0,:]
CRLBlst             = data_simulation[1,:]
poiss_CRLBlst       = data_simulation[2,:]
CRLB_theory         = data_simulation[3,:]
poiss_CRLB_theory   = data_simulation[4,:]

data_simulation = np.load('data_simulation\\data_simulation_bg100.npy')

stdlst2              = data_simulation[0,:]
CRLBlst2             = data_simulation[1,:]
poiss_CRLBlst2       = data_simulation[2,:]
CRLB_theory2         = data_simulation[3,:]
poiss_CRLB_theory2   = data_simulation[4,:]

data_simulation = np.load('data_simulation\\data_simulation_constantbg.npy')

stdlst3              = data_simulation[0,:]
CRLBlst3             = data_simulation[1,:]
poiss_CRLBlst3       = data_simulation[2,:]
CRLB_theory3         = data_simulation[3,:]
poiss_CRLB_theory3   = data_simulation[4,:]

setsize=25


multiplier = 1.26

Is = [35000]
for k in range(setsize-1):
    Is.append(Is[-1]*multiplier)

te_SPAD   = 10e-6 + 5e-9
plt.close('uncertainty')
plt.figure('uncertainty',figsize=(6,4))
plt.grid(which='major',linewidth=.8,linestyle='-',zorder=1)
plt.grid(which='minor',linewidth=.8,linestyle='--',zorder=1)
plt.scatter(np.array(Is)*te_SPAD*255,stdlst[:setsize]*100,s=8,marker='x',label='$\sigma_x$, N=255',color=yellow)
plt.scatter(np.array(Is)*te_SPAD*255,stdlst[setsize:2*setsize]*100,s=8,marker='o',label='$\sigma_x$, N=510',color=blue)
plt.scatter(np.array(Is)*te_SPAD*255,stdlst[2*setsize:3*setsize]*100,s=8,marker='^',label='$\sigma_x$, N=1275',color=green)
plt.plot(np.array(Is)*te_SPAD*255,CRLBlst[:setsize]*100,color=yellow)
plt.plot(np.array(Is)*te_SPAD*255,CRLBlst[setsize:2*setsize]*100,color=blue)
plt.plot(np.array(Is)*te_SPAD*255,CRLBlst[2*setsize:3*setsize]*100,color=green)
plt.plot(np.array(Is)*te_SPAD*255,poiss_CRLBlst[:setsize]*100,linestyle=':', color='black')


plt.plot(0,0,color=yellow,label='CRLB, binomial')
plt.plot(0,0,color='black',label='CRLB, Poisson',linestyle=':')

plt.xscale('log')
plt.legend(loc='lower left')
plt.ylim([0,20])
plt.yticks([0,5,10,15,20])
plt.xlim([70,30000])
plt.ylabel('Uncertainty [nm]')
plt.xlabel('Emitter intensity [photons/image]')
plt.tight_layout()

#%%
# simulation panels
import scipy.stats as sps
import h5py
import numpy as np
from image_model import pixel_values, likelihoods, poiss_likelihoods
from MLE_spad import LM_SPAD_MLE, poiss_LM_SPAD_MLE,CoM, chi2filter
import spadtools as st
import matplotlib.pyplot as plt

Nfact =1
filename = 'data_simulation\\result_sim_N'+str(1)+'.hdf5'
with h5py.File(filename, "r") as f:
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])    
result1 = np.zeros([len(data),15])
for i in range(len(data)):
    for j in range(15):
        result1[i,j] = data[i][j]
result1 = np.delete(result1,[4,5,9,10,11,12,13],1)
del data

mu,sig1 = sps.norm.fit(result1[:,1])
left = 9.7
right = 10.3
x = np.linspace(left,right,500)
n1,bins1 = np.histogram(result1[:,1],np.linspace(left,right,25*Nfact),density=True)
normdata1 = sps.norm.pdf(x,mu,sig1)
    
Nfact =5
filename = 'data_simulation\\result_sim_N'+str(5)+'.hdf5'
with h5py.File(filename, "r") as f:
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])    
result5 = np.zeros([len(data),15])
for i in range(len(data)):
    for j in range(15):
        result5[i,j] = data[i][j]
result5 = np.delete(result5,[4,5,9,10,11,12,13],1)
del data
mu,sig5 = sps.norm.fit(result5[:,1])
left = 9.7
right = 10.3
x = np.linspace(left,right,500)
n5,bins5 = np.histogram(result5[:,1],np.linspace(left,right,20*Nfact),density=True)
normdata5 = sps.norm.pdf(x,mu,sig5)

#%%
squaresize = 1.7

plt.close('2a')
plt.figure('2a',figsize=(2.8,3.2)) 
plt.grid(which='major',linewidth=.8,linestyle='-',zorder=1)
plt.grid(which='minor',linewidth=.8,linestyle='--',zorder=1)
plt.scatter(np.array(Is)*te_SPAD*255,100*np.array(stdlst[:setsize]),color=yellow,s=8,marker='x',label='$\sigma_x$, N=255',zorder=3)
plt.scatter(np.array(Is)*te_SPAD*255,100*np.array(stdlst[setsize:2*setsize]),color=blue,s=8,marker='o',label='$\sigma_x$, N=510',zorder=3)
plt.scatter(np.array(Is)*te_SPAD*255,100*np.array(stdlst[2*setsize:3*setsize]),color=green,s=8,marker='^',label='$\sigma_x$, N=1275',zorder=3)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(CRLBlst[:setsize]),color=yellow,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(CRLBlst[setsize:2*setsize]),color=blue,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(CRLBlst[2*setsize:3*setsize]),color=green,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(poiss_CRLBlst[:setsize]),linestyle=':', color='black',zorder=4,linewidth=2)
plt.plot(0,0,color='black',label='CRLB Binomial')
plt.plot(0,0,color='black',label='CRLB Poisson',linestyle=':')
plt.ylim([0,22])
plt.yticks([0,5,10,15,20])
plt.xscale('log')
plt.legend()
plt.ylabel('Uncertainty [nm]')
plt.xlabel('Emitter intensity [photons/image]')
plt.axvline(x=4500,color=yellow)
plt.axvline(x=7000,color=blue)
plt.tight_layout()
plt.savefig('data_simulation\\2a.svg',format='svg', transparent=True)

#%%
squaresize = 1.2

plt.close('2a2')
plt.figure('2a2',figsize=(2.8,2.8)) 
plt.grid(which='major',linewidth=.8,linestyle='-',zorder=1)
plt.grid(which='minor',linewidth=.8,linestyle='--',zorder=1)
plt.scatter(np.array(Is)*te_SPAD*255,100*np.array(stdlst3[:setsize]),color=yellow,s=8,marker='x',label='$\sigma_x$, N=255',zorder=3)
plt.scatter(np.array(Is)*te_SPAD*255,100*np.array(stdlst3[setsize:2*setsize]),color=blue,s=8,marker='o',label='$\sigma_x$, N=510',zorder=3)
plt.scatter(np.array(Is)*te_SPAD*255,100*np.array(stdlst3[2*setsize:3*setsize]),color=green,s=8,marker='^',label='$\sigma_x$, N=1275',zorder=3)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(CRLBlst3[:setsize]),color=yellow,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(CRLBlst3[setsize:2*setsize]),color=blue,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(CRLBlst3[2*setsize:3*setsize]),color=green,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(poiss_CRLBlst3[:setsize]),linewidth=2,color=yellow,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(poiss_CRLBlst3[setsize:2*setsize]),linewidth=2,color=blue,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(poiss_CRLBlst3[2*setsize:3*setsize]),linewidth=2,color=green,zorder=2)

plt.plot(0,0,color='black',label='CRLB Binomial')
plt.plot(0,0,color='black',label='CRLB Poisson',linestyle=':')
plt.xscale('log')
plt.ylim([0,22])
plt.yticks([0,5,10,15,20])
plt.legend()
plt.ylabel('Uncertainty [nm]')
plt.xlabel('Emitter intensity [photons/image]')
plt.tight_layout()


#%%

plt.close('2a2zoom')
plt.figure('2a2zoom',figsize=(2.8,2.8)) 
plt.grid(which='major',linewidth=.8,linestyle='-',zorder=1)
plt.grid(which='minor',linewidth=.8,linestyle='--',zorder=1)
plt.scatter(np.array(Is)*te_SPAD*255,100*np.array(stdlst3[:setsize]),color=yellow,s=8,marker='x',label='$\sigma_x$, N=255',zorder=3)
plt.scatter(np.array(Is)*te_SPAD*255,100*np.array(stdlst3[setsize:2*setsize]),color=blue,s=8,marker='o',label='$\sigma_x$, N=510',zorder=3)
plt.scatter(np.array(Is)*te_SPAD*255,100*np.array(stdlst3[2*setsize:3*setsize]),color=green,s=8,marker='^',label='$\sigma_x$, N=1275',zorder=3)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(CRLBlst3[:setsize]),color=yellow,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(CRLBlst3[setsize:2*setsize]),color=blue,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(CRLBlst3[2*setsize:3*setsize]),color=green,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(poiss_CRLBlst3[:setsize]),linewidth=2,color=yellow,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(poiss_CRLBlst3[setsize:2*setsize]),linewidth=2,color=blue,zorder=2)
plt.plot(np.array(Is)*te_SPAD*255,100*np.array(poiss_CRLBlst3[2*setsize:3*setsize]),linewidth=2,color=green,zorder=2)

plt.plot(0,0,color='black',label='CRLB Binomial')
plt.plot(0,0,color='black',label='CRLB Poisson',linestyle=':')
plt.xscale('log')
plt.ylim([1,5])
plt.xlim([3900,20000])
plt.legend()
plt.ylabel('Uncertainty [nm]')
plt.xlabel('Emitter intensity [photons/image]')
plt.tight_layout()
#%%
squaresize = 1.2
#1 Histograms of scatter with Gaussian fit
plt.close('2f')
fig2f = plt.figure('2f',figsize=(1.5,1.5))
ax2f = fig2f.add_subplot(1, 1, 1)
ax2f.bar(bins1[:-1]+(bins1[1]-bins1[0])/2,n1*len(result1)/np.sum(n1),bins1[1]-bins1[0],color=blue)
ax2f.plot(x,normdata1*len(result1)/np.sum(n1),color=yellow)
ax2f.set_xlabel('Position [nm]')
ax2f.set_ylabel('Counts')
ax2f.set_yticks([0,40,80])
ax2f.set_xticks([left,10,right])
ax2f.set_xticklabels(['0','30','60'])
ax2f.set_xlim([left,right])
ax2f.set_ylim([0,95])
ax2f.text(10, 82, '$\sigma$ ='+str(np.round(sig1*100,1))+'nm', **text_kwargs)
ax2f.annotate('', xy=(9.95,50), xytext=(9.9,50), arrowprops=dict(arrowstyle='->'))
ax2f.annotate('', xy=(10.04,50), xytext=(10.09,50), arrowprops=dict(arrowstyle='->'))
fig2f.tight_layout()

plt.close('2d')
fig2d = plt.figure('2d',figsize=(squaresize,squaresize))
ax2d = fig2d.add_subplot(1, 1, 1)
im11,xedges,yedges = np.histogram2d(result1[:,1],result1[:,2],10,[[bins1[0],bins1[-1]],[bins1[0],bins1[-1]]])
im1 = ax2d.imshow(im11,cmap='inferno',vmax=80)
ax2d.axis('off')
cbar = fig2d.colorbar(im1,shrink=0.8,ticks=[0, 40, 80])
ax2d.add_patch(Rectangle((5,8.33), 3, .5, alpha=1, facecolor='white',edgecolor='white'))
fig2d.tight_layout()

plt.close('2b')
fig2b = plt.figure('2b',figsize=(squaresize,squaresize))
ax2b = fig2b.add_subplot(1, 1, 1)
imgs1 = st.simulate([[10,10]],[1.02,1.02],10e-6*8973484,1,[20,20],10e-6*8973484/30,255,0, 10e-6)
imgs1[np.where(imgs1==255)] = 255-0.5
im1 = ax2b.imshow(-np.log(1-imgs1[0]/(255)),cmap='inferno',vmin=0,vmax=4)
ax2b.set_xlim([6,14])
ax2b.set_ylim([6,14]) 
ax2b.axis('off')
cbar = fig2b.colorbar(im1,shrink=0.8,ticks=[0, 2, 4])
cbar.ax.set_yticklabels(['0', '2', '>4'])
ax2b.add_patch(Rectangle((11,6.5), 2, .5, alpha=1, facecolor='black',edgecolor='black'))
fig2b.tight_layout()

plt.close('2g')
fig2g = plt.figure('2g',figsize=(1.5,1.5))
ax2g = fig2g.add_subplot(1, 1, 1)
ax2g.bar(bins5[:-1]+(bins5[1]-bins5[0])/2,n5*len(result5)/np.sum(n5),bins5[1]-bins5[0],color=blue)
ax2g.plot(x,normdata5*len(result5)/np.sum(n5),color=yellow)
ax2g.set_xlabel('Position [nm]')
ax2g.set_ylabel('Counts')
ax2g.set_yticks([0,40,80])
ax2g.set_xticks([left,10,right])
ax2g.set_xticklabels(['0','30','60'])
ax2g.set_xlim([left,right])
ax2g.set_ylim([0,95])
ax2g.text(10, 80, '$\sigma$ ='+str(np.round(sig5*100,1))+'nm', **text_kwargs)
ax2g.annotate('', xy=(9.99,50), xytext=(9.95,50), arrowprops=dict(arrowstyle='->'))
ax2g.annotate('', xy=(10.01,50), xytext=(10.05,50), arrowprops=dict(arrowstyle='->'))
fig2g.tight_layout()

plt.close('2e')
fig2e = plt.figure('2e',figsize=(squaresize,squaresize))
ax2e = fig2e.add_subplot(1, 1, 1)
im11,xedges,yedges = np.histogram2d(result5[:,1],result5[:,2],30,[[bins1[0],bins1[-1]],[bins1[0],bins1[-1]]])
im1 = ax2e.imshow(im11,cmap='inferno',vmax=80)
ax2e.axis('off')
cbar = fig2e.colorbar(im1,shrink=0.8,ticks=[0, 40, 80])
ax2e.add_patch(Rectangle((16,25), 9, 1.5, alpha=1, facecolor='white',edgecolor='white'))
fig2e.tight_layout()

plt.close('2c')
fig2c = plt.figure('2c',figsize=(squaresize,squaresize))
ax2c = fig2c.add_subplot(1, 1, 1)
imgs5 = st.simulate([[10,10]],[1.02,1.02],10e-6*8973484/5,1,[20,20],10e-6*8973484/5/30,5*255,0, 10e-6)
imgs5[np.where(imgs5==255*5)] = 255*5-0.5
im5 = ax2c.imshow(-np.log(1-imgs5[0]/(255*5)),cmap='inferno',vmin=0,vmax=4)
ax2c.set_xlim([6,14])
ax2c.set_ylim([6,14]) 
ax2c.axis('off')
cbar = fig2c.colorbar(im5,shrink=0.8,ticks=[0, 2, 4])
cbar.ax.set_yticklabels(['0', '2', '>4'])
ax2c.add_patch(Rectangle((11,6.5), 2, .5, alpha=1, facecolor='white',edgecolor='white'))
fig2c.tight_layout()