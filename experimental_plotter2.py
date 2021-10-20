# -*- coding: utf-8 -*-
"""
Experimental plotter 2

Takes emitter estimates and clusters them into emitter locations
Then plots achieved and theorized uncertainty for experiments
Also creates histograms of emitter estimate clusters

Input:
.hdf5 files of emitter estimates
Output:
Plots of theorized and achieved uncertainty
Plot of fitted Gaussian distribution to emitter estimates
"""

import h5py
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
from photonpy import Dataset
import numpy as np
from scipy.stats import norm
import scipy.stats as sps
import photonpy.utils.multipart_tiff as tiff
from image_model import results
text_kwargs = dict(ha='center', va='center', fontsize=10)


blue = (57/255, 106/255, 177/255) 
yellow = (218/255,124/255,48/255) 
green = (62/255,150/255,81/255)
red = (204/255,37/255,41/255)

def add_cluster_id(result,dFrames=5,dUncertainty=0.1):
    # Add column for cluster id
    result = np.concatenate((result,np.zeros([len(result),1])),1)
    #initate first frame estimates with their own emitter number
    result[(result[:,0]==0),7] = range(len(result[(result[:,0]==0),:]))
    for frame in range(1,int(max(result[:,0]))+1):
        relevant_estimates = result[(result[:,0]>max(frame-dFrames,-1))&(result[:,0]<frame),:]
        frame_estimates = result[(result[:,0]==frame)]
        for i in range(len(frame_estimates)):
            dist = 1e9
            change = False
            for j in range(len(relevant_estimates)):
                if np.sqrt((frame_estimates[i,1]-relevant_estimates[j,1])**2+(frame_estimates[i,2]-relevant_estimates[j,2])**2) < dist:
                    dist = np.sqrt((frame_estimates[i,1]-relevant_estimates[j,1])**2+(frame_estimates[i,2]-relevant_estimates[j,2])**2)
                    if dist < dUncertainty:
                        frame_estimates[i,7] = relevant_estimates[j,7]
                        change = True
            if change == False:
                frame_estimates[i,7] = max(max(result[:,7])+1,max(frame_estimates[:,7]+1))
        result[(result[:,0]==frame)] = frame_estimates
    return result

def get_cluster_data(result,min_ems_cluster=5):
    # cluster id are necessary
    cluster_data = np.zeros([len(result),7])
    # [clusterid, occurences, meanx, meany, stdx, stdy,meanI]
    for i in range(int(max(result[:,7])+1)):
        cluster_data[i,0] = i
        if np.size(np.where(result[:,7]==i))>min_ems_cluster:
            cluster_data[i,2], cluster_data[i,4] = norm.fit(result[np.where(result[:,7]==i),1])
            cluster_data[i,3], cluster_data[i,5] = norm.fit(result[np.where(result[:,7]==i),2])
            cluster_data[i,1] = np.size(np.where(result[:,7]==i))
            cluster_data[i,6] = np.mean(result[np.where(result[:,7]==i),3])
    return np.delete(cluster_data,np.where(cluster_data[:,1]==0),0)

def histofit(x,nbins=20,pixel_size=102.3):
    plt.close('Histogram of most sampled emitter')
    plt.figure('Histogram of most sampled emitter')
    x = np.squeeze(x)
    meanx, stdx = norm.fit(x)
    n, bins, patches = plt.hist(x-meanx,nbins, density=True)
    y = norm.pdf(bins, 0, stdx)
    plt.plot(bins, y, 'r--', linewidth=2)
    xlabels = np.linspace(-3*stdx,3*stdx,5)*pixel_size
    plt.xticks(xlabels/pixel_size,np.round(xlabels,1).astype(str))
    plt.xlabel('Position [nm]')
    plt.ylabel('Counts')

#%% Plot achieved and theorized uncertainties

pixel_value = 102.3
std_exp = []
stderr_exp = []
ints_exp = []
bin_CRLB_true = []
poiss_CRLB_true = []
violindata = []
meanint =0
meanbg  =0
Ns = [2048, 1024, 512, 256, 128, 64, 32, 16]
tes = [1, 2, 4, 8, 16, 32, 64, 128]
 
for k in range(8):
    print(Ns[k])
    filename = "data_experiment\\result_exp_N"+str(Ns[k])+"_RCC.hdf5"

    te_SPAD = te_SPAD   = 1.5e-5*tes[k]#10e-6 + 5e-9*ks[k]
    with h5py.File(filename, "r") as f:
        a_group_key = list(f.keys())[0]
        data = list(f[a_group_key])    
    result = np.zeros([len(data),9])
    for i in range(len(data)):
        for j in range(9):
            result[i,j] = data[i][j]
    result = np.delete(result,[4,5],1)
    # [frame, xloc, yloc, I, bg, CRLBx, CRLBy]
    del data
    
    result = result[np.where(result[:,3]>np.percentile(result[:,3],20))]
    # Cluster estimates and add cluster_id to result
    
    r = results([8,8],[3.5,3.5,13000,150],[1.08,1.08],Ns[k],te_SPAD,np.zeros([8,8]))
    
    dUncertainty = 5 * r.CRLB()[0]
    
    result = add_cluster_id(result,5,dUncertainty)
    # [frame, xloc, yloc, I, bg, CRLBx, CRLBy,cluster]
    # Get cluster information
    cluster_data = get_cluster_data(result,30)
    cluster_data = cluster_data[np.where(cluster_data[:,4]!=0)]
    
    # plot histo fit of most frequently sampled emitter
    most_sampled_cluster = int(cluster_data[np.argmax(cluster_data[:,1]),0])
    
    meanint = 20000
    std_exp.append(np.median(cluster_data[:,4])*pixel_value)
    stderr_exp.append([(np.percentile(cluster_data[:,4],50)-np.percentile(cluster_data[:,4],25))*pixel_value,(np.percentile(cluster_data[:,4],75)-np.percentile(cluster_data[:,4],50))*pixel_value])

    violindata.append(cluster_data[:,4]*pixel_value)
    ints_exp.append((1/(te_SPAD)))

    r = results([8,8],[3.5,3.5,np.mean(cluster_data[:,6]),np.mean(result[:,4])],[1.08,1.08],(1/(te_SPAD*40)),te_SPAD,np.zeros([8,8]))
    bin_CRLB_true.append(r.CRLB()[0]*pixel_value)
    poiss_CRLB_true.append(r.poiss_CRLB()[0]*pixel_value)

    print(std_exp, bin_CRLB_true[-1], np.mean(result[:,3]))
    meanint += np.mean(result[:,3])/len(Ns)
    meanbg += np.mean(result[:,4])/len(Ns)
    
# Plot CRLB, poiss_CRLB and achieved std. devs.
 
te_SPADs = np.linspace(10e-6,3000e-6,100)
CRLB_ref = []
poiss_CRLB_ref = []
for te_SPAD in te_SPADs:
    r = results([8,8],[3.5,3.5,meanint,meanbg],[1.02,1.02],(1/(te_SPAD*40)),te_SPAD,np.zeros([8,8]))
    CRLB_ref.append(r.CRLB()[0]*pixel_value)
    poiss_CRLB_ref.append(r.poiss_CRLB()[0]*pixel_value)
    
widths=[10000]
for i in range(len(ints_exp)-1): widths.append(widths[-1]/2)

#
plt.close('3a')
plt.figure('3a',figsize=(6,4)) #presentation size
plt.plot(1/te_SPADs,CRLB_ref,c=yellow,label= 'CRLB, binomial ',linestyle='--',zorder=2,linewidth=2)
plt.plot(1/te_SPADs,poiss_CRLB_ref,c=blue,label= 'CRLB, Poisson',linestyle='--',zorder=2,linewidth=2)
plt.errorbar(ints_exp,np.array(std_exp),np.array(stderr_exp).transpose(),c=green,fmt='o',label='$\sigma_x$',capsize=5,zorder=3)
parts=plt.violinplot(violindata,ints_exp,showextrema=False, widths =widths)
for pc in parts['bodies']:
    pc.set_facecolor(green)
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)
plt.grid(which='major',linewidth=.8,linestyle='-',zorder=1)
plt.grid(which='minor',linewidth=.8,linestyle='--',zorder=1)
plt.ylabel('Uncertainty [nm]')
plt.xlabel('Frame rate [s$^{-1}$]')
plt.xscale('log')
plt.ylim([0,20])
plt.xlim([300,110000])
plt.legend()
plt.tight_layout()

# KS distance

Ns = [2048, 1024, 512, 256, 128, 64, 32, 16]
poiss_KS = []
bin_KS   = []
poiss_KS_err = []
bin_KS_err   = []
for N in Ns:
    KS_data = np.load('data_experiment\\ks_data_N'+str(N)+'.npy')
    bin_KS.append(np.mean(KS_data[int(len(KS_data)/4):int(len(KS_data)/2)]))
    poiss_KS.append(np.mean(KS_data[:int(len(KS_data)/4)]))

    bin_KS_err.append([(np.percentile(KS_data[:int(len(KS_data)/4)],50)-np.percentile(KS_data[:int(len(KS_data)/4)],25)),(np.percentile(KS_data[:int(len(KS_data)/4)],75)-np.percentile(KS_data[:int(len(KS_data)/4)],50))])
    poiss_KS_err.append([(np.percentile(KS_data[int(len(KS_data)/4):int(len(KS_data)/2)],50)-np.percentile(KS_data[int(len(KS_data)/4):int(len(KS_data)/2)],25)),(np.percentile(KS_data[int(len(KS_data)/4):int(len(KS_data)/2)],75)-np.percentile(KS_data[int(len(KS_data)/4):int(len(KS_data)/2)],50))])

    

plt.close('KS plot')
plt.figure('KS plot',figsize=(4,4))
plt.ylabel('KS distance')
plt.xlabel('Frame rate [s$^{-1}$]')
plt.grid(which='major',linewidth=.8,linestyle='-',zorder=1)
plt.grid(which='minor',linewidth=.8,linestyle='--',zorder=1)
plt.errorbar(np.array(ints_exp),np.array(poiss_KS),np.array(poiss_KS_err).transpose(),c=blue,label= 'Poissonian',marker='s',zorder=2,capsize=5)
plt.errorbar(ints_exp,np.array(bin_KS),np.array(bin_KS_err).transpose(),c=yellow,label= 'Binomial',marker='o',zorder=2,capsize=5)
plt.xscale('log')
plt.ylim([0.05,0.25])
plt.legend()
plt.tight_layout()
plt.annotate('N=16', xy=(540,0.19), xytext=(700,0.2), arrowprops=dict(arrowstyle='->'))
plt.annotate('N=2048', xy=(63000,0.11), xytext=(28000,0.08), arrowprops=dict(arrowstyle='->'))



#%% Picked nanorulers
filename = "data_experiment\\result_exp_N2048_RCC_picked3.hdf5"

with h5py.File(filename, "r") as f:
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])    
result = np.zeros([len(data),15])
for i in range(len(data)):
    for j in range(15):
        result[i,j] = data[i][j]
result = np.delete(result,[4,5,9,10,11,12,13],1)
# [frame, xloc, yloc, I, bg, CRLBx, CRLBy, chi2]
del data

angle = np.pi/30



ox, oy = [86.3,48.2]
qx = ox + np.cos(angle) * (result[:,1] - ox) - np.sin(angle) * (result[:,2] - oy)
qy = oy + np.sin(angle) * (result[:,1] - ox) + np.cos(angle) * (result[:,2] - oy)

result_save = Dataset(len(result), 2, [120,140])
result_save.pos = np.concatenate((qx.reshape([len(qx),1])-np.min(qx),qy.reshape([len(qy),1])-np.min(qy)),1)
result_save.crlb.pos = np.ones([len(qx),2])*0.1
result_save.photons[:] = 10000
result_save.background[:] = 300

left = 85.9
right = 86.7
leftmost = left-0.7
rightmost = right+0.7
mu1,sig1 = sps.norm.fit(qx[np.where(qx<left+0.1)])
mu2,sig2 = sps.norm.fit(qx[(qx>left) & (qx<right-0.1)])
mu3,sig3 = sps.norm.fit(qx[(qx>right+0.3)])
x = np.linspace(leftmost,rightmost,500)
normdata = sps.norm.pdf(x,mu1,sig1)*12+sps.norm.pdf(x,mu2,sig2)*24+sps.norm.pdf(x,mu3,sig3)*9.5


plt.close('3c')
plt.figure('3c',figsize=(1.8,1.8))
plt.plot(x,normdata,c=yellow)
plt.hist(qx,np.linspace(leftmost,rightmost,35),color=blue)
plt.xlabel('Position [nm]')
plt.ylabel('Counts')
plt.ylim([0,150])
plt.yticks([0,75,150])
plt.xticks([85.0,86.25,87.5],['0','125','250'])


leftblob = 30
midblob = 65
rightblob = 25
plt.annotate('', xy=(mu1-.5*sig1,leftblob), xytext=(mu1-0.3,leftblob), arrowprops=dict(arrowstyle='->'))
plt.annotate('', xy=(mu1+.5*sig1,leftblob), xytext=(mu1+0.3,leftblob), arrowprops=dict(arrowstyle='->'))    
plt.annotate('', xy=(mu2-.5*sig2,midblob), xytext=(mu2-0.3,midblob), arrowprops=dict(arrowstyle='->'))
plt.annotate('', xy=(mu2+.5*sig2,midblob), xytext=(mu2+0.3,midblob), arrowprops=dict(arrowstyle='->'))   
plt.annotate('', xy=(mu3-.5*sig3,rightblob), xytext=(mu3-0.3,rightblob), arrowprops=dict(arrowstyle='->'))
plt.annotate('', xy=(mu3+.5*sig3,rightblob), xytext=(mu3+0.3,rightblob), arrowprops=dict(arrowstyle='->'))   
plt.text(mu1, leftblob+35, str(int(sig1*100))+' nm', **text_kwargs)
plt.text(mu2, midblob+60, str(int(sig2*100))+' nm', **text_kwargs)
plt.text(mu3-0.1, rightblob+65, str(int(sig3*100))+ ' nm', **text_kwargs)
plt.tight_layout()


filename = "data_experiment\\result_exp_N16_RCC_picked3.hdf5"

with h5py.File(filename, "r") as f:
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])    
result = np.zeros([len(data),15])
for i in range(len(data)):
    for j in range(15):
        result[i,j] = data[i][j]
result = np.delete(result,[4,5,9,10,11,12,13],1)
# [frame, xloc, yloc, I, bg, CRLBx, CRLBy, chi2]
del data

angle = np.pi/30



ox, oy = [86.3,48.2]
qx = ox + np.cos(angle) * (result[:,1] - ox) - np.sin(angle) * (result[:,2] - oy)
qy = oy + np.sin(angle) * (result[:,1] - ox) + np.cos(angle) * (result[:,2] - oy)

result_save = Dataset(len(result), 2, [120,140])
result_save.pos = np.concatenate((qx.reshape([len(qx),1])-np.min(qx),qy.reshape([len(qy),1])-np.min(qy)),1)
result_save.crlb.pos = np.ones([len(qx),2])*0.1
result_save.photons[:] = 10000
result_save.background[:] = 300

left = 85.9
right = 86.7
leftmost = left-0.7
rightmost = right+0.7
mu1,sig1 = sps.norm.fit(qx[np.where(qx<left-0.1)])
mu2,sig2 = sps.norm.fit(qx[(qx>left+.1) & (qx<right-0.1)])
mu3,sig3 = sps.norm.fit(qx[(qx>right)])
x = np.linspace(leftmost,rightmost,500)
normdata = sps.norm.pdf(x,mu1,sig1)*12+sps.norm.pdf(x,mu2,sig2)*21+sps.norm.pdf(x,mu3,sig3)*9.5


plt.close('3c2')
plt.figure('3c2',figsize=(1.8,1.8))
plt.plot(x,normdata,c=yellow)
plt.hist(qx,np.linspace(leftmost,rightmost,35),color=blue)
plt.xlabel('Position [nm]')
plt.ylabel('Counts')
plt.ylim([0,80])
plt.yticks([0,40,80])
plt.xticks([85.0,86.25,87.5],['0','125','250'])


leftblob = 25
midblob = 40
rightblob = 20
plt.annotate('', xy=(mu1-.5*sig1,leftblob), xytext=(mu1-0.3,leftblob), arrowprops=dict(arrowstyle='->'))
plt.annotate('', xy=(mu1+.5*sig1,leftblob), xytext=(mu1+0.3,leftblob), arrowprops=dict(arrowstyle='->'))    
plt.annotate('', xy=(mu2-.5*sig2,midblob), xytext=(mu2-0.3,midblob), arrowprops=dict(arrowstyle='->'))
plt.annotate('', xy=(mu2+.5*sig2,midblob), xytext=(mu2+0.3,midblob), arrowprops=dict(arrowstyle='->'))   
plt.annotate('', xy=(mu3-.5*sig3,rightblob), xytext=(mu3-0.3,rightblob), arrowprops=dict(arrowstyle='->'))
plt.annotate('', xy=(mu3+.5*sig3,rightblob), xytext=(mu3+0.3,rightblob), arrowprops=dict(arrowstyle='->'))   
plt.text(mu1, leftblob+22, str(int(sig1*100))+' nm', **text_kwargs)
plt.text(mu2, midblob+25, str(int(sig2*100))+' nm', **text_kwargs)
plt.text(mu3, rightblob+13, str(int(sig3*100))+ ' nm', **text_kwargs)
plt.tight_layout()


#%% Bright corner

fn = "data_experiment\\data_raw_N190_k24273 (1).tif"
for img in tiff.tiff_read_file(fn, 0, 1):
    img = img
    
plt.close('brightcorner')
plt.figure('brightcorner',figsize=(8,4))
plt.imshow(img,cmap='gray')
plt.tight_layout()


#%% 3 pixel DCR
fn = "data_experiment\\data_dark_N20482.tif"
px1 = []
px2 = []
px3 = []

imgtotal = np.zeros([128,256])
for img in tiff.tiff_read_file(fn, 0, 1900):
    img = img.astype('float64')
    img -= 1
    
    px1.append((-np.log(1-img[95,95]/2048))/1.e-5)
    px2.append((-np.log(1-img[41,94]/2048))/1.5e-5)
    px3.append((-np.log(1-img[18,184]/2048))/1.5e-5)

plt.figure()
plt.plot(np.arange(len(px1))*0.0315,px1,label='pixel 1: mean = '+str(int(np.mean(px1)))+' cps')
plt.step(np.arange(len(px1))*0.0315,px2,label='pixel 2: mean = '+str(int(np.mean(px2)))+' cps')
plt.plot(np.arange(len(px1))*0.0315,px3,label='pixel 3: mean = '+str(int(np.mean(px3)))+' cps')
plt.xlabel('Time [s]')
plt.ylabel('Dark count rate [cps]')
plt.yscale('log')
plt.ylim([20,100000])
plt.legend()