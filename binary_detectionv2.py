# -*- coding: utf-8 -*-
"""
Binary detection v2

This file uses the .tif data of the TIRF experiment
It uses photonpy and the binary image model to detect emitter locations

Input:
.tif Reference dataset to remove dcr
.tif Dataset to detect emitter locations
Output:
.hdf5 file with emitter locations

"""

import numpy as np
from photonpy import Context, Dataset
import photonpy.cpp.spotdetect as spotdetect
import photonpy.utils.multipart_tiff as tiff
import time
from image_model import results
from MLE_spad import LM_SPAD_MLE, CoM, chi2filter


#Camera parameters
te_SPAD         = 1.5e-5    #Exposure time of SPAD [s]
te_SPAD_dark    = 1.5e-5    #Exposure time of SPAD durrent reference image [s]

imgs_dark       = 130       #Number of reference images
imgs_raw        = 19531     #Number of images in dataset

N               = 2048      #Aggregated frames per image

# Emitter parameters
psfSigma = [1.02,1.02]      #PSF size [pixels]
roisize = int(4 + (psfSigma[0] + 1) * 2)  #ROI size [pixels]

# Cropping image [pixel coordinates]
imgtopleft = [0,0]
imgbotright = [128,256]
imgsize = [imgbotright[0]-imgtopleft[0],imgbotright[1]-imgtopleft[1]]

# Get DCR reference file
dcr = np.zeros([imgsize[0],imgsize[1]])

dcr_reffile = "data_experiment//data_dark_N2048.tif"
for img in tiff.tiff_read_file(dcr_reffile, 0, imgs_dark):
    dcr += img[imgtopleft[0]:imgbotright[0],imgtopleft[1]:imgbotright[1]]/imgs_dark
dcr_ref = -np.log(1-(dcr)/N)/te_SPAD_dark


#%% Get image data set

# Find at DOI: 10.4121/14975013

imgs = np.zeros([imgs_raw,imgsize[0],imgsize[1]])
  
j=0
fn = "data_raw1_N2048.tif"
for img in tiff.tiff_read_file(fn, 0, imgs_raw):
    img = img - 1.1*dcr
    img[np.where(img<0)]=0
    imgs[j,:] = img[imgtopleft[0]:imgbotright[0],imgtopleft[1]:imgbotright[1]]
    j += 1
        
#%% ROI finding

threshold = 1.2*np.mean(imgs) #Treshold set at 1.2x average image intensity
print(f'Threshold: {np.round(threshold,2)}')

print(f"Find ROIs (roisize={roisize})",flush=True)#flush is needed so the tqdm progress bar is not messed up

with Context(debugMode=False) as ctx:   
    context=ctx

    spotDetectorType = spotdetect.SpotDetector(np.mean(psfSigma), roisize, threshold)
    
    sumframes=1

    with Context(context.smlm) as sd_ctx:
        roishape = [roisize,roisize]
    
        img_queue, roi_queue = spotdetect.SpotDetectionMethods(ctx).CreateQueue(
            imgsize, roishape, spotDetectorType, sumframes=1, ctx=sd_ctx)

        numframes = 0
        for img in imgs:
            img_queue.PushFrame(img)
            numframes += 1
            
        # print(f"\nNumframes: {numframes}")
        
        while img_queue.NumFinishedFrames() < numframes//sumframes:
            time.sleep(0.1)
    
        info, rois = roi_queue.Fetch()
        print(f"\nNum spots: {int(rois.shape[0])} ({round(rois.shape[0] / numframes,1)} spots/image)")

#%% Maximum likelihood estimation
# the ROIs and their position in the larger frame are now known, so we can run MLE estimation on them

print("Maximum likelihood estimation")
t = time.time()
successes = 0  #Count succesfully found emitter positions
result = np.zeros([len(rois),12]) #Initiate result output
# [     0             1          2       3    4        5                6            7       8    9    10     11     ]
# [loc in ROI x, loc in ROI y, I_est, bg_est, LL, loc in frame x, loc in frame y, CRLB_x, CRLB_y,its,frame,chi-square]
dcr_rois = np.zeros([len(rois),roisize,roisize]) #Reference DCR
chi2lst = [] #Initiate list of chi-square values for each emitter

for i in range(len(rois)):
    frame = rois[i] #Grab one ROI with potential emitter

    xy0                 = CoM(frame) #initate optimization with centre of mass as location
    x0                  = [xy0[0],xy0[1],np.sum(frame)*10,0] #[x_0,y_0,I_0,bg_0] initial estimate
    theta_est,LL,its    = LM_SPAD_MLE(frame,x0,psfSigma,N,te_SPAD,np.zeros([128,256])) #Maximum LL optimization using Levenberg-Marquardt
    chi2, threshold,Echi2     = chi2filter(frame,theta_est,psfSigma,N,te_SPAD,np.zeros([128,256])) #Get chi-square value for estimate
    chi2lst.append(chi2) #Record chi-square value
    if chi2<2*threshold and theta_est[2]>theta_est[3]: #Only add value if chi-square value is above threshold and intensity is higher than background
        if successes % 100 == 0:
            print(i, successes, info['id'][i])
        result[i,:4]    = theta_est
        result[i,4]     = LL
        result[i,5:7]   = result[i,:2]+[info['y'][i],info['x'][i]]
        result[i,9]     = its
        result[i,10]    = info['id'][i]
        result[i,11]    = chi2
        successes += 1
    
#%% Filtering the estimates
#Remove all locations without a succesfull optimization
result = np.delete(result,np.where(result[:,0]==0),0) 
result = np.delete(result,np.where(np.isnan(result[:,4])),0) 

print(f"MLE done in {np.round(time.time()-t,1)}s")
print(f"filtered {len(rois)-len(result)}/{len(rois)} frames due to chi2 value")

#%% Parameters 
print('Calculating CRLB for estimates')
for i in range(len(result)):
    r = results(roishape,result[i,:4],psfSigma,N,te_SPAD,dcr_rois[i])
    result[i,7:9] = r.CRLB()

result[:,10]= result[:,10]-min(result[:,10])

#%% Save hdf5 file
result_save = Dataset(len(result), 2, [imgsize[0],imgsize[1]])

result_save.pos = np.flip(result[:,5:7],1)
result_save.frame[:] = result[:,10]
result_save.crlb.pos = result[:,7:9]
result_save.photons[:] = result[:,2]
result_save.background[:] = result[:,3]
result_save.chisq[:] = result[:,11]
result_save.save("result_exp2_N"+str(N)+".hdf5")
