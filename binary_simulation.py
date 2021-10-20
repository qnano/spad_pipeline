# -*- coding: utf-8 -*-
"""
Binary simulation

This file simulates binary SPAD data for an ROI with one emitter in a random position
It then locates the emitter using the binary image model

Input:
Emitter intensity
.tif Reference dataset to add and remove dcr
Output:
.hdf5 file with emitter locations
.npy file with emitter locations and characteristics
"""

import numpy as np
from photonpy import Context, Dataset
import photonpy.cpp.spotdetect as spotdetect
import photonpy.utils.multipart_tiff as tiff
import time
from scipy.stats import norm
from image_model import results
from MLE_spad import LM_SPAD_MLE,CoM, chi2filter
import spadtools as st

#%% used for simulating at a list of intensities  
Is = [35000] #Emitter intensity for this simulation [photons]


for k in range(24): #24 simulated intensities spaced at 1.26x from one another
    Is.append(Is[-1]*1.26)

# Lists to record data
stdlst = []
CRLBlst = []
poiss_CRLBlst = []
CRLB_theory = []
poiss_CRLB_theory = []

#%% Parameters 
for Nfact in np.array([1,2,5]): #Number of aggregated frames per image [x255]
    for k in range(len(Is)):
        print(Nfact,k)
        #Camera parameters
        sim_images = 500 #Number of simulated images
        intensity = Is[k]/Nfact #correcting emitter intensity for number of aggregated frames
        bg =  1000 #Background intensity [photons]
        te_SPAD   = 10e-6 + 5e-9 #Simulated SPAD exposure time [s]
        pixel_size = 102.3 #Simulated pixel size [nm]
        N = 255*Nfact #Number of aggregated frames 

        # Emitter parameters 
        psfSigma = [1.02,1.02] #PSF size [pixels]
        roisize = int(4 + (psfSigma[0] + 1) * 2) #ROI size [pixels]
        imgsize = [20,20] #Total image size [pixels]
        emitterList = np.array([[10,10]]) # Locations for emitter(s)
        
        #Correcting intensities for SPAD exposure time
        intensity=intensity*te_SPAD 
        bg=bg*te_SPAD
        
        #Possibility to add drift
        std_drift =0
        
        # blinking characteristics
        p_on=1 #Fraction of time that the emitter is 'ON'
        blink_duration = 2000 #Number of frames that the emitter is 'ON' per blink
        
        # DCR parameters
        ref = 'data_simulation\\data_dark_N2048.tif' #Reference DCR file
        dcr_ref = np.zeros([130,256*128]) #Size of reference DCR file
        i = 0
        for img in tiff.tiff_read_file(ref, 0, 130):

            dcr_ref[i,:] = img.reshape(256*128)
            i += 1
        dcr_ref = -np.log(1-np.mean(dcr_ref,0)/255)/60e-6
        
        # Generate random DCR pattern
        dcr = np.random.choice(dcr_ref,imgsize)
        dcr[np.isnan(dcr)] = 0
        
        #%% Simulate SPAD image

        imgs = st.simulate(emitterList,psfSigma,intensity,sim_images,imgsize,bg,N,dcr, te_SPAD, 2000, 1, std_drift)
        
        #%% ROI finding
        print(f"Find ROIs (roisize={roisize})",flush=True)#flush is needed so the tqdm progress bar is not messed up
        
        #Reference DCR from simulated images
        dcr_av = N*(1-np.exp(-dcr*te_SPAD))
        
        #Threshold for fiding ROI
        threshold=0
        for img in imgs:
            img = img-1*dcr_av
            img[np.where(img<0)]=0
            threshold+=np.mean(img)/int(sim_images)
        k = 23
        threshold = 0.3/(k/3+1)**1.2*(threshold) 
        
        #Get ROI location estimates for each image
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
                    img = img - 1*dcr_av
                    img[np.where(img<0)]=0
                    img_queue.PushFrame(img)
                    numframes += 1
                                
                while img_queue.NumFinishedFrames() < numframes//sumframes:
                    time.sleep(0.1)
            
                info, rois = roi_queue.Fetch()
                print(f"\nNum spots: {int(rois.shape[0])} ({round(rois.shape[0] / numframes,1)} spots/image)")

        #%% Maximum likelihood estimation
        
        # the ROIs and their position in the larger frame are now known, so we can run MLE estimation on them
        
        # Get positions for ROIs
        roipos = np.zeros((len(rois),2),dtype='uint8') 
        roipos[:,0] = info['y']
        roipos[:,1] = info['x']
        
        print("Maximum likelihood estimation")
        t = time.time()
        count = 0 
        
        # result collects the information about the localized emitter positions
        result = np.zeros([len(rois),12])
        dcr_rois = np.zeros([len(rois),roisize,roisize])
        # [     0             1          2       3    4        5                6            7       8    9    10     11     ]
        # [loc in ROI x, loc in ROI y, I_est, bg_est, LL, loc in frame x, loc in frame y, CRLB_x, CRLB_y,its,frame,chi-square]
        chi2lst = [] #Save chi-square values

        # Run optimization for each detected ROI
        for i in range(len(rois)):
            frame   = imgs[info['id'][i]][info['y'][i]:info['y'][i]+roisize,info['x'][i]:info['x'][i]+roisize] #Get ROI
            dcr_roi = dcr[info['y'][i]:info['y'][i]+roisize,info['x'][i]:info['x'][i]+roisize] #Get reference DCR in ROI
            xy0                 = CoM(np.maximum(frame-N*(1-np.exp(-dcr_roi*te_SPAD)),0)) #Remove DCR from ROI
            x0                  = [xy0[0],xy0[1],np.sum(frame),0] #[x_0,y_0,I_0,bg_0] initial estimate
            theta_est,LL,its              = LM_SPAD_MLE(frame,x0,psfSigma,N,te_SPAD,dcr_roi) #Maximum LL optimization using Levenberg-Marquardt
            chi2, threshold, chi2mean     = chi2filter(frame,theta_est,psfSigma,N,te_SPAD,dcr_roi) #Get chi-square values and threshold
            
            chi2lst.append(chi2)
            if chi2<threshold:   #Only save emitters with appropriate chi-square value
                result[i,:4]    = theta_est
                result[i,4]     = LL
                result[i,5:7]   = result[i,:2]+[info['y'][i],info['x'][i]]
                result[i,9]     = its
                result[i,10]    = info['id'][i]
                dcr_rois[i] = dcr_roi


#%% Filtering the estimates
        
        #Remove all locations without a succesfull optimization
        result = np.delete(result,np.where(np.sqrt((result[:,5]-10)**2+(result[:,6]-10)**2)>1),0)
        result = np.delete(result,np.where(result[:,0]==0),0)
        print(f"MLE done in {np.round(time.time()-t,1)}s")
        print(f"filtered {len(rois)-len(result)}/{len(rois)} frames due to chi2 value")

        result = np.delete(result,np.where(np.isnan(result[:,4])),0)
        print(f'Removed {len(np.where(result[:,9]>10))} locations due to nan estimation or >10 iterations')
        
        print('Calculating CRLB for estimates')
        CRLB_temp = []
        poiss_CRLB_temp = []
        for i in range(len(result)):
            r = results(roishape,result[i,:4],psfSigma,N,te_SPAD,dcr_rois[i])
            result[i,7:9] = r.CRLB()
            CRLB_temp.append(r.CRLB()[0])
            poiss_CRLB_temp.append(r.poiss_CRLB()[0])
            
        # #%% Save hdf5 file
        result_save = Dataset(len(result), 2, [imgsize[0],imgsize[1]])
        
        result_save.pos = np.flip(result[:,5:7],1)
        result_save.frame[:] = result[:,10]
        result_save.crlb.pos = result[:,7:9]
        result_save.photons[:] = result[:,2]
        result_save.background[:] = result[:,3]


data_simulation = np.concatenate((np.array([stdlst]),np.array([CRLBlst]),np.array([poiss_CRLBlst]),np.array([CRLB_theory]),np.array([poiss_CRLB_theory])),0)
np.save('data_simulation_constantbg.npy',data_simulation)