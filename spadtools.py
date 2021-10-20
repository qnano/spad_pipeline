# -*- coding: utf-8 -*-
"""
Spad tools

Simulates binary frames that are aggregated into a number of images
Also capable of providing individual regions of interest for binomial or poissonian situation

Input:
emitterList: emitter locations [x,y]
sigma: emitter psf [sigma_x,sigma_y]
intensity: emitter intensity 
sim_images=1: number of images to be simulated
imgsize=[40,40]: size of image [x,y]
N=255: aggregated binary frame per image
bg=0: image background
dcr=0: dcr array of imgsize, 0 if no dcr should be simulated
te_SPAD=10e-6: exposure time SPAD
blink_duration=1e9: duration of emitter blinking in number of frames
p_on: probability of emitter being 'on'
std_drift=0: drift per image
    
Output:
Aggregated binomial images [sim_images x imgsize]
Individual binomial  ROI
Individual poissonian ROI

"""

# Aggregated binomial images [sim_images x imgsize]
def simulate(emitterList, sigma, intensity, sim_images=1, imgsize=[40,40], bg=0, N=255,  
                    dcr=0, te_SPAD=10e-6, blink_duration=1e9, p_on=1, std_drift=0, save=False):

    
    from photonpy import Context
    import numpy as np
    import tifffile
    from photonpy.cpp.gaussian import Gaussian
    
    if abs(np.sum(dcr)) == 0:
        dcr = np.zeros([imgsize[0],imgsize[1]])
    
    with Context(debugMode=False) as ctx:
        print("Generating own example movie")
        
        emitters = np.array([[e[0], e[1], sigma[0], sigma[1], intensity] for e in emitterList])
    
        on_counts = np.zeros(sim_images*N, dtype=np.int32)
    
        gaussianApi = Gaussian(ctx) 
        
        # Generate blinking states
        states = np.random.rand(len(emitters))*100
        blinkstep = 100/blink_duration
        on = np.random.binomial(1,p_on,len(emitterList))
        # fn = "aggregated_movie"+now.strftime("%H%M%S")+".tif"
        
        imgs = np.zeros([sim_images,imgsize[0], imgsize[0]])

        for img in range(int(sim_images)):
            frames = np.zeros((N, imgsize[0], imgsize[0]), dtype=np.float32)
            #Drift:
            emitters[:,:2] += np.random.normal(0,std_drift,2)*np.ones([len(emitters),2])
            for f in range(N):
                frame = bg * np.ones((imgsize[0], imgsize[1]), dtype=np.float32)
                frame_emitters = emitters * 1
                      
                for spot in range(len(emitters)):
                    states[spot] += blinkstep
                    if states[spot]>=100:
                        states[spot] = 0
                        on[spot] = np.random.binomial(1,p_on)
    
                    
                frame_emitters[:, 4] *= on
        
                frames[f] = gaussianApi.Draw(frame, frame_emitters)
                on_counts[f] = np.sum(on)
            mov = np.random.poisson(frames+dcr*te_SPAD)
            mov[np.where(mov>=1)]=1
            imgs[img] = np.sum(mov,0)

            
        
            # tifffile.imwrite(fn, agg, append=True)
            
        return imgs
    
# Individual binomial  ROI
def SPADroi(theta, sigma,N, te_SPAD, imgsize = [8,8]):
    from photonpy import Context
    import numpy as np
    from photonpy.cpp.gaussian import Gaussian
    
    with Context(debugMode=False) as ctx:
        
        emitters = np.array([[theta[0], theta[1], sigma[0], sigma[1], theta[2]*te_SPAD]])
    
        # on_counts = np.zeros(N, dtype=np.int32)
    
        gaussianApi = Gaussian(ctx) 
        
        img = np.zeros((imgsize[0], imgsize[1]), dtype=np.float32)
        for i in range(N):
            frame = theta[3]*te_SPAD * np.ones((imgsize[0], imgsize[1]), dtype=np.float32)
            frame_emitters = emitters * 1
            frames = gaussianApi.Draw(frame, frame_emitters)
            frames = np.random.poisson(frames)
            frames[np.where(frames>1)]=1
            img += frames
    
        return img
   
# Individual poissonian ROI
def roi(theta, sigma,N, te_SPAD, imgsize = [8,8]):
    from photonpy import Context
    import numpy as np
    from photonpy.cpp.gaussian import Gaussian
    
    with Context(debugMode=False) as ctx:
        
        emitters = np.array([[theta[0], theta[1], sigma[0], sigma[1], theta[2]*te_SPAD*N]])
    
        # on_counts = np.zeros(N, dtype=np.int32)
    
        gaussianApi = Gaussian(ctx) 
        frames = np.zeros((imgsize[0], imgsize[1]), dtype=np.float32)
        frame = theta[3]*te_SPAD*N * np.ones((imgsize[0], imgsize[1]), dtype=np.float32)
        frame_emitters = emitters * 1
        frames = gaussianApi.Draw(frame, frame_emitters)
    
        return np.random.poisson(frames)