------ GENERAL INFORMATION ------ 
This code belongs to the paper:
Theoretical Minimum Uncertainty of Single-Molecule Localizations Using a Single-Photon Avalanche Diode Array
DOI: https://doi.org/10.1364/OE.439340

------ DATA AVAILABILITY ------ 
DOI: 10.4121/14975013

------ REQUIRED PACKAGES ------
python v3.7.9
tifffile v2020.10.1
photonpy v1.0.32 (https://pypi.org/project/photonpy/)


Picasso was used to perform RCC drift correction to go from the output of binary_detectionv2.py to the inputs of experimental_plotter2
It can be downloaded at (https://picassosr.readthedocs.io/en/latest/index.html)

------ RECOMMENDED USAGE ------ 
image_model.py gives the python implementation of the binomial image model described in above paper.
spadtools.py can be used to simulate SPAD images
MLE_SPAD.py provides the Levenberg-Marquardt algorithm needed for the maximum likelihood estimation

binary_detectionv2.py detects emitters from experimental SPAD data
the .hdf5 output should be drift corrected using Picasso
the drift corrected .hdf5 files are included in the folder data_experimental
the drift corrected emitter estimates can be clustered and plotted using experimental_plotter2.py

binary_simulation.py uses spadtools.py to simulate SPAD images and then detect emitters in the simulated images
the results of binary_simulation for a number of aggregated frames are included in the folder data_simulation
the emitter estimates can then be clustered and plotted using simulation_plotter.py

NOTE THAT:
The following files take very long to run:
binary_detectionv2
binary_simulation (this can be shortened by decreasing sim_images or # of simulated intensities (now 24)
experimental_plotter2 (de section 'Plot achieved and theorized uncertainties', de andere secties zijn sneller)


------ Python FILES ------ 

"Binary_detectionv2.py"
This file uses the .tif data of the TIRF experiment
It uses photonpy and the binary image model to detect emitter locations

Input:
.tif Reference dataset to remove dcr
.tif Dataset to detect emitter locations
Output:
.hdf5 file with emitter locations

"binary_simulation.py"
This file simulates binary SPAD data for an ROI with one emitter in a random position
It then locates the emitter using the binary image model

Input:
Emitter intensity
.tif Reference dataset to add and remove dcr
Output:
.hdf5 file with emitter locations
.npy file with emitter locations and characteristics
	
"experimental_plotter2.py"

Takes emitter estimates and clusters them into emitter locations
Then plots achieved and theorized uncertainty for experiments
Also creates histograms of emitter estimate clusters

Input:
.hdf5 files of emitter estimates
Output:
Plots of theorized and achieved uncertainty
Plot of fitted Gaussian distribution to emitter estimates

"simulation_plotter.py"
Takes emitter estimates from simulation and clusters them into emitter locations
Then plots achieved and theorized uncertainty as well as histograms of emitter estimates

Input:
.hdf5 files of emitter estimates from "binary_simulation.py"
.npy files of emitter data from "binary_simulation.py"
Output:
Plots of theorized and achieved uncertainty
Plot of fitted Gaussian distribution to emitter estimates

"image_model.py"
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

"MLE_spad.py"
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

"spadtools.py"
simulates binary frames that are aggregated into a number of images
also capable of providing individual regions of interest for binomial or poissonian situation

input:
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
