General information
---
This code belongs to the paper:
Theoretical Minimum Uncertainty of Single-Molecule Localizations Using a Single-Photon Avalanche Diode Array
DOI: https://doi.org/10.1364/OE.439340

Data availability
---
https://doi.org/10.4121/14975013

Windows installation
---
The SPAD code was developed in Python v3.7.9, running on Windows 8.1. The code was also tested for Windows 10.

Steps:

1. Clone repository or download software from https://github.com/qnano/spad_pipeline. If necessary, extract the code in the directory '/spad_pipeline/'

2. Install the Anaconda distribution: https://www.anaconda.com/products/individual. 

3. We recommend creating and activating a new environment for spad_pipeline, as follows:

    - Open the Anaconda prompt.

    - Run the following commands in the terminal:
    ```
        conda create -n spad_env python=3.7.9 anaconda
        conda activate spad_env
    ```
    - Keep the terminal open for step 4.

4. The following packages are needed:

    - tifffile (tested version 2020.10.1). In the terminal from step 3, run:
    ```pip install tifffile==2020.10.1```
    
    - photonpy (tested version 1.0.32). In the terminal from step 3, run:
    ```pip install photonpy==1.0.32```

3. Install Picasso (tested v0.2.8), following the instructions for one-click installation on https://github.com/jungmannlab/picasso.


Data processing
---
We describe the experimental and simulation pipelines that are needed to reproduce the figures from the paper.

### Experimental data

This section describes the experimental data pipeline, using the experimental data from the paper. We will go through the steps to analyse the dataset:
To directly reproduce the result from processed data, steps 1-6 can be omitted by using the available processed data in the directory "/spad_pipeline/data_experimental/" (found on https://doi.org/10.4121/14975013).

1. Download the raw data from https://doi.org/10.4121/14975013 and extract it in the directory "/spad_pipeline/". Verify that the directories "/spad_pipeline/data_experimental/" and "/spad_pipeline/data_simulation/" now exist.

2. Choose the number of aggregated frames you want to analyze. This is denoted by the number after 'N' in the data files: e.g. data_raw1_N2048 has 2048 aggregated frames per image. One needs to run all available N values to reproduce the paper plots.

3. In binary_detectionv2 ensure that the N in line 32 corresponds to that of the loaded data in lines 49 and 59. 

4. Now run binary_detectionv2. This will likely take a long time (~5 hours on tested hardware). This will:
		
    - load the raw TIRF images and the raw dcr images,
		
    - remove DCR from TIRF images,
		
    - detect ROIs that likely contain a spot based on the threshold set on line 68,
		
    - estimate the emitter location within the ROI,
		
    - filter out any locations with bad chi-square values,
		
    - save a .hdf5 file of the accepted emitter locations.
	
5. Load the outputted .hdf5 file, called "result_exp2_N"+str(N)+".hdf5" in Picasso Render and perform RCC drift correction. If needed, instructions can be found on (https://picassosr.readthedocs.io/en/latest/).

6. Save resulting locations as "result_exp2_N"+str(N)+"_RCC.hdf5" in the directory "/spad_pipeline/data_experimental/". 

7. Plot achieved and theorized uncertainty as shown in figure 3 of the paper using experimental_plotter, this will take a long time (~1 hour on tested hardware). This will:
		
    - cluster estimated locations per emitter,
		
    - calculate uncertainty per emitter,
		
    - calculate CRLB per emitter,
		
    - plot CRLB and calculated uncertainty for all available aggregated frame counts. The sample code provides N=[2048, 1024, 512, 256, 128, 64, 32, 16],
		
    - plot supplementary plots of the paper using provided data in the directory "/spad_pipeline/data_experimental/".
	
### Simulation data

This section describes the simulation data pipeline. We will go through the steps to analyse the dataset:
To directly reproduce the result from processed data, steps 1-4 can be omitted by using the available processed data in the directory "/spad_pipeline/data_simulation/" (found on https://doi.org/10.4121/14975013).

1. Download the raw data from https://doi.org/10.4121/14975013 and place it in the directory "/spad_pipeline/". Verify that the directories "/spad_pipeline/data_experimental/" and "/spad_pipeline/data_simulation/" now exist.

2. Open binary_simulation

3. Select the number of sampled images (line 47) and intensities (line 32). The latter are now spaced at 1.26x from one another starting at 35000 photons.

4. Select the number of emitters and their locations (line 58).

5. Run binary_simulation. This will:
		
    - create simulated SPAD images containing the emitters,
		
    - localize the ROIs that likely contain the emitters,
		
    - estimate the emitter location within the ROI,
		
    - filter out any locations with bad chi-square values,
		
    - save a .npy file with data on the simulated emitters and the estimated location.

6. Run simulation_plotter. This will generate the plots in Figure 2 of the paper.


Description of individual Python files
---
### binary_detectionv2.py
This file uses the .tif data of the TIRF experiment
It uses photonpy and the binary image model to detect emitter locations

**Input**

.tif Reference dataset to remove DCR

.tif Dataset to detect emitter locations

**Output**

.hdf5 file with emitter locations

### binary_simulation.py
This file simulates binary SPAD data for an ROI with one emitter in a random position
It then locates the emitter using the binary image model

**Input**

Emitter intensity

.tif Reference dataset to add and remove dcr

**Output**

.hdf5 file with emitter locations

.npy file with emitter locations and characteristics
	
### experimental_plotter2.py

Takes emitter estimates and clusters them into emitter locations
Then plots achieved and theorized uncertainty for experiments
Also creates histograms of emitter estimate clusters

**Input**

.hdf5 files of emitter estimates

**Output**

Plots of theorized and achieved uncertainty

Plot of fitted Gaussian distribution to emitter estimates

### simulation_plotter.py
Takes emitter estimates from simulation and clusters them into emitter locations
Then plots achieved and theorized uncertainty as well as histograms of emitter estimates

**Input**

.hdf5 files of emitter estimates from "binary_simulation.py"

.npy files of emitter data from "binary_simulation.py"

**Output**

Plots of theorized and achieved uncertainty

Plot of fitted Gaussian distribution to emitter estimates

### image_model.py
Contains the expressions for the binomial and Poissonian image model
Also contains first and second derivatives of the likelihood
Expressions of binomial image model are derived in the supplement of DOI: 10.1364/OE.439340
Expression of poissonian image model are derived in the supplement of DOI: 10.1038/nmeth.1449

**Input**

ROI: intensity matrix in ROI

sigma: emitter psf [sigma_x,sigma_y]

theta: emitter parameters: [loc_x,loc_y,intensity,background]

N: aggregated binary frame per image

te_SPAD: exposure time SPAD
    
**Output**

Binomial image model

Binomial image model, first derivative

Binomial image model, second derivative

Binomial likelihood

Binomial likelihood, first derivative

Binomial likelihood, second derivative

Binomial likelihood, Cramér-Rao lower bound (CRLB)

Poissonian likelihood

Poissonian likelihood, first derivative

Poissonian likelihood, second derivative

Poissonian likelihood, Cramér-Rao lower bound (CRLB)

### MLE_spad.py
Performs maximum likelihood estimation using Levenberg Marquardt

**Input**

ROI: intensity matrix in ROI

sigma: emitter psf [sigma_x,sigma_y]

theta: emitter parameters: [loc_x,loc_y,intensity,background]

N: aggregated binary frame per image

te_SPAD: exposure time SPAD
    
**Output**

Centre of mass

Optimized theta for binomial image model

Optimized theta for poissonian image model

Chi-square value and threshold for ROI

### spadtools.py
Simulates binary frames that are aggregated into a number of images
Also capable of providing individual regions of interest for binomial or poissonian situation

**Input**

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
    
**Output**

Aggregated binomial images [sim_images x imgsize]

Individual binomial  ROI

Individual poissonian ROI
