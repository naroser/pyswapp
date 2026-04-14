#!/usr/bin/env python
# coding: utf-8

# # 4.b. Post-process dispersion curves

# Filter, smooth, resample dispersion curves

from pyswapp import *


# ### 1. Set the directories

prj_dir = '../../data/syn_data' # project directory
path2raw = os.path.join(prj_dir,'raw') # path to raw data
path2geom = f'{prj_dir}/geometry.csv' # path to the geometry file


# ### 2. User defined settings for processing and plotting

settings = create_settings(fmin=10, fmax=50,                  # frequency range
                           vmin=10, vmax=1000, velstep=1)     # testing phase velocity range and step


# ### 3. Set up the MASW2DManager

# Dispersion curve processing is possible with all Managers using the identical syntax.

# #### 3.1 Create a new project

# Create a new project from scratch

swam = MASW2DManager(f'{prj_dir}/proc/4b_MASW2D',             # project directory path
                   path2raw=path2raw,                       # optional, copy & per default rename data from path (outside project directory)
                   path2geom=path2geom,                     # optional, copy geom from path (outside project directory)
                   settings=settings,                       # optional, define settings for visualisations and processing
                   rename = False,                          # optional, rename copied raw data to preferred filename format (Shotfile_<index>), default value is False!
                   overwrite = False,                       # optional, overwrite database .db file if it already exists --> create new project data base
                   )


# #### 3.2 Load a project

# Load an existing project

swam = MASW2DManager(f'{prj_dir}/proc/4b_MASW2D') # project directory path


# #### 3.3 Preprocess data

# Apply one or several pre-processing options to the data

# 3.3.1 Trimming

processing_kwargs = {'min': 2, 'max': 26} # arguments for the processing
swam.preprocess(type='trim',by = 'offset',**processing_kwargs)


# ### 4. Dispersion curve extraction

# automatic dispersion curve extraction using MOPA
swam.extract(method = 'MOPA', stopAtChi2 = 1, abs_err = 0.01)


# ### 5. Apply processing to dispersion curves

# 5.1 Smoothing

# Kernel size 
processing_kwargs = {'kernel_size':3}


# Apply smoothing and override old curve in database
# swam.process_curves(type='smooth', method='MOPA', **processing_kwargs)


# Apply smoothing and save as new curve (new procset)
new_procset1 = 'smo'
swam.process_curves(procset= new_procset1,type='smooth', method='MOPA', **processing_kwargs)


# 5.2 Resampling

# New frequency sampling
processing_kwargs = {'pmin':25, 'pmax':45, 'pn':30}


# Apply resampling and override old curve in database
# swam.process_curves(type='resample', method='MOPA', **processing_kwargs)


# Apply resampling and save as new curve (new procset)
new_procset2 = 'res'
swam.process_curves(procset= new_procset2, type='resample', method='MOPA', **processing_kwargs)


# 5.3 Filtering

# Accepted frequeny value range
processing_kwargs = {'pmin':30, 'pmax':40, 'param':'frequency'}


# Accepted velocity value range
# processing_kwargs = {'pmin':100, 'pmax':300, 'param':'velocity'}


# Apply filtering and save as new curve (new procset)
new_procset3 = 'fil'
swam.process_curves(procset= new_procset3, type='filter', method='MOPA', **processing_kwargs)


# 5.3 Estimate error

# Estimate error and save as new curve (new procset)
new_procset4 = 'err'
swam.process_curves(procset= new_procset4,type='estimate_error', method='MOPA')


# ### 6. Plotting

from matplotlib.transforms import ScaledTranslation


fig, ax = plt.subplot_mosaic([['a)'],['b)']])

color = ['lightgrey','dodgerblue','red','k']
marker = ['s','o','o','o']

for i,procset in enumerate(['proc1', new_procset1, new_procset2, new_procset3]):
    swam.plot_curves(apply_to = 'cur', procset= procset, method='MOPA',
                     color = color[i], axes = ax['a)'], 
                     marker = marker[i], size = 60)

swam.plot_curves(apply_to = 'cur', procset= new_procset4,
                 method='MOPA', color = 'k', axes = ax['b)'], showErr = True,
                 label='original', marker = 'o', size = 25, alpha = 1)

for label, ax in ax.items():

    ax.set_xlim([5, 60])
    ax.set_ylim([150, 1000])

    if label in ['a)','b)']:
        ax.text(
            -0.075,0.93, label, transform=(
                ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                va='bottom')

plt.tight_layout()

