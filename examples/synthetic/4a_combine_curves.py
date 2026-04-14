#!/usr/bin/env python
# coding: utf-8

# # 4.a. Dispersion curve combination

# Combine dispersion curves based on two approaches.

from pyswapp import *


# ### 1. Set the directories

prj_dir = '../../data/syn_data' # project directory
path2raw = os.path.join(prj_dir,'raw') # path to raw data
path2geom = f'{prj_dir}/geometry.csv' # path to the geometry file


# ### 2. User defined settings for processing and plotting

settings = create_settings(fmin=10, fmax=50,                  # frequency range
                           vmin=10, vmax=1000, velstep=1)     # testing phase velocity range and step


# ### 3. Set up the MASW2DManager

# Dispersion curve combination is possible with the MASW2D Manager.

# #### 3.1 Create a new project

# Create a new project from scratch

swam = MASW2DManager(f'{prj_dir}/proc/4a_MASW2D',             # project directory path
                   path2raw=path2raw,                       # optional, copy & per default rename data from path (outside project directory)
                   path2geom=path2geom,                     # optional, copy geom from path (outside project directory)
                   settings=settings,                       # optional, define settings for visualisations and processing
                   rename = False,                          # optional, rename copied raw data to preferred filename format (Shotfile_<index>), default value is False!
                   overwrite = True,                       # optional, overwrite database .db file if it already exists --> create new project data base
                   )


# #### 3.2 Load a project

# Load an existing project

swam = MASW2DManager(f'{prj_dir}/proc/4a_MASW2D') # project directory path


# #### 3.3 Preprocess data

# Apply one or several pre-processing options to the data

# 3.3.1 Windowing

# Run moving window along data
swam.moving_window(minoffset = 1, maxoffset = 1e6, wlen = 48, wmove = 1)


# ### 4. Dispersion curve extraction

# automatic dispersion curve extraction using MOPA
swam.extract(method = 'MOPA', stopAtChi2 = 1, abs_err = 0.01)


# ### 5. Dispersion curve combination

# 5.1 Combine dispersion curves based on binning (after Olafsdottir, 2018)

# Settings to define binning intervals
CC_kwargs = {'combination_method' : 'binning',
            'a':15,            # parameter controlling the wavelength interval
            'xlim' : [5,55],  # xlimit for plotting
            'ylim' : [100,500],# ylimit for plotting
            'show': False}      # show the combined dc


# Run combination
swam.combine(method = 'MOPA', filter = False, **CC_kwargs)


# Run combination with prior manual filtering of curves assigned to same surface location
swam.combine(method = 'MOPA', filter = True, **CC_kwargs)


# 5.2 Combine the dispersion curves based on resampling

# Settings
CC_kwargs = {'combination_method' : 'resampling',
            'pmin':10,            # minimum frequency for common frequency range
            'pmax':50,           # maximum frequency for common frequency range
            'pn': 30,            # number of new sampling points
            'pspace': 'log',     # log or linear scale
            'kind': 'cubic',     # kind of interpolation
            'xlim' : [5,80],  # xlimit for plotting
            'ylim' : [50,400],# ylimit for plotting
            'show': False}      # show the combined dc


swam.combine(method = 'MOPA', filter = True, **CC_kwargs)


# ### 6. Process combined dispersion curves

swam.process_CC(type='smooth', method='MOPA')


# ### 7. Plotting

# plot the combined curves as pseudosection
swam.plot_CC(method='MOPA', cmap = 'cividis', vmax = 500)


# #### 8. Save combined dispersion curves

swam.save_CC(method='MOPA')




