#!/usr/bin/env python
# coding: utf-8

# # 2. MASW2D Manager

# Using the MASW2DManager with spatial windowing

from pyswapp import *


# ### 1. Set the directories

prj_dir = '../../data/syn_data' # project directory
path2raw = os.path.join(prj_dir,'raw') # path to raw data
path2geom = f'{prj_dir}/geometry.csv' # path to the geometry file


# ### 2. User defined settings for processing and plotting

settings = create_settings(fmin=1, fmax=100,                  # frequency range
                           vmin=10, vmax=1000, velstep=1)     # testing phase velocity range and step


# ### 3. Basic Operations

# The MASW2DManager class is tailored for 2D processing, i.e., automatically applies all operations to the whole dataset unless otherwise specified.

# #### 3.1 Create a new project

# Create a new project from scratch

swam = MASW2DManager(f'{prj_dir}/proc/2b_MASW2D',             # project directory path
                   path2raw=path2raw,                       # optional, copy & per default rename data from path (outside project directory)
                   path2geom=path2geom,                     # optional, copy geom from path (outside project directory)
                   settings=settings,                       # optional, define settings for visualisations and processing
                   rename = False,                          # optional, rename copied raw data to preferred filename format (Shotfile_<index>), default value is False!
                   overwrite = False,                       # optional, overwrite database .db file if it already exists --> create new project data base
                   )


# #### 3.2 Load a project

# Load an existing project

swam = MASW2DManager(f'{prj_dir}/proc/2b_MASW2D') # project directory path


# #### 3.3 Preprocess data

# Apply one or several pre-processing options to the whole dataset

# 3.3.1 Windowing

# Run moving window along data by specifying:
# 1. the permitted near-offset range for the windows (minoffset and maxoffset)
# 2. the window length in trace counts (wlen)
# 3. the move increment, i.e. by how many traces the window should be moved (wmove)
swam.moving_window(minoffset = 3, maxoffset = 10, wlen = 24, wmove = 10)


# 3.3.2 MISC

# remove traces of zero amplitude
swam.preprocess(type='check_traces')


# ### 4. Wave-field transformations

# Perform wavefield transformation based on the phaseshift, fdbfm or radon transform

swam.transform(method = 'fdbf')


# ### 5. Manual dispersion curve extraction

# Manually extract dispersion curves based on picking

# manual dc picking
swam.gui_interact('pick')


# ### 6. Automatic dispersion curve extraction

# Automatically extract dispersion curves based on maximum amplitudes or mopa

# automatic dispersion curve extraction using max
swam.extract(method = 'max')


# automatic dispersion curve extraction using MOPA
swam.extract(method = 'MOPA', stopAtChi2 = 1, abs_err = 0.01)


# #### 7. Plotting

# Plot the pseudosection

# plot the corresponding pseudosections
swam.plot('pseudosection',method = 'fdbf')
swam.plot('pseudosection',method = 'max')
swam.plot('pseudosection',method = 'MOPA')




