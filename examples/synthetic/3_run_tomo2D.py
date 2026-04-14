#!/usr/bin/env python
# coding: utf-8

# # 3. Tomographic like approach

# Using the Tomo2D Manager

from pyswapp import *


# ### 1. Set the directories

prj_dir = '../../data/syn_data' # project directory
path2raw = os.path.join(prj_dir,'raw') # path to raw data
path2geom = f'{prj_dir}/geometry.csv' # path to the geometry file


# ### 2. User defined settings for processing and plotting

settings = create_settings(fmin=10, fmax=50,                  # frequency range
                           vmin=10, vmax=1000, velstep=1)     # testing phase velocity range and step


# ### 3. Basic Operations

# The Tomo2DManager class can be called specifically to perform the tomoraphic-like approach.

# #### 3.1 Create a new project

# Create a new project from scratch

swam = Tomo2DManager(f'{prj_dir}/proc/3_tomo2D',             # project directory path
                   path2raw=path2raw,                       # optional, copy & per default rename data from path (outside project directory)
                   path2geom=path2geom,                     # optional, copy geom from path (outside project directory)
                   settings=settings,                       # optional, define settings for visualisations and processing
                   rename = False,                          # optional, rename copied raw data to preferred filename format (Shotfile_<index>), default value is False!
                   overwrite = False,                       # optional, overwrite database .db file if it already exists --> create new project data base
                   )


# #### 3.2 Load a project

# Load an existing project

swam = Tomo2DManager(f'{prj_dir}/proc/3_tomo2D') # project directory path


# #### 3.3 Preprocess data

# Apply one or several pre-processing options to the data

# 3.3.1 Prepare streams

# Retrieve subsets from data corresponding to forward and reverse shots
swam.prepare_streams(min_offset=2, max_offset=1e6, min_rec =16, max_rec=48)


# 3.3.2 Filtering

# Isolate the fundamental mode in FK domain
swam.gui_interact('filter','FK')


# 3.3.4 Phase differences

# compute phase differences
swam.compute_phasediff()


# ### 4. Run inversion

# Run tomographic-like approach with fixed regularization strength

# # Run the tomo2D
swam.run(lam = 20, min_offset = 2, max_offset = 1e6, rel_err = 3/100)


# Run tomographic-like approach with optimized regularization strength

# # Run the tomo2D
swam.run(min_offset = 2, max_offset = 1e6, rel_err = 3/100, opt_lam=True)


# ### 5. Plotting

# Plot the pseudosection

swam.plot('pseudosection', method = 'tomo2D', cmap='cividis',vmax=500, width = 0.3)


# ### 6. Save dispersion curves

# Save dispersion curves for each method to disk

# save the manually picked dispersion curves in dispersion images
swam.save(method='tomo2D')

