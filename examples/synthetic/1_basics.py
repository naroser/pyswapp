#!/usr/bin/env python
# coding: utf-8

# # 2. Basic operations

# Overview of basic library operations and options.

from pyswapp import *


# ### 1. Set the directories

prj_dir = '../../data/syn_data' # project directory
path2raw = os.path.join(prj_dir,'raw') # path to raw data
path2geom = f'{prj_dir}/geometry.csv' # path to the geometry file


# ### 2. User defined settings for processing and plotting

settings = create_settings(fmin=1, fmax=100,                  # frequency range
                           vmin=10, vmax=1000, velstep=1)     # testing phase velocity range and step


# Full list of controllable parameters and their default values:
# 
# - zero_padding = False          # bool, add zeros to increase signal length
#     
# - freq_step=1                   # int, desired frequency step for zero padding
#     
# - normalize = True              # bool, normalize amplitudes
#     
# - local_max = True              # bool, normalize each trace based on local maximum
#     
# - fmin=0                        # float, minimum frequency to analyse
#     
# - fmax=100                      # float, maximum frequency to analyse
#     
# - vmin=100                      # float, minimum testing velocity
#     
# - vmax=1000                     # float, maximum testing velocity
#     
# - velstep=1                     # float, velocity increment
#     
# - SFR_time = 2                  # float, time for computing swept-frequency record (SFR)
#     
# - kmin = 0                      # float, minimum wavenumber
#     
# - kmax = None                   # float, maximum wavenumber
#     
# - normalize_power = True        # bool, normalize amplitudes
#     
# - local_max_power = False       # bool, normalize each trace based on local maximum
# 
# - window_length = 10            # int, spatial window length
# 
# - window_min_offset = -np.inf   # float, minimum offset to exclude near-field
# 
# - window_max_offset = np.inf    # float, maximum offset to exclude near-field
# 
# - window_move_increment = 1     # int, move increment of spatial window

# ### 3. Base Manager Operations

# The BaseManager class can be called specifically to inspect a single data set. Although not its intended application, the class can also be used to manually, one by one, apply data processing to multiple shot files. It is, however, recommended to use the tailored 2D manager classes for 2D processing. Below, basic operations are showcased that can also be performed with the 2D managers.

# #### 3.1 Create a new project

# Create a new project from scratch

swam = BaseManager(f'{prj_dir}/proc/1_basics',              # project directory path
                   path2raw=path2raw,                       # optional, copy & per default rename data from path (outside project directory)
                   path2geom=path2geom,                     # optional, copy geom from path (outside project directory)
                   settings=settings,                       # optional, define settings for visualisations and processing
                   rename = False,                          # optional, rename copied raw data to preferred filename format (Shotfile_<index>), default value is False!
                   overwrite = False,                       # optional, overwrite database .db file if it already exists --> create new project data base
                   )


# #### 3.2 Load a project

# Load an existing project

swam = BaseManager(f'{prj_dir}/proc/1_basics') # project directory path


# #### 3.3 Load/select data

# 3.3.1 load data from database based on existing procset label

swam.load_procset('raw')


# 3.3.2 Set a new procset label

swam.set_procset_label('proc_new')


# 3.3.3 Select a shot file

swam.select_data(sin=1, rep=1, inplace=True)


# #### 3.4 Preprocess data

# Apply one or several pre-processing options to the data

# 3.4.1 Trimming

# Cut traces outside of offset limits considering forward, reverse or both offset shots
swam.preprocess(type = 'trim', by = 'offset', min = 5, max = 48, which = 'both')


# Cut recording time based on top and bottom cut-off values (min and max, respectively)
swam.preprocess(type = 'trim', by = 'time', min = 0, max = 0.5)


# Select traces with specified geophone separation of dx (in the unit of the data)
swam.preprocess(type = 'trim', by = 'separation', dx=2)


# Select traces within a window defined by window midpoint (xmid) and window length (number of traces)
swam.preprocess(type = 'trim', by = 'window', xmid = 25, wlen = 20)


# Select traces
swam.preprocess(type = 'trim', by = 'select', trace_ids = [1,2,3])


# Remove traces
swam.preprocess(type = 'trim', by = 'remove', trace_ids = [1,2,3])


# 3.4.2 Filtering

# Apply bandpass, lowpass or highpass frequency filtering
swam.preprocess(type = 'filter', by = 'frequency', min = 5, max = 20, filter_type = 'bandpass')


# Apply linear move-out with specified velocity
swam.preprocess(type = 'filter', by = 'lmo', velocity = 100, bulk_shift = 0)


# Mute amplitudes of selected traces
swam.preprocess(type = 'filter', by = 'mute', trace_ids = [1,2,3])


# Reverse polarity of selected traces
swam.preprocess(type = 'filter', by = 'reverse_polarity', trace_ids = [1,2,3])


# Apply Hamming window
swam.preprocess(type = 'filter', by = 'taper')


# Filter in FK based on a file containing the filter (points along line to mute data abov or below)
swam.preprocess(type='filter', by = 'FK', fname = 'FKfilter.txt')


# 3.3.3 Import/export filter for whole database

# save picked filter from database to disk for whole procset 
swam.save_filter(fname = 'filter.csv', ftype='FK')


# import filter from disk for whole procset and add to database
swam.import_filter(fname = 'filter.csv')


# apply filter to data for whole procset
swam.apply_filter(ftype='FK')


# 3.4.4 MISC

# remove traces of zero amplitude
swam.preprocess(type='check_traces')


# #### 4. Wave-field transformations

# Perform wavefield transformation based on the phaseshift, fdbfm or radon transform

swam.transform(method = 'phaseshift')


swam.transform(method = 'fdbf')


# #### 5. Automatic dispersion curve extraction

# Automatically extract dispersion curves based on maximum amplitudes or mopa

# automatic dispersion curve extraction using max
swam.extract(method = 'max')


# automatic dispersion curve extraction using MOPA
swam.extract(method = 'MOPA', stopAtChi2 = 1, abs_err = 0.01, showResults =  True)


# #### 6. Save dispersion curves

# Save dispersion curves for each method to disk

# save the manually picked dispersion curves in dispersion images
swam.save(method = 'fdbf')
swam.save(method = 'phaseshift')
swam.save(method = 'radon')


# save the automatically extracted dispersion curves
swam.save(method = 'MOPA')
swam.save(method = 'max')


# #### 7. Plotting 

swam.plot('geometry')


swam.plot('seismogram', amp_scale = 1, color = 'k')


swam.plot('FK')


swam.plot('spectra')


swam.plot('spectrogram')


swam.plot('FV', method = 'phaseshift')


swam.plot('FV', method = 'fdbf')


swam.plot('pseudosection', method = 'MOPA')


swam.plot('curve', method = 'max')


# #### 8. GUI

# Data viewer
swam.gui_view()


# filter data in TX-domain
swam.gui_interact('filter',domain = 'TX')


# filter data in FK-domain
swam.gui_interact('filter',domain = 'FK')


# pick dispersion curves (in FV or FK domain)
swam.gui_interact('pick')

