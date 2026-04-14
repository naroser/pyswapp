#!/usr/bin/env python
# coding: utf-8

# # 1. geometry file generation

# Create the geometry file automatically if the seismic raw data contains information on source & receiver location.

from pyswapp import *


# #### 1. Set the directories

prj_dir = '../../data/syn_data' # project directory
path2raw = os.path.join(prj_dir,'raw') # path to raw data
path2geom = f'{prj_dir}/geometry.csv' # path where geometry file should be stored


# #### 2. Call the utilities function

create_geometry(path2raw, path2geom=path2geom)

