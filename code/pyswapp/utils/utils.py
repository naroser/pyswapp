import os
import re
from collections.abc import Iterable

import glob
import numpy as np
import pandas as pd

import logging
import shutil

from scipy import interpolate

supported_extensions = ['.sg2','.dat','.syn','.sgy','.syn']

# %% Logging
def create_logging(name):
    """ create logger """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger

def assert_exists(path, check, kind):
    """assert a path exists"""
    if path and not check(path):
        raise FileNotFoundError(f'{kind} "{path}" does not exist.')

def create_projectdir(prjdir = ''):
    """create project directory"""

    subdirs = ['01_data/raw','02_geom','03_proc','04_figs']
    for sd in subdirs:
        safe_makedirs(os.path.join(prjdir,sd))


def rename_files(path2raw, extension = '.sg2', prjdir = '', channel_nr = 1000, rename = False):
    """copy & optionally rename file for easier handling of geometry"""

    if not '.' in extension:
        extension = '.' + extension

    original = []
    for fname in os.listdir(path2raw):
        if fname.endswith(extension):
            original.append(fname)
    original = natural_sort(original)
    renamed = [f'Shotfile_{channel_nr+i}{extension}' for i in range(len(original))]

    # Copy and rename the copied file
    for fname_old,fname_new in zip(original,renamed):
        #path, name = os.path.split(fname)
        shutil.copy(os.path.join(path2raw,fname_old),
                    os.path.join(prjdir,f'01_data/raw/{fname_old}'))

        if rename:
            shutil.move(os.path.join(prjdir,f'01_data/raw/{fname_old}'),
                        os.path.join(prjdir,f'01_data/raw/{fname_new}'))


# %% file tools for reading/writing
def print_inventory(dct):
    """print the dictionary items to console"""
    for key in dct.keys():
        print("{}\t|\t{}".format(key, len(dct[key])))

# %% file tools
def get_num_from_str(string):
    """extract numbers from string"""

    p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'

    if re.search(p, string) is not None:
        return re.findall(p, string)

def natural_sort(l):
    """sort list based on numbers in ascending order"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def read_filter(fname):
    """import existing filters from file"""

    points_top = {}
    points_bot = {}

    with open(fname, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):

        parts = lines[i].strip().split('\t')

        # Expect lines like: t <npoints>   or   b <npoints>
        if parts[0] in ('t', 'b') and len(parts) == 2:
            label = parts[0]
            npoints = int(parts[1])

            # Collect the following npoints lines
            for j in range(i + 1, i + 1 + npoints):
                x, y = lines[j].strip().split('\t')

                x = float(x)
                y = float(y)

                if label == 't':
                    points_top[x] = y
                else:
                    points_bot[x] = y

            # Skip past the block
            i += 1 + npoints
        else:
            i += 1

    return [points_top], [points_bot]

def write_filter(fname, points, key):
    """export filter to file"""

    if len(points) == 0:
        return

    # points is assumed to be a dict {x: y}
    x, y = zip(*sorted(points.items()))

    with open(fname, "w") as f:
        f.write(f"{key}\t{len(x)}\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.4f}\t{yi:.4f}\n")

def filter_df2dict(df):

    if df.empty:
        return {},{}

    top = df[df['key'] == 't'].sort_values('x_value')
    bot = df[df['key'] == 'b'].sort_values('x_value')

    points_top = dict(zip(top['x_value'], top['y_value']))
    points_bot = dict(zip(bot['x_value'], bot['y_value']))

    return points_top, points_bot

def save2csv(outfile, freq, vel, err):
    """save dispersion curve in csv format"""

    f = open(outfile, 'w')
    f.write('#Frequency,Velocity,Velstd\n')
    for line in range(len(freq)):
        f.write('%.3f,%.3f,%.9f\n' %
                (freq[line],
                 vel[line],
                 err[line]))
    f.close()

def get_shotfiles_from_geometry(path2raw, shot_files, extension = '.sg2', sort_ascending = True):
    """get the paths to the shot files from the geometry.csv file"""
    fnames = []
    for fname in os.listdir(path2raw):
        if fname.endswith(extension):
            fnames.append(fname)
    fnames = natural_sort(fnames)

    if not sort_ascending:
        fnames = list(reversed(fnames))

    path2sht = []

    for sht in shot_files:
        if isinstance(sht, Iterable):
            sht_reps = []
            for i, fname in enumerate(fnames):
                sfn = str(re.findall(r'\d+', fname.replace(extension,''))[0])

                for rep in sht:
                    if rep ==sfn:
                        sht_reps.append(os.path.join(path2raw, fnames[i]))

            path2sht.append(sht_reps)
        else:
            for i, fname in enumerate(fnames):
                sfn = str(re.findall(r'\d+', fname.replace(extension,''))[0])

                if sht == sfn:
                    path2sht.append(os.path.join(path2raw, fnames[i]))

    return path2sht


def create_geometry(path2raw: str, path2geom: str = "geometry.csv") -> None:
    """
    create a geometry.csv from seismic shot files (.sgy and .sg2 file formats)

    This function can be used to convert the source and geophone coordinate information contained
    in the seismic raw data to the geometry file format. The geometry file is a csv file that stores
    an abstract representation of the survey layout, that can be optionally passed to some modules.
    Check out docs/geometry_file.pdf for a description of the format.

    Parameters
    ----------
    path2raw: str, paths to shot files
    path2geom: str, path to geometry file
    """

    from pyswapp.stream import SeismicStream

    logger = create_logging(name="Create Geometry")

    if not os.path.isdir(path2raw):
        logger.error(f"Path {path2raw} does not exist.")
        return

    shot_files = natural_sort([entry.path for entry in os.scandir(path2raw) if entry.is_file()])

    if not shot_files:
        logger.error(f"No data in {path2raw}.")
        return

    if shot_files[0].endswith(".syn"):
        logger.error('Files with ".syn" extension do not contain geometry information.')
        return

    # add receiver stations first
    geom_rows = []

    for path in shot_files:
        stream = SeismicStream(path)
        stream.read_data(path, channel_nr=1001, extract_geometry=True)
        stream.set_shot_params()

        receivers = stream.receiver
        ngeo = stream.nchannels

        if len(np.unique(receivers)) != ngeo:
            logger.error(f"Unique geophone coordinates do not match expected count "
                         f"({len(np.unique(receivers))} != {ngeo}) in file {path}")
            return

        # add geophone rows
        for r in receivers:
            geom_rows.append(dict(x=r, y=0, z=0, geo=1, shot="-1", first_geo="-1", ngeo="-1"))

    # remove duplicates & sort
    geom = pd.DataFrame(geom_rows).drop_duplicates(subset=["x"]).sort_values("x")
    geom.insert(0, "id", np.arange(len(geom)))
    geom.reset_index(drop=True, inplace=True)

    x_to_id = dict(zip(geom.x.values, geom.id.values))

    # add the shots
    for i in range(len(shot_files)):

        # seismic stream containing survey geometry
        stream = SeismicStream(shot_files[i])
        stream.read_data(shot_files[i], channel_nr=1001, extract_geometry = True)
        stream.set_shot_params()                    # set the shot parameters from the seismic data

        first_geo = stream.receiver[0]              # first geophone
        ngeo = stream.nchannels                     # number of geophones
        source = stream.source                      # source x-coordinate
        sin = str(int(get_num_from_str(stream.pre )[0]))   # numerical part of file

        if first_geo not in x_to_id:
            logger.error(f"Receiver {first_geo} not found in geometry table.")
            return
        first_geo_id = x_to_id[first_geo] + 1

        if source in x_to_id:
            sid = x_to_id[source]
            geom.loc[sid,'shot'] = _append_to_str_field(geom.loc[sid, "shot"], sin)
            geom.loc[sid, 'first_geo'] = _append_to_str_field(geom.loc[sid, "first_geo"], str(first_geo_id))
            geom.loc[sid, 'ngeo'] = _append_to_str_field(geom.loc[sid, "ngeo"], str(ngeo))
        else:
            # new source row
            new_row = dict(
                x=source,
                y=0,
                z=0,
                geo=0,
                shot=sin,
                first_geo=str(first_geo_id),
                ngeo=str(ngeo),
            )
            geom = pd.concat([geom, pd.DataFrame([new_row])], ignore_index=True)
            x_to_id[source] = geom.index[-1]

    geom = geom.sort_values("x").drop(columns=["id"])
    geom = geom.astype({"shot": str, "first_geo": str, "ngeo": str})
    geom.to_csv(path2geom, index=False, header=False)
    logger.info(f"Geometry written to {path2geom}")

def _append_to_str_field(existing: str, new: str) -> str:
    if existing == "-1":
        return new
    return f"{existing};{new}"

def get_fileList(path2raw):
    """get the paths to the shot files from the geometry.csv file"""

    for ext in supported_extensions:
        fname_list = glob.glob(os.path.join(path2raw, '*' + ext))

        if len(fname_list) > 0:
            return fname_list, ext

    return None, None

def read_DC_csv(fname):
    """read a dispersion curve file from a csv file"""
    dat = np.genfromtxt(fname,skip_header=1,delimiter=',')
    return dat,None


def safe_makedirs(*args):
    """safe generation of directories"""
    try:
        return os.makedirs(*args)
    except OSError:
        pass  # Ignore errors


def combine_dict(d1, d2):
    """combine two dictionaries"""

    for key, value in d2.items():
        if key in d1:
            d1[key].append(value)
        else:
            d1[key] = [value]
    return d1


# %% helper functions for waveform transformation
def nextpow2(A):
    """exponent of next higher power of 2"""
    p = 1
    count = 0
    while p < abs(A):
        p *= 2
        count+=1
    return count,p

# %% Interpolation
def interp(x,y,xx,yerr = None, kind='cubic',**kwargs):
    """interpolate data"""

    f = interpolate.interp1d(x, y, kind=kind, **kwargs)
    yy = f(xx)

    if yerr is not None:
        ferr = interpolate.interp1d(x, yerr, kind=kind, **kwargs)
        yyerr = ferr(xx)
        return xx,yy,yyerr
    else:
        return xx, yy, None