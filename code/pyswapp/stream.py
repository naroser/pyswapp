"""python class for manipulating an Obspy stream"""
import copy
import os.path
from collections import OrderedDict
import numbers

import numpy as np
from scipy import signal, special, sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import rfft, rfftfreq
from scipy.linalg import solve
import obspy
import obspy.signal

from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredText

from .curve import DispersionCurve
from .utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


class SeismicStream:
    """class to manipulate a seismic record for the analysis of surface waves"""

    def __init__(self, fname, settings = None,
                 channel_nr=1001, pre_trigger=0):
        """
        initialize stream class and settings

        Parameters
        ----------
        fname : str, containing a file name
        settings : dict, settings
        channel_nr : int, optional
        pre_trigger : float, optional
        """

        # logger
        self.logger = create_logging(name = 'STREAM')

        # seismic record filename
        self._st = None
        self._pst = None

        self.fname = fname
        _, fi = os.path.split(self.fname)
        pre, ext = os.path.splitext(fi)
        self.pre = pre
        self.ext = ext

        # initialization of parameters
        self._init_params()

        # processing settings
        if settings is not None:
            self._settings = settings
        else:
            self._settings = create_settings()

        # set processing settings
        self._apply_settings()

        # read the seismic record file and initialize the stream object
        self.channel_nr = channel_nr

        if os.path.isfile(fname):
            self.read_data(fname=fname,
                           channel_nr = self.channel_nr,
                           pre_trigger=pre_trigger)
        else:
            raise FileNotFoundError

    def _apply_settings(self):
        """apply processing settings from settings DataFrame
        for pre-processing, wave field transformation and picking"""

        # preprocessing settings
        self.pad = self._settings['zero_padding'].item() # apply padding
        self.df = self._settings['df'].item() # step length

        self.norm_amps = self._settings['normalize_amps'].item()  # apply amplitude normalization
        self.use_local_max = self._settings['local_max'].item() # use local maxima

        # DC picking settings
        self._pck_mode = 'manual' # picking mode

        # wave field transformation settings
        self.fmin = self._settings['fmin'].item()  # min frequency
        self.fmax = self._settings['fmax'].item()  # max frequency
        self.vmin = self._settings['vmin'].item()  # min testing velocity
        self.vmax = self._settings['vmax'].item()  # max testing velocity
        self.vstep = self._settings['velstep'].item()  # velocity increment
        self.SFR_time = self._settings['SFR_time'].item()  # time for the SFR

        # plotting
        self.kmin = self._settings['kmin'].item() # min wave number
        self.kmax = self._settings['kmax'].item() # max wave number
        self.norm_power = self._settings['normalize_amps'].item() # apply amplitude normalization
        self.use_local_max_power = self._settings['local_max_power'].item() # use local maxima

        # windowing
        self.wlen = self._settings["window_length"].item() # length of the window
        self.wmino = self._settings["window_min_offset"].item() # minimum offset
        self.wmaxo= self._settings["window_max_offset"].item() # maximum offset
        self.wmove = self._settings["window_move"].item() # move increment

    def read_data(self, fname, channel_nr = 1001, pre_trigger=0, extract_geometry = False):
        """
        read a shotfile and extract relevant information

        Parameters
        ----------
        fname : str, file name
        channel_nr : int, optional
        pre_trigger : float, optional
        extract_geometry : bool, True if geometry should be extracted, False otherwise
        """

        # read and update the header of the .sg2 file
        if fname.endswith('.sg2') or fname.endswith('.dat'):
            if extract_geometry:
                self._read_sg2_geom(fname, channel_nr)
            else:
                self._read_sg2(fname, channel_nr)

        # read and update header of the .sgy file
        elif fname.endswith('.sgy'):
            if extract_geometry:
                self._read_sgy_geom(fname, channel_nr)
            else:
                self._read_sgy(fname, channel_nr)

        # read and update header of synthetic data
        elif fname.endswith('.syn') or fname.endswith('.mseed'):
            self._read_syn(fname, channel_nr, pre_trigger)
        else:
            raise NotImplementedError("File format not supported.")

    def _read_sg2_geom(self, fname, channel_nr):
        """
        Read stream data with .sg2 file format with geometry information in header

        Parameters
        ----------
        fname : str, file name
        channel_nr : int, optional
        """

        st_new = obspy.Stream()
        st = obspy.read(fname)
        for ti, trace in enumerate(st):

            if len(trace.stats.seg2['RECEIVER_LOCATION'].split(' ')) > 1:
                rx = float(trace.stats.seg2['RECEIVER_LOCATION'].split(' ')[0])
            else:
                rx = float(trace.stats.seg2['RECEIVER_LOCATION'])

            if len(trace.stats.seg2['SOURCE_LOCATION'].split(' ')) > 1:
                sx = float(trace.stats.seg2['SOURCE_LOCATION'].split(' ')[0])
            else:
                sx = float(trace.stats.seg2['SOURCE_LOCATION'])

            # header information
            starttime = st[0].stats.starttime
            ch = str(channel_nr + ti)  # set channel number
            dt = trace.stats.delta  # sampling interval in s
            npts = trace.stats.npts  # number of samples
            sr = trace.stats.sampling_rate
            delay = float(trace.stats.seg2.DELAY)  # pre trigger
            sn = str(trace.stats.seg2.SHOT_SEQUENCE_NUMBER)  # shot sequence number

            # new trace
            header = {
                'starttime': starttime,
                'channel': ch,
                'delta': dt,
                'npts': npts,
                'delay': delay,
                'sampling_rate': sr,
                'station': sn,
                'rx': rx,
                'sx': sx}
            trace_new = obspy.core.trace.Trace(data=trace.data, header=header)
            st_new.append(trace_new)

        self._st = st_new

    def _read_sgy_geom(self, fname, channel_nr):
        """
        Read stream data with .sg2 file format with geometry information in header

        Parameters
        ----------
        fname : str, file name
        channel_nr : int, optional
        """

        st_new = obspy.Stream()
        st = obspy.read(fname)

        for ti, trace in enumerate(st):

            scale = trace.stats.segy['trace_header']['scalar_to_be_applied_to_all_coordinates']
            sx_int = int(trace.stats.segy['trace_header']['source_coordinate_x'])
            sy_int = int(trace.stats.segy['trace_header']['source_coordinate_y'])
            #sz_int = int(trace.stats.segy['trace_header']['source_coordinate_z'])
            rx_int = int(trace.stats.segy['trace_header']['group_coordinate_x'])
            ry_int = int(trace.stats.segy['trace_header']['group_coordinate_y'])
            #rz_int = int(trace.stats.segy['trace_header']['group_coordinate_z'])

            if sx_int != 0:
                sx = sx_int / abs(scale) if scale < 0 else sx_int * scale
            elif sy_int != 0:
                sx = sy_int / abs(scale) if scale < 0 else sy_int * scale
            else:
                sx = 0

            if rx_int != 0:
                rx = sx_int / abs(scale) if scale < 0 else rx_int * scale
            elif ry_int != 0:
                rx = ry_int / abs(scale) if scale < 0 else ry_int * scale
            else:
                rx = 0

            sn = str(trace.stats["station"])

            starttime = st[0].stats.starttime
            dt = trace.stats.delta  # sampling interval in s
            npts = trace.stats.npts  # number of samples
            sr = trace.stats.sampling_rate
            delay = trace.stats.segy['trace_header']['delay_recording_time']  # set the pre-trigger value
            ch = str(channel_nr + ti)  # set channel number

            # new trace
            header = {
                'starttime': starttime,
                'channel': ch,
                'delta': dt,
                'npts': npts,
                'delay': delay,
                'sampling_rate': sr,
                'station': sn,
                'rx': rx,
                'sx': sx}
            trace_new = obspy.core.trace.Trace(data=trace.data, header=header)
            st_new.append(trace_new)

        self._st = st_new

    def _read_sg2(self, fname, channel_nr):
        """
        Read stream data with .sg2 file format

        Parameters
        ----------
        fname : str, file name
        channel_nr : int, optional
        """

        st_new = obspy.Stream()
        st = obspy.read(fname)

        for ti, trace in enumerate(st):

            # header information
            starttime = st[0].stats.starttime
            ch = str(channel_nr + ti)  # set channel number
            dt = trace.stats.delta  # sampling interval in s
            npts = trace.stats.npts  # number of samples
            sr = trace.stats.sampling_rate
            delay = float(trace.stats.seg2.DELAY)  # pre trigger
            # sn = str(trace.stats.seg2.SHOT_SEQUENCE_NUMBER)  # shot sequence number

            # new trace
            header = {
                'starttime': starttime,
                'channel': ch,
                'delta': dt,
                'npts': npts,
                'delay': delay,
                'sampling_rate': sr}

            trace_new = obspy.core.trace.Trace(data=trace.data, header=header)
            st_new.append(trace_new)

        self._st = st_new

    def _read_sgy(self, fname, channel_nr):
        """Read stream data with .sg2 file format

        Parameters
        ----------
        fname : str, file name
        channel_nr : int, optional
        """

        st_new = obspy.Stream()
        st = obspy.read(fname)

        for ti, trace in enumerate(st):

            starttime = st[0].stats.starttime
            dt = trace.stats.delta  # sampling interval in s
            npts = trace.stats.npts  # number of samples
            sr = trace.stats.sampling_rate
            delay = trace.stats.segy['trace_header']['delay_recording_time']  # set the pre-trigger value
            ch = str(channel_nr + ti)  # set channel number

            # new trace
            header = {
                'starttime': starttime,
                'channel': ch,
                'delta': dt,
                'npts': npts,
                'delay': delay,
                'sampling_rate': sr}
            trace_new = obspy.core.trace.Trace(data=trace.data, header=header)
            st_new.append(trace_new)

        self._st = st_new

    def _read_syn(self, fname, channel_nr, pre_trigger):
        """Read synthetic data in .mseed format

        Parameters
        ----------
        fname : str, file name
        channel_nr : int, optional
        pre_trigger : float, optional
        """

        st_new = obspy.Stream()
        st = obspy.read(fname)

        for ti, trace in enumerate(st):

            starttime = st[0].stats.starttime
            ch = str(channel_nr + ti)  # set channel number
            dt = trace.stats.delta  # sampling interval in s
            npts = trace.stats.npts  # number of samples
            sr = trace.stats.sampling_rate
            delay = pre_trigger  # set the pre-trigger value

            # new trace
            header = {
                #'starttime': starttime,
                'channel': ch,
                'delta': dt,
                'npts': npts,
                'delay': delay,
                'sampling_rate': sr}
            trace_new = obspy.core.trace.Trace(data=trace.data, header=header)
            st_new.append(trace_new)

        self._st = st_new

    def apply_geometry(self, source_coordinates, receiver_coordinates):
        """Apply the survey geometry to the seismic data"""

        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()

        # check the number of traces and receiver coordinates
        if len(receiver_coordinates) != len(st):
            raise ValueError(f'Number of receiver coordinates does not match number of traces. '
                             f'{len(receiver_coordinates)} != {len(st)}')

        for ti, trace in enumerate(st):
            # update channel based on receiver id
            st[ti].stats['channel'] = str(self.channel_nr + (receiver_coordinates.loc[ti, 'rin']-1))
            # add coordinate information
            st[ti].stats['rx'] = receiver_coordinates.loc[ti,'rx']
            st[ti].stats['sx'] = source_coordinates['sx'].item()

        if self._pst is None:
            self._st = st
        else:
            self._pst = st

        # set shot parameters
        self.set_shot_params()

    def _create_st(self, amps, par):
        """Create a new seismic stream"""

        st_new = obspy.Stream()
        for i,amp in enumerate(amps):

            # new trace
            header = {
                'delta': par.dt.item(),
                'npts': par.npts.item(),
                'delay': par.delay.item(),
                'sampling_rate': par.sampling_rate.item()}

            trace = obspy.core.trace.Trace(data=amp, header=header)
            st_new.append(trace)

        # update
        self._pst = st_new
        self.tapered_amps = par.tapered_amps.item()

    def update_pst(self,amps,sht,recs,par):
        """update stream data"""
        self._pst = self._st.copy()
        self._create_st(amps, par)
        self.apply_geometry(sht, recs)

    def reset_pst(self):
        self._pst = None

    ###################################################################################################################

    def _receiver(self, st=None):
        """return sensor positions"""

        if st is None:
            if self._pst is None:
                st = self._st.copy()
            else:
                st = self._pst.copy()
        else:
            st = st

        nchannels = len(st)
        gx = np.zeros(nchannels)
        for ti, trace in enumerate(st):
            gx[ti] = trace.stats.rx
        return gx

    def _source(self):
        """return source location (x-coordinate)"""
        return self._st[0].stats.sx

    def set_shot_params(self):
        """set shot parameters"""

        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()

        try:
            self.aqu_date = st.stats.seg2.ACQUISITION_DATE # acquisition time and date
        except AttributeError:
            self.aqu_date = None

        self.dt = st[0].stats.delta  # sampling interval in s
        self.npts = st[0].stats.npts  # number of samples
        self.delay = float(st[0].stats.delay)  # pre trigger
        self.sampling_rate = st[0].stats.sampling_rate # sampling rate
        #self.sn = int(self._st[0].stats.sn)  # shot sequence number

        self.receiver = self._receiver() # receiver positions
        self.source = self._source() # source location
        self.midpoint = self._midpoint(self.receiver) # receiver spread midpoint
        self.nchannels = self._nchannels() # number of receivers
        self.dx_list = self._dx(self.receiver) # receiver separation
        self.dx = np.median(abs(self.dx_list)) # median receiver separation

    def _init_params(self):
        """initialize some parameters"""

        self._st_shift = None
        self.fstep = 1
        #self.nstacks = 1
        self.tapered_amps = 0
        self.trafo_type = None
        self.extraction_method = None
        self.dispersive_energy = None
        self.velocity = None
        self.frequency = None
        self.wavenumber = None

        self.FK_data = {}

        self._pick = False
        self.picks = {}
        self.curve = None

    def _update_params_from_amps(self, amps):
        """update npts and nchannels"""
        nchannels,npts = amps.shape
        self.npts = npts
        self.nchannels = nchannels

    def _update_params_from_stream(self, st):
        """update shot parameters"""
        self.npts = len(st[0].data)
        self.nchannels = len(st)

        self.receiver = self._receiver()
        self.midpoint = self._midpoint(self.receiver)
        self.dx_list = self._dx(self.receiver)  # receiver separation
        self.dx = np.median(abs(self.dx_list))

    def _return_stream(self):
        """return current stream object"""
        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()
        return st

    def print_stats(self, which = 'stream'):
        """print stream information"""

        def pretty_print(header, data):
            row_format = "{:>15}" * (len(data) + 1)
            print(row_format.format("", *header))
            print(row_format.format("", *data))

        def dataList(st):
            receiver = self._receiver(st)  # receiver positions
            midpoint = self._midpoint(receiver)  # receiver spread midpoint
            dx_list = self._dx(receiver)  # receiver separation
            dx = np.median(abs(dx_list))  # median receiver separation
            data = [str(st[0].stats.delta),
                    str(st[0].stats.sampling_rate),
                    str(self.delay),
                    str(st[0].stats.npts),
                    str(len(receiver)),
                    str(dx),
                    str(midpoint)]
            return data

        columns = ['dt (s)', 'dt (Hz)', 'delay (s)', 'npts', 'nrec', 'dx', 'midpoint']

        if (which == 'stream') or (which == 'both'):
            print("#### RAW STREAM STATS ####")
            data = dataList(self._st)
            pretty_print(columns,data)

            if self._pst is not None:
                print("#### PROC STREAM STATS ####")
                data = dataList(self._pst)
                pretty_print(columns, data)

        if (which == 'trace') or (which == 'both'):
            if self._pst is None:
                for trace in self._st:
                    print("#### RAW DATA STATS ####")
                    print(trace.stats)
            else:
                for trace in self._pst:
                    print("#### PROC DATA STATS ####")
                    print(trace.stats)

    def save_stream(self, fot, pre):
        """save stream in .mseed file format"""

        safe_makedirs(fot)
        outfile = os.path.join(fot, f"Shot_{pre}.syn")

        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()

        for i, trace in enumerate(st):
            trace.stats["mseed"] = {'dataquality': 'D'}

        st.write(outfile, format='mseed')

    def _nchannels(self, st=None):
        """return number of channels"""

        if st is None:
            if self._pst is None:
                st = self._st.copy()
            else:
                st = self._pst.copy()
        else:
            st = st

        nreceiver = len(self.receiver)
        ntraces = len(st)

        if nreceiver == ntraces:
            return ntraces
        else:
            raise ValueError('Number of receiver and traces is not matching!')

    def _npts(self):
        """return number of samples"""
        return len(self._amps()[0])

    def _midpoint(self, receiver):
        """compute midpoint of a receiver spread"""
        return (np.max(receiver)+np.min(receiver))/2

    def _offsets(self, receiver, source):
        """compute shot receiver offsets"""
        return source - receiver

    def _aoffsets(self, receiver, source):
        """compute absolute shot receiver offsets"""
        return abs(self._offsets(receiver, source))

    @property
    def offset(self):
        return self._offsets(self.receiver,self.source)

    def _dx(self,receiver):
        """compute receiver separations"""
        return np.diff(receiver)

    def check_traces(self, st=None):
        """check whether a trace only contains zeros and remove it"""
        if st is None:

            if self._pst is None:
                st = self._st.copy()
            else:
                st = self._pst.copy()

        trace_list = []
        for ti, trace in enumerate(st):

            if np.sum(trace.data) == 0:
                trace_list.append(ti)

        self._remove_trace(trace_list)

    def st2amps(self, st = None):
        """store amplitudes in ndarray (nchannels,nsamples)"""
        return self._amps(st = st)

    def amps2st(self,amps,st = None):
        """store amplitudes (nchannels,nsamples) as obspy stream data"""
        self._amps2st(amps, st = st)

    def _amps(self, st=None):
        """store amplitudes in ndarray (nchannels,nsamples)"""

        if st is None:

            if self._pst is None:
                st = self._st.copy()
            else:
                st = self._pst.copy()

        amps = np.zeros((len(st), len(st[0].data)))
        for ti, trace in enumerate(st):
            amps[ti] = trace.data
        return amps

    def _amps2st(self,amps,st = None):
        """store amplitudes (nchannels,nsamples) as obspy stream data"""

        if st is None:
            if self._pst is None:
                st_proc = self._st.copy()
            else:
                st_proc = self._pst.copy()
        else:
            st_proc = st.copy()

        if len(amps) == len(st_proc):
            for i in range(len(st_proc)):

                st_proc[i].data = amps[i]
                st_proc[i].stats.npts = len(amps[i])

        self._pst = st_proc.copy()

    def resample(self, sampling_rate, window='hann', no_filter=True, strict_length=False):
        """resample data in all traces using the method from obspy method"""

        if self._pst is None:
            st_proc = self._st.copy()
        else:
            st_proc = self._pst.copy()

        st_proc.resample(sampling_rate, window, no_filter, strict_length)
        self.dt = 1/sampling_rate  # sampling interval in s
        self.sampling_rate = sampling_rate # sampling rate
        self._pst = st_proc # update stream

    def _detrend_signal(self, amps):
        """remove linear trend"""

        detrend_amps = np.zeros_like(amps)

        for i in range(amps.shape[0]):

            amp = amps[i]
            detrend_amps[i] = signal.detrend(amp)

        return detrend_amps

    def _normalize_amps(self, amps):
        """amplitude normalization"""
        norm_amps = np.zeros_like(amps)
        global_max = np.max(np.abs(amps))

        for i in range(norm_amps.shape[0]):

            amp = amps[i]
            if self.use_local_max:
                local_max = np.max(np.abs(amp))
                if local_max != 0:
                    amp = amp / np.max(np.abs(amp))
            else:
                amp = amp/global_max
            norm_amps[i] = amp

        return norm_amps

    def _apply_taper(self, st = None, inplace = True):
        """apply taper"""

        if st is None:

            if self._pst is None:
                st = self._st.copy()
            else:
                st = self._pst.copy()

        amps = self._amps(st=st)
        tapered_amps = self._taper_amps(amps)

        if inplace:
            self._amps2st(tapered_amps)
        else:
            return tapered_amps


    def _taper_amps(self, amps):
        """apply taper window"""

        taper_amps = np.zeros_like(amps)
        receiver = self.receiver

        # tapering function
        taper_func = getattr(signal.windows, 'hamming')
        taper_win = taper_func(len(receiver))

        for i in range(len(amps[0])):
            amp = amps[:, i]
            taper_amps[:, i] = amp * taper_win

        return taper_amps

    def _flip(self):
        """reverse order of traces"""
        if self._pst is None:
            st_proc = self._st.copy()
        else:
            st_proc = self._pst.copy()

        st_proc.reverse()

        self._pst = st_proc

    def trim(self, by, **kwargs):
        """non interactive time or offset based trimming of the trace data"""

        if by == 'time':
            min = kwargs.setdefault('min', 0)
            max = kwargs.setdefault('max', 0)
            self._trim_times_obspy(min,max)
        elif by == 'time_range':
            min = kwargs.setdefault('min', 0)
            max = kwargs.setdefault('max', 0)
            self._trim_times(min,max)
        elif by == 'offset':
            min = kwargs.setdefault('min', -1e10)
            max = kwargs.setdefault('max', 1e10)
            which = kwargs.setdefault('which', 'forward')
            self._trim_by_offset_both(min,max,which)
        elif by == 'separation':
            sep = kwargs.setdefault('dx', self.dx)
            self._trim_by_receiver_separation(sep)
        elif by == 'window':
            xmid = kwargs.setdefault('xmid', self.midpoint)
            nwin = kwargs.setdefault('wlen', len(self.receiver))
            self._trim_by_trace_window(nwin, xmid)
        elif by == 'select':
            trace_indices = kwargs.setdefault('trace_ids', np.arange(len(self.receiver)))
            self._select_traces(trace_indices)
        elif by == 'remove':
            trace_indices = kwargs.setdefault('trace_ids', [])
            self._remove_trace(trace_indices)
        else:
            print(f'Preprocessing function "{by}" not implemented.')

    def _trim_times_obspy(self, start_cut_off, end_cut_off):
        """cut times of stream by providing the amount of time that should be cut-off
           at the beginning and end of the stream"""

        if self._pst is None:
            st_proc = self._st.copy()
        else:
            st_proc = self._pst.copy()

        if isinstance(start_cut_off, numbers.Number) and isinstance(end_cut_off, numbers.Number):
            for t, trace in enumerate(st_proc.traces):
                trace.trim(start_cut_off, end_cut_off)

            self._pst = st_proc
            self._update_params_from_stream(st_proc)
        else:
            warn_msg = f'Start and end time must be numeric not {type(start_cut_off), type(end_cut_off)}. No process applied.'
            self.logger.warning(warn_msg)

    def _trim_times(self, start_time, end_time):
        """cut times of stream by providing the start and end time of the stream"""

        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()

        amps = self._amps(st=st)
        nchannels, npts = amps.shape

        t = np.arange(npts) * st[0].stats.delta  # time s

        start_cut_off = start_time
        end_cut_off = np.max(t) - end_time

        if isinstance(start_cut_off, numbers.Number) and isinstance(end_cut_off, numbers.Number):
            for t, trace in enumerate(st.traces):
                trace.trim(start_cut_off, end_cut_off)

            self._pst = st
            self._update_params_from_stream(st)
        else:
            warn_msg = f'Start and end time must be numeric not {type(start_cut_off), type(end_cut_off)}. No process applied.'
            self.logger.warning(warn_msg)

    def trim_by_offset(self, min_offset, max_offset, nrec = None):
        """cut traces outside of the offsets limits"""

        if self._pst is None:
            st_proc = self._st.copy()
        else:
            st_proc = self._pst.copy()

        if isinstance(min_offset, numbers.Number) and isinstance(max_offset, numbers.Number):
            receiver = self.receiver
            source = self.source
            offset = receiver - source

            oids = np.argwhere((offset >= min_offset) & (offset <= max_offset))[:, 0]

            if (nrec is not None) and (isinstance(nrec, numbers.Number)):

                if (min_offset < 0) and (max_offset < 0):
                    oids = oids[-nrec:]
                else:
                    oids = oids[:nrec]

            if len(oids) > 0:
                st_new = obspy.Stream()
                for i in oids:
                    st_new.append(st_proc[i])

                # update stream
                self._pst = st_new
                self._update_params_from_stream(st_new)

            return oids
        else:
            warn_msg = (f'Min and max offset must be numeric not {type(min_offset), type(max_offset)}. '
                        f'No process applied.')
            self.logger.warning(warn_msg)
            return []

    def _trim_by_offset_both(self, min_offset, max_offset, which = 'forward'):
        """cut traces outside of the offset limits considering forward and reverse offset shot"""

        if self._pst is None:
            st_proc = self._st.copy()
        else:
            st_proc = self._pst.copy()

        if isinstance(min_offset, numbers.Number) and isinstance(max_offset, numbers.Number):
            receiver = self.receiver
            source = self.source
            offset = receiver - source

            # sort min max
            mmo = sorted([np.abs(min_offset), np.abs(max_offset)])

            # offset ids associated to forward shot
            oids_fw = np.argwhere((offset >= mmo[0]) & (offset <= mmo[1]))[:, 0]

            # offset ids associated to reverse shot
            oids_rv = np.argwhere((offset >= -mmo[1]) & (offset <= -mmo[0]))[:, 0]

            if (len(oids_fw) > 0) & (len(oids_rv) > 0):
                oids = oids_fw if which == 'forward' else oids_rv
            elif len(oids_fw) > 0:
                oids = oids_fw
            elif len(oids_rv) > 0:
                oids = oids_rv
            else:
                warn_msg = 'Min and max offset out of bounds. No process applied to stream.'
                self.logger.warning(warn_msg)
                return [],[]

            st_new = obspy.Stream()
            for i in oids:
                st_new.append(st_proc[i])

            self._pst = st_new
            self._update_params_from_stream(st_new)
            return oids_fw, oids_rv

        else:
            warn_msg = (f'Min and max offset must be numeric not {type(min_offset), type(max_offset)}. '
                        f'No process applied.')
            self.logger.warning(warn_msg)
            return [], []

    def _trim_by_receiver_separation(self,sep):
        """select channels with specified geophone separation"""

        if self._pst is None:
            st_proc = self._st.copy()
        else:
            st_proc = self._pst.copy()

        if isinstance(sep, numbers.Number):

            receiver = self.receiver
            unique_sep = np.unique(self.dx_list)

            # in case of equidistant geophone separation
            if len(unique_sep) == 1:
                if sep % self.dx == 0:
                    step = int(sep/self.dx)
                else:
                    step = round(sep / self.dx)

                st_new = obspy.Stream()
                for trace in st_proc[::step]:
                    st_new.append(trace)

                # update
                self._pst = st_new
                self._update_params_from_stream(st_new)
        else:
            warn_msg = f'Separation must be numeric not {type(sep)}. No process applied.'
            self.logger.warning(warn_msg)

    def _trim_by_trace_window(self, nwin, xmid):
        """select traces around xmid and window size"""

        nchannels = self.nchannels
        nwin = int(nwin)

        if nwin <= nchannels:
            if (nwin % 2) == 0:
                if xmid in self._receiver():
                    trace_select = range(int((nchannels - nwin) / 2), int((nchannels + nwin) / 2 + 1))
                else:
                    trace_select = range(int((nchannels - nwin) / 2), int((nchannels + nwin) / 2))
            else:
                if xmid in self._receiver():
                    trace_select = range(int((nchannels - nwin - 1) / 2), int((nchannels + nwin - 1) / 2) + 2)
                else:
                    trace_select = range(int((nchannels - nwin - 1) / 2), int((nchannels + nwin - 1) / 2) + 1)

            self._select_traces(trace_select)

    def _select_traces(self, trace_indices):
        """select a subset of a stream based on trace indices"""

        if not isinstance(trace_indices, Iterable):
            if isinstance(trace_indices, int):
                trace_indices = [trace_indices]
            else:
                raise ValueError(f'Indices should be int, list, or array-like, not {type(trace_indices)}')

        receiver = self.receiver

        if self._pst is None:
            st_proc = self._st.copy()
        else:
            st_proc = self._pst.copy()

        st_new = obspy.Stream()
        receiver_subset = []
        for i, trace in enumerate(st_proc):
            if i in trace_indices:
                st_new.append(trace)
                receiver_subset.append(receiver[i])

        # update
        self._pst = st_new
        self._update_params_from_stream(st_new)

    def _remove_trace(self, trace_indices):
        """remove a trace based on an index"""

        if not isinstance(trace_indices, Iterable):
            if isinstance(trace_indices, int):
                trace_indices = [trace_indices]
            else:
                raise ValueError(f'Indices should be int, list, or array-like, not {type(trace_indices)}')

        if self._pst is None:
            st_proc = self._st.copy()
        else:
            st_proc = self._pst.copy()

        st_new = obspy.Stream()
        for i, trace in enumerate(st_proc):
            if i not in trace_indices:
                st_new.append(trace)

        # update
        self._pst = st_new
        self._update_params_from_stream(st_new)

    def filter(self, by, **kwargs):
        """non-interactive FD based filtering of the trace data"""

        if by == 'frequency':
            type = kwargs.setdefault('filter_type', 'bandpass')
            min = kwargs.setdefault('min', -np.inf)
            max = kwargs.setdefault('max', np.inf)
            self._filter_freq(type,min,max)
        elif by == 'FK':
            self._fk_filter_from_file(**kwargs)
        elif by == 'lmo':
            vel = kwargs.setdefault('velocity', None)
            bulk_shift = kwargs.setdefault('bulk_shift', 0)
            self._lmo(vel, bulk_shift)
        elif by == 'mute':
            trace_indices = kwargs.setdefault('trace_ids', [])
            self._mute_traces(trace_indices)
        elif by == 'resample':
            self.resample(**kwargs)
        elif by == 'reverse_polarity':
            trace_indices = kwargs.setdefault('trace_ids', [])
            self._reverse_polarity(trace_indices)
        elif by == 'taper':
            self._apply_taper()
        else:
            print(f'Preprocessing function "{by}" not implemented.')

    def _filter_freq(self, type, freqmin=None,freqmax=None):
        """apply frequency filtering"""

        if self._pst is None:
            st_proc = self._st.copy()
        else:
            st_proc = self._pst.copy()

        if type == 'bandpass':
            if (freqmin is not None) and (freqmax is not None):
                st_proc.filter(type=type,
                               freqmin=freqmin,
                               freqmax=freqmax,
                                corners = 2,
                                zerophase = True)


        elif type in ['lowpass','highpass']:
            if freqmin is not None:
                freq = freqmin
            elif freqmax is not None:
                freq = freqmax
            else:
                raise ValueError

            st_proc.filter(type=type,
                           freq=freq,
                            corners = 2,
                            zerophase = True)

        self._pst = st_proc

    def _reverse_polarity(self, trace_indices):
        """reverse polarity of selected traces"""

        if not isinstance(trace_indices, Iterable):
            if isinstance(trace_indices, int):
                trace_indices = [trace_indices]
            else:
                raise ValueError(f'Indices should be int, list, or array-like, not {type(trace_indices)}')

        if self._pst is None:
            st_proc = self._st.copy()
        else:
            st_proc = self._pst.copy()

        for i, trace in enumerate(st_proc):
            if i in trace_indices:
                st_proc[i].data *= -1

        # update
        self._pst = st_proc

    def _mute_traces(self, trace_indices):
        """set a trace to 0"""

        if not isinstance(trace_indices, Iterable):
            if isinstance(trace_indices, int):
                trace_indices = [trace_indices]
            else:
                raise ValueError(f'Indices should be int, list, or array-like, not {type(trace_indices)}')

        if self._pst is None:
            st_proc = self._st.copy()
        else:
            st_proc = self._pst.copy()

        for i, trace in enumerate(st_proc):
            if i in trace_indices:
                st_proc[i].data *= 0

        # update
        self._pst = st_proc

    def _linear_mute(self, x_data, y_data,key='t', **kwargs):
        """
        linear mute

        Parameters
        ----------
        x_data : list, point list containing x data
        x_data : list, point list containing y data
        key :  str, which type of mute to apply ('t'-top, 'b'-bottom)
        kwargs :
        """

        # tapering type and settings
        taper_type = kwargs.pop('taper_type', 'tukey')
        taper_func = getattr(signal.windows, taper_type)

        kwargs_taper = {}
        if taper_type == 'tukey':
            # shape of the tukey window
            taper = kwargs.pop('tapering', 'mild')
            if taper == 'mild':
                taper_alpha = 0.3
            elif taper == 'strong':
                taper_alpha = 0.5
            else:
                taper_alpha = 0.2
            kwargs_taper['alpha'] = taper_alpha

        # data
        if self._pst is None:
            st_proc = self._st.copy()
        else:
            st_proc = self._pst.copy()

        receiver = np.array(self.receiver)
        dt = self.dt  # sampling rate in sec
        delay = self.delay  # pre trigger in sec
        ndelay = int(abs(delay / dt))
        npts = len(st_proc[0].data)

        (idx1, idx2) = x_data
        (t1, t2) = y_data

        x1 = receiver[int(idx1)]
        x2 = receiver[int(idx2)]

        idx_between = np.where((receiver >= x1) & (receiver <= x2))[0]
        slope = (t2 - t1) / (x2 - x1)
        times = t1 + slope * (receiver[idx_between] - x1)

        # top mute boundary (straight line)
        top_idx = np.zeros_like(receiver, dtype=int)
        if key == 't':
            top_idx_between = np.array((times / dt) + ndelay, dtype=int)
            top_idx[idx_between] = top_idx_between

        # bottom mute boundary (straight line)
        bottom_idx = np.ones_like(receiver, dtype=int) * (npts - 1)
        if key == 'b':
            bottom_idx_between = np.array((times / dt) + ndelay, dtype=int)
            bottom_idx[idx_between] = bottom_idx_between

        # Perform windowing
        window = np.zeros(npts)
        for i, (start, stop) in enumerate(zip(top_idx, bottom_idx)):
            taper_win = taper_func(stop - start, **kwargs_taper)
            window[start:stop] = taper_win
            st_proc[i].data *= window
            window *= 0

        self._pst = st_proc

    def linear_mute(self, points,key='t', **kwargs):

        # extract points from points_list
        if len(points) > 0:
            x, y = zip(*sorted(points.items()))
            self._linear_mute(x,y, key, **kwargs)

    def apply_linear_mute(self, points_list, key_list,**kwargs):

        for points, key in zip(points_list, key_list):
            self.linear_mute(points,key, **kwargs)

    def _reset_mute(self):
        self._pst = self._st.copy()

    def _reset_mute_last(self):

        if self._pst is None:
            self._st = self._st_backup_last_mute
        else:
            self._pst = self._st_backup_last_mute

    def _zero_padding(self,amps):
        """append zeros to amplitudes to achieve a desired resolution"""

        df = self.df
        dt = self.dt

        _,npts = amps.shape
        new_npts = int(round(1 / (df * dt)))

        if new_npts > npts:
            padding = new_npts - npts
            npts = new_npts
        else:
            padding = 0 # no padding

        zero_padded_amps = np.pad(amps, [(0, 0), (padding, padding)], mode='constant', constant_values=0)

        return zero_padded_amps,padding,npts

    def argfreq(self):
        """return the frequency ids of the selection"""

        dt = self.dt

        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()

        amps = self._amps(st=st)

        if self.pad:
            amps, _, _ = self._zero_padding(amps)

        nchannels, npts = amps.shape

        # frequency range
        omega_fs = 2 * np.pi * 1 / dt
        omega = np.arange(npts) * (omega_fs / npts)
        freq = omega / (2 * np.pi)

        min_id = np.argmin(np.abs(freq - self.fmin))
        max_id = np.argmin(np.abs(freq - self.fmax))
        fids = np.arange(min_id, max_id + 1, step=self.fstep)

        return fids, freq

    def _lmo(self, vel = None, bulk_shift=0):
        """
        Linear move-out

        Parameters
        ----------
        vel : float, velocity for the LMO in m/s
        bulk_shift : float, optional, static shift in s applied to the time
        """

        receiver = self.receiver
        source = self.source

        # estimate the velocity from the data if it is not provided
        if vel is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            self._plotSeismogram(axes=ax, amp_scale=1)

            text = '\n'.join((
                r'$\bf{Keyboard \quad commands:}$',
                r'Press $\bf{v}$ to estimate the velocity.',
                r'Press $\bf{e}$ to stop the process.',))
            at = AnchoredText(text,
                              loc='lower right', prop=dict(size=6), frameon=True, bbox_to_anchor=(1., 1.15),
                              bbox_transform=ax.transAxes
                              )
            ax.add_artist(at)

            plot = TXInteractive(ax, receiver=receiver)
            ax.set_title(f'Velocity estimation', fontweight='bold')
            plt.tight_layout()
            plt.show()

            if plot.slope is not None:
                vel = 1 / plot.slope

        if vel is not None:

            if self._pst is None:
                st = self._st.copy()
            else:
                st = self._pst.copy()

            amps = self._amps(st=st)

            # sample size
            iX, iT = amps.shape

            # source-receiver offsets
            offsets = self._aoffsets(receiver, source)

            t = np.arange(iT) * self.dt

            amps_lmo = np.zeros_like(amps)
            for i in range(len(amps)):
                tlmo = offsets[i] / vel
                amps_lmo[i] = np.interp(t - bulk_shift, t - tlmo, amps[i])

            self._amps2st(amps=amps_lmo)

    # def _dlmo(self):
    #     """
    #     dynamic linear moveout correction by Park et al. (1998)
    #     """

    def _add_fk_data_to_dict(self,FK_data, theta, kw , freq, iX, iT):
        self.FK_data.update({'FK_abs': FK_data, 'theta': theta, 'kw': kw, 'freq': freq, 'iX': iX, 'iT': iT})

    # %% wavefield transformation
    def _inverse_fk_transform(self,FK_unwrap,iT,iX):
        """Transformation to x-t domain"""

        iF = nextpow2(iT)[1]
        FK = np.zeros((iF,iF)).astype("complex")

        pos_freq = FK.shape[0] // 2
        FK_unwrap = np.flipud(FK_unwrap)

        # exploit symmetry
        # top half
        FK[:pos_freq+1] = FK_unwrap
        # bottom half
        FK[pos_freq+1:] = np.conj(np.flipud(np.fliplr(FK_unwrap[1:-1])))

        # back transformation
        FK = np.fft.ifftshift(FK)
        amps_padded = np.fft.ifft2(FK,s=(iF,iF)).real
        amps = np.transpose(amps_padded)[:iX,:iT]

        # source-receiver offsets
        receiver = self.receiver
        source = self.source
        if source > receiver[-1]:
            amps = np.flipud(amps)

        return amps

    def _fk_transform(self):
        """Transformation to F-K domain"""

        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()

        # amplitude data & processing
        if self.tapered_amps == 0:
            amps = self._apply_taper(st = st, inplace = False)
        else:
            amps = self._amps(st=st)

        # source-receiver offsets
        receiver = self.receiver
        source = self.source
        if source > receiver[-1]:
            amps = np.flipud(amps)

        # sample spacing
        dx = self.dx
        dt = self.dt

        # sample size
        iX,iT = amps.shape
        iF = nextpow2(iT)[1]

        # FK transformation
        amps = np.transpose(amps)
        FK = np.fft.fft2(amps,s=(iF,iF))

        # shift to 0|0
        FK = np.fft.fftshift(FK)

        # select only positive frequencies (upper half)
        pos_freq = FK.shape[0] // 2
        FK_unwrap = FK[:pos_freq+1]
        FK_unwrap = np.flipud(FK_unwrap)

        # wavenumber and frequencies
        k = np.fft.fftfreq(iF, dx)
        kw = k*2*np.pi
        kw = np.fft.fftshift(kw)

        fpos = np.linspace(0, 0.5/dt, iF // 2+1)

        theta = np.angle(FK_unwrap)
        FK_abs = abs(FK_unwrap)

        self._add_fk_data_to_dict(FK_abs, theta, kw, fpos, iX, iT)

        return FK_abs, theta, kw, fpos,iX,iT

    def apply_fk_filter(self, points_list, key_list, **kwargs):
        """apply fk filter from picking boundaries"""

        FK_abs, theta, kw, freq, iX, iT = self._fk_transform(**kwargs)

        df = freq[1] - freq[0]

        k_grid, f_grid = np.meshgrid(kw, freq)
        mask = np.ones_like(FK_abs, dtype=float)

        # Interpolate user-defined fk boundary
        for points, key in zip(points_list, key_list):
            if points:
                x, y = zip(*sorted(points.items()))
                k_boundary = np.array(x)
                f_boundary = np.array(y)
                f_interp = interpolate.interp1d(f_boundary, k_boundary, bounds_error=False, fill_value="extrapolate")
                k_limit = f_interp(np.abs(f_grid))  # apply absolute to support symmetry

                if key == 'b':
                    mask[np.abs(k_grid) >= k_limit] = 0
                elif key == 't':
                    mask[np.abs(k_grid) <= k_limit] = 0

                taper_len = int(kwargs.pop('taper_length', 5) / (df))
                taper = signal.windows.hann(2 * taper_len)
                for j in range(mask.shape[0]):
                    k_cut = k_limit[j]
                    idx = np.where(np.abs(kw) >= k_cut)[0]
                    if len(idx) > 0:
                        start = max(idx[0] - taper_len, 0)
                        end = min(idx[0] + taper_len, len(kw))
                        mask[j, start:end] *= taper[:end - start]

        FK_abs_filt = FK_abs * mask

        # back transformation
        FK_filt = FK_abs_filt * np.exp(1j * theta)
        FK_filt[:, :FK_abs_filt.shape[1]//2] = 0

        amps = self._inverse_fk_transform(FK_filt, iT, iX)
        self._amps2st(amps)

        self._add_fk_data_to_dict(FK_filt, theta, kw, freq, iX, iT)

    def reset_FK(self):
        """Reset FK filter"""
        self._pst = self._st.copy()

    def _fk_filter_from_file(self, fname=None, **kwargs):
        """apply the fk filter based on a file containing the bounds"""

        if os.path.isfile(fname):
            points_top, points_bot = read_filter(fname)

            for points, key in zip([points_top,points_bot],['t','b']):
                self.apply_fk_filter(points,key,**kwargs)
        else:
            self.logger.error(f"File {fname} not found. No FK filter applied to data")

    def transform(self, method = 'phaseshift', **kwargs):
        """Apply the wave field transformation"""

        keys = ['fdbf', 'phaseshift']
        self.trafo_type = method

        # apply the transformation based on the chosen method
        if method.lower() == 'fdbf':
            self._fdbf()
        elif method.lower() == 'phaseshift':
            self._phaseshift()
        elif method.lower() == 'radon':
            self._inverse_radon()
        else:
            raise NotImplementedError(f'Transformation "{method}" not supported. '
                                      f'Use one of the keys: {keys}')

    def update_FV(self, method, vel, kw, freq, FV):
        """update FV data"""

        self.trafo_type = method
        self.dispersive_energy = FV
        self.velocity = vel
        self.wavenumber = kw
        self.frequency = freq

    def _steering_vector(self,kx,steering = 'cylindrical',sign = -1):
        """set of phase delays

        References
        ----------
        Zywicki, D.J., 1999. Advanced signal processing methods applied
        to engineering analysis of seismic surface289
        waves, Ph.D. dissertation, Georgia Institute of Technology.
        """
        if steering == 'cylindrical':
            # cylindrical steering vector exp(-1j[phi(H0(kx))])
            # H0 ... Hankel function Hn = Jn + i Yn
            return np.exp(-1j * np.angle(special.j0(kx) + 1j*special.y0(kx)))
        else:
            # plane steering vector
            return np.exp(1j * kx * sign)

    def _phaseshift(self):
        """Phase-shift transformation method after Park et al.(1998)."""

        self.dispersive_energy = None
        self.velocity = None
        self.frequency = None

        dt = self.dt
        receiver = self.receiver
        source = self.source

        # amplitude data & processing
        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()

        amps = self._amps(st=st)

        if self.pad:
            amps,_,_ = self._zero_padding(amps)
        if self.norm_amps:
            amps = self._normalize_amps(amps)

        # sample size
        iX, iT = amps.shape

        # source-receiver offsets
        offsets = self._aoffsets(receiver, source)
        if source > receiver[-1]:
            offsets = np.flipud(offsets)
            amps = np.flipud(amps)

        # frequency range
        omega_fs = 2 * np.pi * 1/dt
        omega = np.arange(iT) * (omega_fs/iT)
        freq = omega/(2*np.pi)

        min_id = np.argmin(np.abs(freq - self.fmin))
        max_id = np.argmin(np.abs(freq - self.fmax))
        fids = np.arange(min_id, max_id + 1, step=self.fstep)
        freq = freq[fids]

        # testing phase velocities
        vels = np.arange(self.vmin, self.vmax, self.vstep)

        # transformation to U(x,w) domain
        u = np.fft.fft(amps)

        # transformation to V(f,v) domain
        ks = np.zeros_like(vels)
        V = np.zeros((len(vels), len(freq))).astype(complex)
        for j, (f_index, f) in enumerate(zip(fids, freq)):
            for i, vel in enumerate(vels):
                ktrial = wavenumber(f, vel)
                kx = ktrial*offsets
                steer = self._steering_vector(kx,steering = 'plane', sign=1)
                V[i, j] = np.abs(np.sum(steer * u[:, f_index]/np.abs(u[:, f_index])))
                ks[j] = ktrial

        V = V / len(offsets)  # normalization by trace number (suggested by Olafsdottir,2018)

        self.frequency = freq
        self.dispersive_energy = V
        self.velocity = vels
        self.wavenumber = ks

    def _rfest(self, u):
        """spatiospectral correlation matrix

        References
        ----------
        Zywicki, D.J., 1999. Advanced signal processing methods applied
        to engineering analysis of seismic surface289
        waves, Ph.D. dissertation, Georgia Institute of Technology.
        """

        # unpack dimensions
        iX, iF, nblocks = u.shape

        # weights: shape (iX, iF)
        weights = 1 / np.abs(np.mean(u, axis=-1))

        # broadcast weights across blocks
        u_weighted = u * weights[..., None]

        return np.einsum("ifk,jfk->ijf", u_weighted, u_weighted.conj()) / nblocks

    def _fdbf(self, steering = 'cylindrical'):
        """frequency-domain beamforming (Zywicki, 1999)

        References
        ----------
        Zywicki, D.J., 1999. Advanced signal processing methods applied
        to engineering analysis of seismic surface289
        waves, Ph.D. dissertation, Georgia Institute of Technology.
        """

        self.dispersive_energy = None
        self.velocity = None
        self.frequency = None

        dt = self.dt
        receiver = self.receiver
        source = self.source

        # receiver-source offsets; time series matrix
        offsets = self._aoffsets(receiver, source)

        # amplitude data & processing
        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()

        amps = self._amps(st=st)

        if self.pad:
            amps,_,_ = self._zero_padding(amps)
        if self.norm_amps:
            amps = self._normalize_amps(amps)

        # sample size
        iX, iT = amps.shape

        if source > receiver[-1]:
            offsets = np.flipud(offsets)
            amps = np.flipud(amps)

        # frequency range
        omega_fs = 2 * np.pi * 1/dt
        omega = np.arange(iT) * (omega_fs/iT)
        freq = omega/(2*np.pi)

        min_id = np.argmin(np.abs(freq - self.fmin))
        max_id = np.argmin(np.abs(freq - self.fmax))
        fids = np.arange(min_id, max_id + 1, step=self.fstep)
        freq = freq[fids]

        # testing phase velocity
        vels = np.arange(self.vmin, self.vmax, self.vstep)

        # transformation to U(x,w) domain
        amps = amps.reshape(iX, iT, 1)
        u = np.fft.fft(amps, axis=1)

        # Calculate the spatiospectral correlation matrix
        sscm = self._rfest(u)
        sscm = sscm[:,:,fids]

        # weighting
        offsets_n = offsets[:, None]
        w = offsets_n @ offsets_n.T

        ks = np.zeros_like(vels)
        V = np.zeros((len(vels), len(freq))).astype(complex)
        for i, f in enumerate(freq):
            R_f = sscm[:, :, i]*w
            for j, vel in enumerate(vels):
                ktrial = wavenumber(f,vel)
                kx = ktrial*offsets
                steer = self._steering_vector(kx,steering)
                V[j, i] = (steer.conjugate().T @ R_f @ steer)
                ks[j] = ktrial

        V = V / len(offsets)  # normalization by trace number (suggested by Olafsdottir,2018)

        self.frequency = freq
        self.dispersive_energy = V
        self.velocity = vels
        self.wavenumber = ks


    def _MOPA(self,std=None,rel_err=None,abs_err=None,stopAtChi2=2,**kwargs):
        """Multi-offset phase analysis (MOPA; Strobbia & Foti, 2014)."""

        dt = self.dt
        receiver = self.receiver
        source = self.source

        # receiver-source offsets and amplitudes
        offsets = self._aoffsets(receiver,source)
        min_nrec = kwargs.pop('min_nrec',12)

        if std is None and rel_err is None and abs_err is None:
            weighted = False
        else:
            weighted = True

        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()

        amps = self._amps(st=st)

        if self.pad:
            amps,_,_ = self._zero_padding(amps)

        if source > receiver[-1]:
            offsets = np.flipud(offsets)
            amps = np.flipud(amps)

        iX, iT = amps.shape

        # frequency range
        omega_fs = 2 * np.pi * 1/dt
        omega = np.arange(iT) * (omega_fs/iT)
        freq = omega/(2*np.pi)

        min_id = np.argmin(np.abs(freq - self.fmin))
        max_id = np.argmin(np.abs(freq - self.fmax))
        fids = np.arange(min_id, max_id + 1, step=self.fstep)
        freq = freq[fids]

        # FFT
        u = np.fft.fft(amps, axis=1)

        results_dict_plotting = {}
        picked_freqs = []
        picked_vels = []
        picked_chi2 = []

        # MOPA Loop
        for idx_out, f_index in enumerate(fids):

            chi2 = np.inf
            off0 = 0  # index of first usable receiver
            N = len(offsets)  # number of stations

            # Unwrap phase
            ph_raw = np.angle(u[:, f_index])
            ph = np.unwrap(ph_raw)

            u_fft = u[:, f_index]
            mag_all = np.abs(u_fft)

            # χ²-based offset removal
            while chi2 >= stopAtChi2 and (N - off0) >= min_nrec:

                offs = offsets[off0:]
                phi = ph[off0:]
                mag = mag_all[off0:]

                # Weights
                if weighted:
                    if std is not None:
                        weights = 1.0 / (std ** 2)
                    else:
                        # relative error or absolute error
                        if abs_err is None:
                            phase_error = rel_err * np.abs(phi)
                        else:
                            phase_error = np.full_like(phi, abs_err)

                        weights = 1.0 / phase_error
                else:
                    weights = np.ones_like(phi)

                k0, phi0 = linear_LSQR(offs, phi, weights)

                # Model response
                phi_pred = phase_response(offs, k0, phi0)  # vector

                _, _, chi2 = compute_chi2(phi, phi_pred, weights)

                # Try removing another near-offset receiver next iteration
                off0 += 1

            # Store result if converged
            if chi2 <= stopAtChi2:
                f_val = freq[idx_out]
                vel_val = 2 * np.pi * f_val / k0

                picked_freqs.append(f_val)
                picked_vels.append(vel_val)
                picked_chi2.append(chi2)

            # Plotting dictionary
            results_dict_plotting[idx_out] = {
                "offsets": offs,
                "mag": mag,
                "phase": ph_raw[off0 - 1:],  # raw phase for plotting only
                "k0": k0,
                "phi0": phi0,
                "frequency": freq[idx_out],
                "velocity": 2 * np.pi * freq[idx_out] / k0,
                "chi2": chi2,
            }

        # Final results: dispersion curve
        if len(picked_vels) > 0:

            self._pick = True
            self.picks['auto'] = {
                0: {
                    "frequency": np.array(picked_freqs),
                    "velocity": np.array(picked_vels),
                    "chi2": np.array(picked_chi2),
                }
            }

            if kwargs.setdefault("showResults", False):
                ax = self._plotMOPA(results_dict_plotting, **kwargs)
                plt.show()

        else:
            self.logger.warning("No dispersion curve extracted.")

    def compute_phasediffs(self):
        """compute phase differences between adjacent receivers for one shot file"""

        receiver = self.receiver
        source = self.source

        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()

        amps = self._amps(st=st)

        if source > receiver[-1]:
            amps = np.flipud(amps)

        # frequency range
        fids,freqs = self.argfreq()

        # fft
        u = np.fft.fft(amps)

        # extract phase & compute phase differences
        phase_diff = np.zeros((len(fids),len(receiver)-1))
        for j, f_index in enumerate(fids):

            phase = np.angle(u[:, f_index])
            phase_unwrapped = np.unwrap(phase, axis=0)
            if source > receiver[-1]:
                phase_unwrapped = np.flipud(phase_unwrapped)
            di = np.diff(phase_unwrapped)  # phase difference between adjacent geophones

            phase_diff[j,:] = di

        return phase_diff, fids, freqs[fids]

    def dcpicking(self, pck_mode = 'auto', auto_method = 'max', **kwargs):
        """automatic or interactive dispersion curve picking"""

        if pck_mode == 'auto':
            # Automatic dispersion curve extraction

            if auto_method == 'max':
                # locate amplitude maxima in dispersion image at each frequency, i.e. apparent dispersion curve
                self.extraction_method = f'max'

                if self.dispersive_energy is None:
                    self.transform()

                peaks_idx = np.argmax(self.dispersive_energy, axis=0)

                freq_pick = self.frequency
                vel_pick = self.velocity[peaks_idx]

                self.picks[pck_mode] = {0:{'frequency': freq_pick, 'velocity': vel_pick}}

            elif auto_method == 'MOPA':
                # Apply MOPA to extract the dispersion curves
                self.extraction_method = 'MOPA'
                self._MOPA(**kwargs)
            else:
                raise NotImplementedError
        else:
            raise ValueError(f'Picking mode not recognized. Use key: "auto" \n '
                             f'for automatic dispersion curve extraction. \n'
                             f'Use the gui for manual dispersion curve picking.')

    # %% WINDOWING
    def moving_window(self, **kwargs):
        """Move a window along the trace data to select a subset and store it in a dictionary"""

        all_receiver = self.receiver
        source = self.source

        # window settings
        # minimum offset (m)
        if 'minoffset' in kwargs:
            minoffset = kwargs['minoffset']
        else:
            minoffset = self.wmino

        # maximum offset (m)
        if 'maxoffset' in kwargs:
            maxoffset = kwargs['maxoffset']
        else:
            maxoffset = self.wmaxo

        # move increment (number of traces to skip between windows)
        if 'wmove' in kwargs:
            win_move = kwargs['wmove']
        else:
            win_move = self.wmove

        mintrace = 0

        # window length (number of traces)
        if 'wlen' in kwargs:
            maxtrace = kwargs['wlen']
        else:
            maxtrace = self.wlen

        win_id = 0

        windows = {}

        while maxtrace <= len(all_receiver):

            trace_select = range(mintrace, maxtrace)
            temp_stream = copy.deepcopy(self)
            temp_stream._select_traces(trace_indices=trace_select)

            receiver = temp_stream.receiver
            offsets = temp_stream._aoffsets(receiver, source)

            shot_condition = ((np.min(offsets) <= maxoffset) &
                              (np.min(offsets) >= minoffset) &
                              (np.min(receiver) > source)) | \
                             ((np.max(receiver) < source) &
                              (np.min(offsets) >= minoffset) &
                              (np.min(offsets) <= maxoffset))

            if shot_condition:
                xmid = round(temp_stream.midpoint, 2)
                use_loc = True

                # optionally: use only xmids provided in list
                if 'xmids' in kwargs:
                    if xmid not in kwargs['xmids']:
                        use_loc = False

                if use_loc:
                    windows[win_id] = temp_stream

                    win_id += 1

            mintrace += win_move
            maxtrace += win_move

        return windows

    def preprocess_windows(self, windows, type = 'filter', **kwargs):
        """run a preprocessing step on all windows"""

        processed_windows = {}
        for win_id, win in windows.items():
            func = getattr(win, type)
            func(**kwargs)
            processed_windows[win_id] = win

        return processed_windows

    def transform_windows(self, windows, trafo_type = 'phaseshift'):
        """apply wavefield transformation to data stored in all windows"""

        processed_windows = {}
        for win_id, win in windows.items():
            win.transform(trafo_type = trafo_type)
            processed_windows[win_id] = win

        return processed_windows

    def _curve(self, id = 0):
        """set the dispersion curve"""

        if len(self.picks) > 0:
            if id in self.picks[self._pck_mode]:
                self.curve = DispersionCurve()
                self.curve.init_data(freq=self.picks[self._pck_mode][id]['frequency'],
                                      vel=self.picks[self._pck_mode][id]['velocity'])
            else:
                raise ValueError(f'Mode id {id} does not exist yet.')
        else:
            raise ValueError('No pick set has been created so far.')


    def _save_dc(self, path2disp, fname = None, id = 0, ext = 'csv',**kwargs):
        """
        save picked dispersion curve to file

        Parameters
        ----------
        path2disp : str, storage location
        fname : str, file name
        id : int, mode id
        ext : str, file format in which the dispersion curve should be saved
        kwargs
        -------
        """

        xmid = self.midpoint

        prjdir = os.path.join(path2disp, str(xmid))
        safe_makedirs(prjdir)

        if fname is None:
            fname = f'{self.pre}_ch{len(self._pst)}_{self.trafo_type}'

        if self.curve is not None:
            self.curve.save(prjdir,fname, ext)

        else:
            if len(self.picks) > 0:
                if id in self.picks[self._pck_mode]:
                    curve = DispersionCurve()
                    curve.init_data(freq=self.picks[self._pck_mode][0]['frequency'],
                                    vel=self.picks[self._pck_mode][0]['velocity'])
                    curve.save(prjdir, fname, ext)
                else:
                    warn_msg = f'Mode id {id} does not exist yet. No data has been saved.'
                    self.logger.warning(warn_msg)
            else:
                warn_msg = 'No pick set has been created so far. No data has been saved.'
                self.logger.warning(warn_msg)

    def _norm_power(self,power):
        """normalize power spectrum"""

        norm_power = np.zeros_like(power)
        global_max = np.nanmax(np.abs(power))

        for i in range(norm_power.shape[1]):
            pow = power[:,i]
            if self.use_local_max_power:
                local_max = np.nanmax(np.abs(pow))
                if local_max != 0:
                    pow = pow / np.nanmax(np.abs(pow))
            else:
                pow = pow/global_max
            norm_power[:,i] = pow

        return norm_power

    # %% PLOTTING
    def plot(self,type = '',**kwargs):
        """Create static plots"""

        if type == 'geometry' or type == 'geom':
            fig = self._plotGeometry(**kwargs)
            return fig
        elif type == 'seismogram' or type == '' or type == 'TX':
            fig = self._plotSeismogram(**kwargs)
            return fig
        elif type == 'spectrogram' or type == 'FX':
            fig = self._plotSpectrogram(**kwargs)
            return fig
        elif type == 'spectra':
            fig = self._plotSpectra(**kwargs)
            return fig
        elif type == 'spectrogramComposite':
            fig = self._plotSpectrogramComposite(**kwargs)
            return fig
        elif type == 'FK':
            fig = self._plotFK(**kwargs)
            return fig
        elif type == 'SFR':
            fig = self._plotSFR(**kwargs)
            return fig
        elif type == 'FV':
            fig = self._plotDispersionImage(**kwargs)
            return fig
        elif type == 'FVComposite':
            fig = self._plotDispersionImageComposite(**kwargs)
            return fig
        elif type == 'geomShort':
            fig = self._plotGeomShort()
            return fig

        else:
            self.logger.error(f'Invalid attribute "{type}".')

    def _plotGeomShort(self,**kwargs):
        """plot acquisition setup of shot file"""

        fig = Figure(figsize=(8, 4), constrained_layout=True)
        ax = fig.add_subplot(111)

        color = kwargs.pop('color', 'lightgrey')

        all_receiver = self._receiver(self._st)
        ax.scatter(all_receiver, np.zeros(len(all_receiver)), marker='v', c='k',label='active channels',alpha = 0.7)
        ax.scatter(self.receiver, np.zeros(len(self.receiver)), marker='v', c=color,label='selected channels')
        ax.scatter(self.source, 0.2, marker='*', s= 60,c='r', label='shot location')

        # Use the min and max source position instead
        if self.source < np.min(all_receiver):
            xmin = self.source
            xmax = np.max(all_receiver)
        elif self.source > np.max(all_receiver):
            xmin = np.min(all_receiver)
            xmax = self.source
        else:
            xmin = np.min(all_receiver)
            xmax = np.max(all_receiver)

        ax.set_xlim([xmin - 1,
                      xmax + 1])
        ax.set_ylim([-0.5, 0.5])
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        return fig

    def _plotGeometry(self,axes=None, outfile=None, fmt=None, show=True, gui = False,**kwargs):
        """plot acquisition setup of shot file"""

        figsize = kwargs.pop('figsize', (8, 4))
        if axes is None:
            if gui:
                fig = Figure(figsize=figsize, constrained_layout=True)
                ax = fig.add_subplot(111)
            else:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            ax = axes
            fig = ax.figure

        all_receiver = self._receiver(self._st)

        ax.scatter(all_receiver, np.zeros(len(all_receiver)), marker='v', c='k',label='active channels',alpha = 0.7)
        ax.scatter(self.receiver, np.zeros(len(self.receiver)), marker='v', c='lightgrey',label='selected channels')
        ax.scatter(self.source, 0.2, marker='*', s= 60,c='r', label='shot location')
        ax.legend(loc='upper right', ncol=3, frameon=True,
                  edgecolor = 'k',fontsize = 6,bbox_to_anchor=(1, 1.5)).get_frame().set_boxstyle('Square', pad=0.2)

        # Use the min and max source position instead
        if self.source < np.min(all_receiver):
            xmin = self.source
            xmax = np.max(all_receiver)
        elif self.source > np.max(all_receiver):
            xmin = np.min(all_receiver)
            xmax = self.source
        else:
            xmin = np.min(all_receiver)
            xmax = np.max(all_receiver)

        ax.set_xlim([xmin - 1,
                      xmax + 1])
        ax.set_ylim([-0.5, 0.5])
        ax.set_xlabel('x (m)')
        ax.get_yaxis().set_ticks([])
        ax.grid(axis='x', which='both', linestyle=":")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1)

        if axes is not None:
            return ax

        if show:
            plt.show()

        if outfile:
            if fmt:
                fig.savefig(outfile, format=fmt)
            else:
                fig.savefig(outfile)

        return fig

    def _plotSeismogram(self, axes=None, st = None, amp_scale=None, outfile=None,
                        fmt=None, show=True, gui = False, **kwargs):
        """plot seismogramm of a single shot file"""

        figsize = kwargs.pop('figsize', (8, 8))
        if axes is None:
            if gui:
                fig = Figure(figsize=figsize, constrained_layout=True)
                ax = fig.add_subplot(111)
            else:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            ax = axes
            fig = ax.figure

        if st is None:
            if self._pst is None:
                st = self._st.copy()
            else:
                st = self._pst.copy()

        try:
            receiver = self.receiver
        except AttributeError:
            receiver = np.arange(1,len(st)+1,1)

        color = kwargs.pop('color', 'dimgrey')
        linewidth = kwargs.pop('linewidth', 0.5)
        title = kwargs.pop('title', f'Shotfile {self.pre}')
        step = kwargs.pop('tick_scale', len(receiver)//3)
        alpha = kwargs.pop('alpha',0.5)
        show_map = kwargs.pop('show_map', False)

        amps = self._amps(st=st)
        amps = self._detrend_signal(amps)

        if self.norm_amps:
           amps = self._normalize_amps(amps)

        nchannels, npts = amps.shape

        t = np.arange(npts) * st[0].stats.delta

        if amp_scale is None:
            amp_scale = np.mean(np.diff(np.array(receiver))) / 2

        if amp_scale == 0:
            amp_scale = 1

        for i, pos in enumerate(receiver):
            amps[i] = amps[i] * amp_scale

        if show_map:
            ax.imshow(amps.T,
                     extent=[-1,(len(amps)-1)+1,t[-1],t[0]],
                     cmap=kwargs.pop('cmap','bwr'),
                     aspect='auto',
                     vmin=-1, vmax=1,
                     interpolation='bicubic')

        for i, trace in enumerate(amps):

            if not show_map:

                tmp_trace = copy.deepcopy(trace)
                tmp_trace[trace <= 0] = 0

                ax.fill_betweenx(t, np.ones_like(tmp_trace)*i, tmp_trace+i,
                                 color=color, linewidth = 0)

            if ((i+1) % 5 == 0) & ((i+1) % 10 != 0):
                ax.plot(trace+i, t, color=color, linewidth=linewidth+0.5,alpha = alpha)
            elif (i+1) % 10 == 0:
                ax.plot(trace+i, t, color=color, linewidth=linewidth+1,alpha = alpha)
            else:
                ax.plot(trace + i, t, color=color, linewidth=linewidth, alpha=alpha)

        if title is not None:
            title = title.replace('_',' ')
            ax.set_title(title, fontweight='bold')
        ax.set_ylabel("time (s)")

        ax.set_xlim(-2,(len(amps)-1)+2)

        xticks = np.arange(len(receiver))[step - 1::step]
        ax.set_xticks(np.arange(len(receiver))[step - 1::step])

        if kwargs.pop('show_xticks',True):

            ax.xaxis.tick_top()
            xticklabels = []
            for i in xticks:
                xticklabels.append(f'{st[i].stats.channel}\n{receiver[i]}')
            ax.set_xticklabels(xticklabels)

            text = 'channel:\nx (m):'
            at = AnchoredText(text,
                              loc='lower left',
                              prop=dict(ha='right',fontweight = 'bold'),
                              bbox_to_anchor=(-0.125, 1.0),
                              pad=0, borderpad=0.8, frameon=False,
                              bbox_transform=ax.transAxes)
            ax.add_artist(at)
        else:
            xticklabels = []
            for i in xticks:
                xticklabels.append(f'{receiver[i]}')
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel('x (m)')
            ax.locator_params(axis='x', nbins=4)

        ymin = kwargs.pop('ymin', np.min(t))
        ymax = kwargs.pop('ymax', np.max(t))
        ax.set_ylim([ymin, ymax])

        ax.grid(axis='y', linestyle=":")
        ax.invert_yaxis()

        if axes is not None:
            return ax

        if outfile:
            if fmt:
                fig.savefig(outfile, format=fmt)
            else:
                fig.savefig(outfile)
        elif show:
            #plt.tight_layout()
            plt.show()

        return fig

    def _plotSpectrogram(self, axes=None, outfile=None, fmt=None, show=True, gui = False, **kwargs):
        """plot spectrogram"""

        figsize = kwargs.pop('figsize', (8, 8))
        if axes is None:
            if gui:
                fig = Figure(figsize=figsize, constrained_layout=True)
                ax = fig.add_subplot(111)
            else:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            ax = axes
            fig = ax.figure

        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()

        dt = self.dt
        npts = self.npts
        receiver = self.receiver
        source = self.source

        step = kwargs.pop('tick_scale', 10)

        # receiver-source offsets; time series matrix
        offsets = self._aoffsets(receiver, source)

        amps = self._amps(st=st)
        if kwargs.pop("norm", True):
            amps = self._normalize_amps(amps)

        if source > receiver[-1]:
            offsets = offsets[::-1]
            amps = np.flipud(amps)
        offsets = np.array(offsets)

        # frequency range
        omega_fs = 2 * np.pi * 1 / dt
        omega = np.arange(npts) * (omega_fs / npts)
        freq = omega / (2 * np.pi)

        # fft
        u = np.fft.fft(amps)/amps.shape[1]
        df = (1/self.df)/amps.shape[1]

        # absolute amplitudes in db
        abs_amps = np.empty((len(freq), len(receiver)))
        for row in range(len(receiver)):
            ui = np.abs(u[row, :])
            abs_amps[:, row] = ui
        abs_amps[np.isnan(abs_amps)] = 0

        # db
        abs_amps = 10 * np.log10(abs_amps)

        abs_amps_min = round(np.percentile(abs_amps,5))
        abs_amps_max = round(np.percentile(abs_amps, 95))

        title = kwargs.pop('title', None)

        img = ax.contourf(np.arange(len(offsets)),
                        freq,
                        abs_amps,
                        np.linspace(abs_amps_min, abs_amps_max, 21),
                        #extend='both',
                        cmap=plt.cm.get_cmap(kwargs.setdefault('cmap', 'viridis')))

        ax.set_xlim(-2, (len(self.receiver) - 1) + 2)

        if kwargs.pop('show_xticks',True):
            ax.xaxis.tick_top()
            offsets = self.receiver
            xticks = np.arange(len(offsets))[step-1::step]
            ax.set_xticks(np.arange(len(offsets))[step-1::step])
            xticklabels = []
            for i in xticks:
                xticklabels.append(f'{st[i].stats.channel}\n{offsets[i]}')
            ax.set_xticklabels(xticklabels)

            text = 'channel nr.:\ndistance (m):'
            at = AnchoredText(text,
                              loc='lower left',
                              prop=dict(ha='right'),
                              bbox_to_anchor=(-0.15, 1.0),
                              pad=0, borderpad=0.8, frameon=False,
                              bbox_transform=ax.transAxes)
            ax.add_artist(at)

        plt.colorbar(img, label='amplitude (db)', ax=ax)
        ax.set_ylabel('frequency (Hz)')
        ax.set_ylim([1, 100])
        ax.grid(True, linestyle=':')

        if title is not None:
            fig.suptitle(self.pre)

        if outfile:
            if fmt:
                fig.savefig(outfile, format=fmt)
            else:
                fig.savefig(outfile)

        if axes is not None:
            return ax

        elif show:
            #plt.tight_layout()
            plt.show()

        return fig

    def _plotSpectra(self, axes=None, outfile=None, fmt=None, show=True, gui = False, **kwargs):
        """plot spectra"""

        figsize = kwargs.pop('figsize', (8, 8))
        if axes is None:
            if gui:
                fig = Figure(figsize=figsize, constrained_layout=True)
                ax = fig.add_subplot(111)
            else:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            ax = axes
            fig = ax.figure

        if self._pst is None:
            st = self._st.copy()
        else:
            st = self._pst.copy()

        dt = self.dt
        npts = self.npts
        receiver = self.receiver
        source = self.source

        amps = self._amps(st=st)
        if kwargs.pop("norm", True):
            amps = self._normalize_amps(amps)

        if source > receiver[-1]:
            amps = np.flipud(amps)

        # frequency range
        omega_fs = 2 * np.pi * 1 / dt
        omega = np.arange(npts) * (omega_fs / npts)
        freq = omega / (2 * np.pi)

        # fft
        u = np.fft.fft(amps)/amps.shape[1]
        df = (1/self.df)/amps.shape[1]

        # compute absolute normalized psd
        norm_amps = np.zeros((len(freq), len(receiver)))
        for row in range(len(receiver)):
            ui = np.abs(u[row, :])
            norm_amps[:, row] =ui

        cmap = getattr(plt.cm, kwargs.pop('cmap', 'viridis'))
        color = cmap(np.linspace(0, 1, len(receiver)))
        for row in range(len(receiver)):
            ax.plot(norm_amps[:, row], freq, color=color[row])
        ax.xaxis.tick_top()
        ax.set_xlabel('normalized amplitudes')
        ax.xaxis.set_label_position('top')
        ax.grid(True, linestyle=':')
        ax.set_ylabel('frequency (Hz)')
        ax.set_ylim([1, 100])

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)

        import matplotlib as mpl
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', color, len(color))
        norm = plt.Normalize(st[0].stats.channel, st[-1].stats.channel)
        smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(smap, ax=ax)
        cbar.set_label('Channel Nr.')

        if outfile:
            if fmt:
                fig.savefig(outfile, format=fmt)
            else:
                fig.savefig(outfile)

        if axes is not None:
            return ax

        elif show:
            #plt.tight_layout()
            plt.show()

        return fig

    def _plotSpectrogramComposite(self, axes=None, outfile=None, fmt=None, show=True, **kwargs):
        """plot seismogram, spectrogram and spectra"""

        if axes is None:
            fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        else:
            ax = axes
            fig = axes.figure

        self._plotSeismogram(axes=ax[0], title=None, show=False,**kwargs)
        self._plotSpectrogram(axes=ax[1],show=False,**kwargs)
        self._plotSpectra(axes=ax[2],show=False,**kwargs)

        if outfile:
            if fmt:
                fig.savefig(outfile, format=fmt)
            else:
                fig.savefig(outfile)

        if axes is not None:
            return ax

        elif show:
            plt.tight_layout()
            plt.show()

        return fig

    def _plotFK(self,FK_data=None,axes= None, outfile=None, fmt=None, show=True, gui = False, **kwargs):
        """plot FK image"""

        figsize = kwargs.pop('figsize', (8, 8))
        cmap = kwargs.pop('cmap', 'Greys')

        if axes is None:
            if gui:
                fig = Figure(figsize=figsize, constrained_layout=True)
                ax = fig.add_subplot(111)
            else:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            ax = axes
            fig = ax.figure

        if FK_data is None:
            if self.FK_data:
                FK_data, theta, kw, freq, iX, iT = self.FK_data.values()
            else:
                FK_data, theta, kw, freq, iX, iT = self._fk_transform(**kwargs)
        else:
            iF = nextpow2(self.npts)[1]
            k = np.fft.fftfreq(iF, self.dx)
            kw = k * 2 * np.pi
            freq = np.linspace(0, 0.5/self.dt, iF // 2+1)

        # Determine kmin/kmax values
        kmax_val = kwargs.pop('kmax', self.kmax if self.kmax is not None else np.max(kw))
        kmin_val = kwargs.pop('kmin', self.kmin if self.kmin is not None else np.min(kw))
        self.kmax = kmax_val
        self.kmin = kmin_val

        # Frequency and wavenumber indices
        fmin_idx = np.argmin(np.abs(freq - self.fmin))
        fmax_idx = np.argmin(np.abs(freq - self.fmax))
        kmin_idx = np.argmin(np.abs(kw - self.kmin))
        kmax_idx = np.argmin(np.abs(kw - self.kmax))

        # Slice FK_data if it was generated internally
        if FK_data is not None and FK_data.ndim == 2:
            FK_data = FK_data[fmin_idx:fmax_idx, kmin_idx:kmax_idx]

            if FK_data.size == 0:
                return None

            # Normalize if requested
            if self.norm_power:
                FK_data = self._norm_power(FK_data)
                label = "amplitudes (normalized)"
            else:
                label = "amplitudes"

            # Plot image
            img = ax.imshow(
                np.abs(FK_data),
                aspect='auto',
                origin='lower',
                extent=[self.kmin, self.kmax, self.fmin, self.fmax],
                cmap=plt.cm.get_cmap(cmap)
            )

            ax.set_xlabel('wavenumber (rad/m)')
            ax.set_ylabel('frequency (Hz)')
            ax.grid(linestyle=':')

            for spine in ax.spines.values():
                spine.set_linewidth(1)

            plt.colorbar(img, label=label, ax=ax)

            if outfile:
                fig.savefig(outfile, format=fmt if fmt else None)

            if axes is not None:
                return ax

            elif show:
                #plt.tight_layout()
                plt.show()

            return fig

        return None

    def _plotSFR(self, axes=None, st = None, amp_scale=None, outfile=None,
                        fmt=None, show=True, gui = False, **kwargs):
        """plot swept-frequency record (SFR)"""

        figsize = kwargs.pop('figsize', (8, 8))
        if axes is None:
            if gui:
                fig = Figure(figsize=figsize, constrained_layout=True)
                ax = fig.add_subplot(111)
            else:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            ax = axes
            fig = ax.figure

        if st is None:
            if self._pst is None:
                st = self._st.copy()
            else:
                st = self._pst.copy()

        color = kwargs.pop('color', 'dimgrey')
        linewidth = kwargs.pop('linewidth', 0.5)
        scale = kwargs.pop('scale', 1)
        alpha = kwargs.pop('alpha',0.5)
        detrend = kwargs.pop('detrend', False)

        amps = self._amps(st=st)
        if detrend:
            amps = self._detrend_signal(amps)
        if self.norm_amps:
            amps = self._normalize_amps(amps)

        # add empty traces in plot if channels were removed for the visualisation
        receiver = self.receiver
        trace_sep = np.diff(receiver)
        empty_tr_idx = np.where(trace_sep != self.dx)[0]
        nchannels, npts = amps.shape

        # time
        t = np.arange(npts) * self.dt

        # stretch function
        stretch = np.sin(2*np.pi*self.fmin*t + np.pi*(self.fmax-self.fmin)*t**2/self.SFR_time)

        # convolution
        SFR = np.zeros((nchannels,2*npts-1))
        for i in range(nchannels):

            SFR[i] = np.convolve(amps[i],stretch)

        if amp_scale is None:
            amp_scale = np.mean(np.diff(np.array(self.receiver))) / 2

        for i, pos in enumerate(self.receiver):
            SFR[i] = SFR[i] * amp_scale

        # time
        t = np.arange(len(SFR[i])) * self.dt

        # plot SFR
        for i, trace in enumerate(SFR):
            ax.plot(trace+i, t, color=color, linewidth=linewidth,alpha = alpha)

        ymin = kwargs.pop('ymin', np.min(t))
        ymax = kwargs.pop('ymax', np.max(t))
        ax.set_ylim([ymin, ymax])

        ax.grid(axis='y', linestyle=":")
        ax.invert_yaxis()

        ax.set_xlim(-2, (len(self.receiver) - 1) + 2)

        if kwargs.pop('show_xticks',True):
            ax.xaxis.tick_top()
            offsets = self.receiver
            step = int(len(offsets) / scale)
            xticks = np.arange(len(offsets))[step::step]
            ax.set_xticks(np.arange(len(offsets))[step::step])
            xticklabels = []
            for i in xticks:
                xticklabels.append(f'{st[i].stats.channel}\n{offsets[i]}')
            ax.set_xticklabels(xticklabels)

            text = 'channel nr.:\ndistance (m):'
            at = AnchoredText(text,
                              loc='lower left',
                              prop=dict(ha='right'),
                              bbox_to_anchor=(-0.15, 1.0),
                              pad=0, borderpad=0.8, frameon=False,
                              bbox_transform=ax.transAxes)
            ax.add_artist(at)

        if outfile:
            if fmt:
                fig.savefig(outfile, format=fmt)
            else:
                fig.savefig(outfile)

        if axes is not None:
            return ax

        elif show:
            plt.tight_layout()
            plt.show()

        return fig

    def _plotDispersionImage(self,axes=None, outfile=None, fmt=None, show=True,
                             gui = False, **kwargs):
        """plot dispersion image"""

        figsize = kwargs.pop('figsize', (8, 8))
        if axes is None:
            if gui:
                fig = Figure(figsize=figsize, constrained_layout=True)
                ax = fig.add_subplot(111)
            else:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            ax = axes
            fig = ax.figure

        dispersive_energy = self.dispersive_energy

        if self.dispersive_energy is None:
            self.logger.error('Dispersion image cannot be displayed, '
                              'because wavefield transformation not yet performed.')
            return None

        plotkey = kwargs.pop("plotkey","velocity")

        daty = self.velocity
        labely = "phase velocity (m/s)"

        datx = self.frequency
        labelx = "frequency (Hz)"

        if self.norm_power:
            # work around
            local_max = self.use_local_max_power
            self.use_local_max_power = False
            dispersive_energy = self._norm_power(dispersive_energy)
            self.use_local_max_power = local_max
            label = "amplitudes (normalized)"
            limits = (0, 1)
        else:
            label = "amplitudes"
            limits = (np.min(dispersive_energy), np.max(dispersive_energy))

        # plot dispersion image and dispersion curves
        contours = np.linspace(limits[0], limits[1], 21)

        img = ax.contourf(datx,
                           daty,
                           dispersive_energy.real,
                           contours,
                           #extend='both',
                           cmap=plt.cm.get_cmap(kwargs.pop('cmap', 'Greys')))

        ax.grid(True, linestyle=':')
        ax.set_xlabel(labelx)
        ax.set_ylabel(labely)
        ax.set_ylim([np.min(daty), np.max(daty)])

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)

        if kwargs.pop("show_cbar", True):
            plt.colorbar(img, label=label, ax=ax,orientation = 'vertical', pad = 0.01)

        if outfile:
            if fmt:
                fig.savefig(outfile, format=fmt)
            else:
                fig.savefig(outfile)

        if axes is not None:
            return ax

        elif show:
            #plt.tight_layout()
            plt.show()

        return fig

    def _plotDispersionImageComposite(self, axes=None, outfile=None, fmt=None, show=True,title = None,
                             **kwargs):
        """plot geometry, seismogram and dispersion image"""

        if axes is None:
            fig = plt.figure(figsize=(10, 6))
            gs = fig.add_gridspec(3, 2)
            ax0 = fig.add_subplot(gs[0, 0:2])
            ax1 = fig.add_subplot(gs[1:3, 0])
            ax2 = fig.add_subplot(gs[1:3, 1])
        else:
            ax0 = axes[0]
            ax1 = axes[1]
            ax2 = axes[2]
            fig = axes.figure

        # plot geometry
        self._plotGeometry(axes = ax0)
        ax0.set_title(self.pre, fontweight='bold')

        # plot seismogram
        self._plotSeismogram(axes=ax1, show = False,  **kwargs)

        # plot dispersion image and dispersion curves
        ax2.set_title(title)
        self._plotDispersionImage(axes=ax2,keyy='velocity',**kwargs)
        if self._pick:
            for key1 in self.picks.keys():
                for key2 in self.picks[key1].keys():

                    ax2.plot(self.picks[key1][key2]['frequency'], self.picks[key1][key2]['velocity'],
                            markeredgecolor='k',
                            marker="s",
                            markersize=5, color='white',
                            linewidth=0)

        #fig.align_ylabels()
        plt.tight_layout()

        if outfile:
            if fmt:
                fig.savefig(outfile, format=fmt)
            else:
                fig.savefig(outfile)

        if axes is not None:
            return axes

        elif show:
            plt.tight_layout()
            plt.show()

        return fig

    def _plotMOPA(self, data, stop=1, axes=None, **kwargs):

        if axes is None:
            fig,ax = plt.subplots(3, 1, figsize=(8, 8))
        else:
            ax = axes
            fig = axes.figure

        cmap = kwargs.pop('cmap', 'Greys')
        cmap_discret = getattr(plt.cm, cmap)

        colors = cmap_discret(np.linspace(0, 0.9, len(np.arange(len(data)))))

        for ii in np.arange(len(data)):

            # amplitude
            ax[0].scatter(data[ii]['offsets'], data[ii]['mag'],
                                marker="o", s=15, c=colors[ii].reshape(1, -1), edgecolor='k',
                                linewidth=0)

            # fitted lines
            ax[1].plot(data[ii]['offsets'],
                       phase_response(data[ii]['offsets'],
                                      data[ii]['k0'],
                                      data[ii]['phi0']),
                       color=colors[ii].reshape(1, -1),
                       alpha=0.75, linewidth=0.8, linestyle='dashed')

            # phase data
            ax[1].scatter(data[ii]['offsets'],
                                np.unwrap(data[ii]['phase'], axis=0),
                                marker="o", s=15, c=colors[ii].reshape(1, -1),
                                edgecolor='k',
                                linewidth=0, zorder=2)

            # dispersion curve from MOPA
            if data[ii]['chi2'] > stop:
                ec = 'r'
                label = f'MOPA - chi^2 > {stop}'
            else:
                ec = 'k'
                label = f'MOPA - chi^2 <= {stop}'

            ax[2].scatter(data[ii]['frequency'], data[ii]['velocity'], marker="o", s=25, c=colors[len(colors) // 2].reshape(1, -1),
                          edgecolor=ec, linewidth=0.8, zorder=2,
                          label=label, alpha=0.7)

        # if kwargs.setdefault('showResults', True):
        #     # %% dispersion curve obtained in classical sense as comparison
        #     self._fdbf()
        #     self.dcpicking(pck_mode='auto')
        #     f_fdbf = self.picks['auto'][0]['frequency']
        #     v_fdbf = self.picks['auto'][0]['velocity']
        #     ax[2].scatter(f_fdbf, v_fdbf, marker="s", s=25, c='k',
        #                   edgecolor='k', linewidth=0.8, zorder=-2, label='FDBF')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax[2].legend(by_label.values(), by_label.keys(), loc='upper right',
                     frameon=True, title='Transformation')

        for axi in ax.flat:
            axi.grid(True, linestyle=':')

        ax[0].set_xlabel("offset (m)")
        ax[1].set_xlabel("offset (m)")
        ax[0].set_ylabel(f"amplitude")
        ax[1].set_ylabel(f"phase (rad)")
        plot_colorBar(ax[0], self.fmin, self.fmax, label='f (Hz)', orientation='vertical', cmap=cmap)
        plot_colorBar(ax[1], self.fmin, self.fmax, label='f (Hz)', orientation='vertical', cmap=cmap)

        ax[2].set_ylabel(f"phase velocity (m/s)")
        ax[2].set_xlabel(f"frequency (Hz)")
        ax[2].set_ylim([self.vmin, self.vmax])
        ax[2].set_xlim([self.fmin, self.fmax])
        ax[0].set_title('MOPA - Shotfile: ' + self.pre, fontweight='bold')
        fig.align_ylabels(ax)

        plt.tight_layout()

        return ax


