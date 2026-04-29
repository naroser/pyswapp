import matplotlib.path as mpltPath

from .utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class DispersionCurve:
    """class to manipulate and save dispersion curve data"""

    def __init__(self,mode = 0, wave = 'rayleigh'):

        self.data = None
        self.data_orig = None

        if mode < 0:
            raise ValueError('Mode number must be 0 or higher (0 = fundamental mode).')
        if wave not in ['rayleigh', 'love']:
            raise ValueError('Wave type must "love" or "rayleigh".')

        self._mode = mode
        self._wave = wave

    def init_data(self, freq, vel, err = None):
        """
        Initialize data

        Parameters
        ----------
        freq : array-like, frequencies
        vel : array-like, velocities
        err : array-like, optional, data errors

        """

        self.frequency = freq
        self.period = 1/freq
        self.velocity = vel
        self.wavenumber = self._compute_wavenumber(self.frequency,self.velocity)
        self.wavelength = self._compute_wavelength(self.frequency,self.velocity)

        if err is None:
            self.error = np.zeros(len(self.wavelength))
        else:
            self.error = err

        self.data = self.get_data()
        self.npts = len(self.data)

        if self.data_orig is None:
            self.data_orig = self.data.copy()
            self.npts_orig = len(self.data_orig)

    def read(self, fname):
        """
        Import data from file

        Parameters
        ----------
        fname : str, path to dispersion curve data

        """

        if fname.endswith('.csv'):
            dat, _ = read_DC_csv(fname)
            err = dat[:, 2]

        else:
            raise NotImplementedError

        self.init_data(dat[:, 0], dat[:, 1], err)

    def _estimate_error(self,offsets=None,nchannels=24,dx=1,**kwargs):
        """
        Estimate dispersion-curve uncertainty after O'Neill et al. (2003).

        Parameters
        ----------
        offsets : array-like or None, receiver offsets. If provided, dx and nchannels are inferred from it
        nchannels : int, optional, Number of channels (ignored if offsets provided)
        dx : float, optional, Receiver spacing (ignored if offsets provided)

        Returns
        -------
        ndarray

        References
        ----------
        O’Neill, A., Dentith, M., & List, R., 2003. Full-waveform P-SV
        reflectivity inversion of surface waves for shallow engineering
        applications, Exploration Geophysics, 34(3), 158–173.

        """
        err = lorentzian_err(offsets, self.velocity, self.frequency, nchannels, dx, **kwargs)
        return err

    def estimate_error(self,**kwargs):
        """Estimate error"""

        err = self._estimate_error(**kwargs)
        data = self.data.copy()
        data['error'] = err

        curve_new = DispersionCurve(self._mode, self._wave)
        curve_new.init_data(freq=data['frequency'], vel=data['velocity'], err=data['error'])
        return curve_new

    def set(self,param, value, orig = True):
        """
        Set a parameter

        Parameters
        ----------
        param : str, parameter name
        value : array-like, parameter
        orig : bool, default True

        """

        self.data[param] = value
        if orig:
            self.data_orig[param] = value

    def _compute_wavenumber(self,frequency,velocity):
        """
        Compute wavenumber

        Parameters
        ----------
        frequency : ndarray, frequency
        velocity : ndarray, velocity

        Returns
        -------
        ndarray

        """
        return wavenumber(frequency,velocity)

    def _compute_wavelength(self,frequency,velocity):
        """
        Compute wavelength


        Parameters
        ----------
        frequency : ndarray, frequency
        velocity : ndarray, velocity

        Returns
        -------
        ndarray

        """
        return wavelength(frequency,velocity)

    def _compute_velocity(self,frequency,wavenumber):
        """
        Compute velocity

        Parameters
        ----------
        frequency : ndarray, frequency
        wavenumber : ndarray, wavenumber

        Returns
        -------
        ndarray

        """
        return phase_velocity(frequency,wavenumber)

    def _compute_frequency(self,wavelength,velocity):
        """
        Compute frequency

        Parameters
        ----------
        wavelength : ndarray, wavelength
        velocity : ndarray, velocity

        Returns
        -------
        ndarray

        """
        return frequency(wavelength,velocity)

    def get_data(self):
        """store data in a data frame"""

        data = np.array([self.frequency, self.velocity, self.wavelength, self.wavenumber, self.error]).T
        data = pd.DataFrame(data, columns=['frequency', 'velocity', 'wavelength', 'wavenumber', 'error'])

        return data

    def resample(self, pmin, pmax, pn, pspace = 'log', param = 'frequency', kind = 'cubic', inplace = False,**kwargs):
        """
        Resample all curves

        Parameters
        ----------
        pmin : float, minimum value
        pmax : float, maximum value
        pn : int, number of points
        pspace : str, default 'log', scale
        param: str, specify the parameter the new sampling interval is applied to
        kind : str, default 'cubic', specifies the kind of interpolation
        inplace : bool, default True, if True, perform operation in-place

        """

        data = self.data.copy()

        if param in ['frequency','wavelength']:
            if pspace == 'log':
                parx_new = np.geomspace(pmin,pmax,pn)
            else:
                parx_new = np.linspace(pmin,pmax,pn)

            parx = data[param].values
            pary = data['velocity'].values
            err = data['error'].values
            parx_new, pary_new, err_new = interp(parx,pary,parx_new,yerr=err,kind = kind,**kwargs)

            if param == 'wavelength':
                parx_new = self._compute_frequency(parx_new, pary_new)

            if inplace:
                self.init_data(freq=parx_new,vel=pary_new,err=err_new)
                self.npts = len(self.data)
            else:
                curve_new = DispersionCurve(self._mode,self._wave)
                curve_new.init_data(freq=parx_new,vel=pary_new,err=err_new)

                return curve_new
        else:
            raise ValueError('Wrong parameter key handed.')

    @staticmethod
    def _median_filter(vel, kernel_size=5):
        if kernel_size % 2 != 1:
            raise ValueError("kernel size must be odd")
        k2 = (kernel_size - 1) // 2
        y = np.zeros((len(vel), kernel_size), dtype=float)
        y[:, k2] = vel
        for i in range(k2):
            j = k2 - i
            y[j:, i] = vel[:-j]
            y[:j, i] = vel[0]
            y[:-j, -(i + 1)] = vel[j:]
            y[-j:, -(i + 1)] = vel[-1]
        return np.median(y, axis=1)

    @staticmethod
    def _moving_average_filter(vel, kernel_size=5):
        kernel = np.ones(kernel_size) / kernel_size
        vel_convolved = np.convolve(vel, kernel, mode='same')

        return vel_convolved

    def smooth(self,kernel_size=5,inplace = False):
        """
        Smooth data

        Parameters
        ----------
        kernel_size : int, kernel size
        inplace : bool, default True, if True, perform operation in-place

        """

        data = self.data.copy()

        vel = data['velocity'].values
        freq = data['frequency'].values
        err = data['error'].values

        vel_filt = self._median_filter(vel, kernel_size)
        vel_filt = self._moving_average_filter(vel_filt, kernel_size)

        extend = int(kernel_size/2+1)
        pary_new = vel_filt[extend:-extend]
        parx_new = freq[extend:-extend]
        err_new = err[extend:-extend]

        if inplace:
            self.init_data(freq=parx_new, vel=pary_new, err=err_new)
            self.npts = len(self.data)
        else:
            curve_new = DispersionCurve(self._mode, self._wave)
            curve_new.init_data(freq=parx_new, vel=pary_new, err=err_new)
            return curve_new

    @staticmethod
    def _points_within_poly(vertices, f, v):
        """create boolean vector for points inside a polygon"""

        path = mpltPath.Path(vertices)
        flags = path.contains_points(
            np.vstack((f, v)).T)

        return flags

    def markInvalid_by_poly(self, vertices):
        """mark invalid rows based on points inside a polygon"""

        data = self.data.copy()
        data = data.reset_index(drop=True)

        f = data.frequency
        vel = data.velocity

        flags = self._points_within_poly(vertices, f, vel)
        data['invalid'] = flags
        self.data = data

    def markInvalid(self, pmin=None, pmax=None, param = 'frequency', mask = None):
        """mark invalid data points"""

        data = self.data.copy()

        data = data.reset_index(drop=True)
        par = data[param].values
        min, max = pmin, pmax

        data['invalid'] = np.zeros_like(par, dtype=bool)

        if mask is not None:
            flagidx = mask
        else:
            if (min is None) and (max is None):
                flagidx = []
            elif (min is not None) and (max is None):
                flagidx = np.where(par < min)[0]
            elif (max is not None) and (min is None):
                flagidx = np.where(par > max)[0]
            else:
                flagidx = np.where((par < min) | (par > max))[0]

        data.loc[flagidx, ['invalid']] = True
        self.data = data

    def dropInvalid(self,inplace):
        """drop invalid data points"""

        data = self.data.copy()

        data = data.drop(data[data.invalid == True].index)
        data.reset_index(drop=True)
        data = data.drop(columns = ['invalid'])

        if inplace:
            self.init_data(freq=data['frequency'], vel=data['velocity'], err=data['error'])
            self.npts = len(self.data)
        else:
            curve_new = DispersionCurve(self._mode, self._wave)
            curve_new.init_data(freq=data['frequency'], vel=data['velocity'], err=data['error'])
            return curve_new

    def filter(self, pmin=None, pmax=None, param = 'frequency', inplace = False):
        """remove points outside of [pmin,pmax]"""
        self.markInvalid(pmin,pmax,param)
        return self.dropInvalid(inplace)

    def save(self, prjdir, pre, format):
        """save dispersion curve data to file"""

        data = self.data.copy()

        if format == 'csv':
            safe_makedirs(os.path.join(prjdir, "0_csv"))

            outfile = os.path.join(prjdir, f"0_csv/{pre}.csv")

            save2csv(outfile, np.asarray(data.frequency), np.asarray(data.velocity), np.asarray(data.error))

        else:
            raise ValueError(f"Data format not provided. "
                             f"Choose one of the formats: {'csv'} or {['ParkSeis','PS','dat','DAT']}.")

    def plot(self, data = None, axes=None, outfile=None, fmt=None, show=True, show_orig = False, **kwargs):
        """plot dispersion curve"""

        if data is None:
            data = self.data.copy()

        keyy = kwargs.pop('y_value', 'velocity')
        keyx = kwargs.pop('x_value', 'frequency')
        color = kwargs.pop('color', 'royalblue')
        label = kwargs.pop('label','data')
        alpha = kwargs.pop('alpha',0.7)
        marker = kwargs.pop('marker','o')
        marker_size = kwargs.pop('size',20)
        axis_style = kwargs.pop('axis_style', False)
        figsize = kwargs.pop('figsize',(8,8))
        edgecolor = kwargs.pop('edgecolor',color)

        if keyy == 'velocity':
            err = data.error.values
            labely = "phase velocity (m/s)"
        elif keyy == 'wavenumber':
            err_vr = data.error.values
            err = wavenumber(data.frequency.values,err_vr)
            labely = 'wavenumber (rad/m)'
        else:
            raise KeyError
        if keyx == 'frequency':
            labelx = "frequency (Hz)"
        elif keyx == 'wavelength':
            labelx = 'wavelength (m)'
        else:
            raise KeyError

        if axes is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout = True)
        else:
            ax = axes
            fig = ax.figure

        if (self.data_orig is not None) and show_orig:
            data_orig = self.data_orig.copy()
            ax.scatter(data_orig[keyx], data_orig[keyy], marker=marker, s=marker_size, c='k', edgecolor='k',
                       linewidth=0.8, zorder=2, label='orig. data',alpha = alpha, **kwargs)

            if keyy == 'velocity':
                err_orig = data_orig.error.values
            elif keyy == 'wavenumber':
                err_vr = data_orig.error.values
                err_orig = wavenumber(data.frequency.values, err_vr)
            else:
                raise KeyError

            ax.fill_between(x=data_orig[keyx], y1=data_orig[keyy] - err_orig, y2=data_orig[keyy] + err_orig, alpha=0.1,
                            color='k', linewidth=2, edgecolor=None)

        ax.scatter(data[keyx], data[keyy], marker=marker, s=marker_size, c=color, edgecolor=edgecolor,
                   linewidth=0.8, zorder=2, label = label,alpha = alpha)

        if (np.sum(err) != 0) & kwargs.setdefault('showErr', False):

            ax.fill_between(x = data[keyx], y1 = data.velocity - err, y2 =  data.velocity + err, alpha = 0.2,
                            color = color, linewidth = 2, edgecolor = None, label = 'data error')

        ax.grid(True, linestyle=':')

        if label:
            ax.legend(loc='upper right', edgecolor='k', frameon=True, fontsize=kwargs.setdefault('fontsize', 12))

        ax.set_xlabel(labelx)
        ax.set_ylabel(labely)

        # configure styling
        if axis_style:

            plt_xmin = kwargs.pop('xmin',np.min(data[keyx]))
            plt_xmax = kwargs.pop('xmax',np.max(data[keyx]))

            plt_ymin = kwargs.pop('ymin',np.min(data[keyy])-np.max(err))
            plt_ymax = kwargs.pop('ymax',np.max(data[keyy])+np.max(err))

            ax.set_ylim([plt_ymin,plt_ymax])

            if self.data_orig is not None:
                data_orig = self.data_orig.copy()
                ax.set_xlim([np.min(data_orig[keyx])-2.5, np.max(data_orig[keyx])+2.5])
            else:
                ax.set_xlim([plt_xmin,plt_xmax])

        if axes is not None:
            return ax

        if outfile:
            if fmt:
                fig.savefig(outfile, format=fmt)
            else:
                fig.savefig(outfile)
        elif show:
            plt.show()

        return fig

    def plotColumn(self, data = None, axes=None, outfile=None, fmt=None, show=True, **kwargs):

        if axes is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            ax = axes
            fig = ax.figure

        if data is None:
            data = self.data.copy()

        plot_vphase(ax, data.velocity.to_numpy(), f=data.frequency.to_numpy(), lam=None, **kwargs)

        if axes is not None:
            return ax

        plt.tight_layout()

        if outfile:
            if fmt:
                fig.savefig(outfile, format=fmt)
            else:
                fig.savefig(outfile)
        elif show:
            plt.show()
        else:
            return fig