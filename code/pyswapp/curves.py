from .utils.utils import *
from .utils.physics import *
from .curve import DispersionCurve
import warnings

import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

class CombineCurves:
    """combine dispersion curves and perform simple statistics"""

    def __init__(self, data):

        self.data = data

    def _binning(self, lam_vec, vel_vec, lam_min = 1, lam_max = 150, a=3, minvelerr=None):
        """combination of dispersion curves from SW measurements (Olafsdottir, 2018)

        References
        ----------
        Olafsdottir, E. ´A., Bessason, B. & Erlingsson, S., 2018. Combination of
        dispersion curves from MASW measurements, Soil Dynamics and Earthquake
        Engineering, 113, 473-487.
        """

        # define wavelength intervals
        lam_vec = lam_vec.flatten()
        vel_vec = vel_vec.flatten()

        try:
            qmin = int(np.round((np.log2(lam_min) + 1 / (2 * a)) * a - 1))
        except ValueError:
            qmin = int(np.round((np.log2(1) + 1 / (2 * a)) * a - 1))

        lam_eq, vel_mean, vel_std = [], [], []

        q = qmin
        while True:
            lam_eqi = 2 ** ((q - 1) / a)
            lam_lo = lam_eqi * 2 ** (-1 / (2 * a))
            lam_up = lam_eqi * 2 ** (1 / (2 * a))

            if lam_lo > lam_max:
                break

            idx = (lam_vec >= lam_lo) & (lam_vec <= lam_up)
            if np.count_nonzero(idx) > 1:
                vels = vel_vec[idx]
                vel_mean.append(np.nanmean(vels))
                vel_std.append(np.nanstd(vels))
                lam_eq.append(lam_eqi)

            q += 1

        lam_eq = np.array(lam_eq)
        vel_mean = np.array(vel_mean)
        vel_std = np.array(vel_std)

        if minvelerr is not None:
            vel_std = np.maximum(vel_std, minvelerr)

        f_mean = vel_mean / lam_eq

        return f_mean, vel_mean, vel_std

    def _resample(self, data, pmin = 1, pmax = 100, pn = 30,pspace = 'log', kind = 'cubic'):
        """resample all curves"""

        if pspace == 'log':
            parx_new = np.geomspace(pmin, pmax, pn)
        else:
            parx_new = np.linspace(pmin, pmax, pn)

        resampled_list = []

        for (sin, rep, wid), group in data.groupby(['sin', 'rep', 'wid']):
            group = group.sort_values('frequency')

            # Interpolate velocity and error
            parx_new, pary_new, err_new = interp(group['frequency'],
                                                 group['velocity'],
                                                 parx_new, yerr=group['error'],
                                                 kind=kind,fill_value="extrapolate")

            resampled_list.append(pd.DataFrame({
                                'sin': sin,
                                'rep': rep,
                                'wid': wid,
                                'frequency': parx_new,
                                'velocity': pary_new,
                                'error': err_new}))

        return pd.concat(resampled_list, ignore_index=True)

    def _resample_and_average(self,data,**kwargs):
        """compute the mean of all curves"""

        data_resampled = self._resample(data, **kwargs)

        stats = (data_resampled.groupby("frequency").agg(velocity_mean=("velocity", np.nanmean),
                                               velocity_std=("velocity", np.nanstd)).reset_index())

        return stats['frequency'].to_numpy(), stats['velocity_mean'].to_numpy(), stats['velocity_std'].to_numpy()

    def combination(self, combination_method = 'binning', axes = None, show=True, **kwargs):
        """run combination of dispersion curves"""

        data = self.data
        x, y = data['frequency'].to_numpy(), data['velocity'].to_numpy()

        xlim = kwargs.pop('xlim',[0,80])
        ylim = kwargs.pop('ylim',[0,800])

        # binning
        if combination_method == 'binning':
            lam_vec, vel_vec = wavelength(x,y), y
            lam_min, lam_max = np.min(lam_vec),np.max(lam_vec)
            f_mean, vel_mean, vel_std = self._binning(lam_vec, vel_vec,
                                                    lam_min=lam_min,
                                                    lam_max=lam_max,
                                                    a = kwargs.pop('a',4))
        # resampling
        else:
            f_mean, vel_mean, vel_std = self._resample_and_average(data,
                           pmin = kwargs.pop('pmin',1),
                           pmax = kwargs.pop('pmax',50),
                           pn = kwargs.pop('pn',30),
                           pspace = kwargs.pop('pspace','log'),
                           kind = kwargs.pop('kind','cubic'))

        dc_mean = DispersionCurve()
        dc_mean.init_data(freq=f_mean,vel=vel_mean,err=vel_std)

        # plot
        if show:
            if axes is None:
                fig, ax = plt.subplots(figsize=(6, 4))
            else:
                ax = axes
                fig = ax.figure

            ax.scatter(x, y, color="royalblue", s=20, label = 'data')

            ax.errorbar(x=f_mean, y=vel_mean, yerr = vel_std, capsize = 2,capthick = 1.5,
                            color='k', elinewidth=1.5, label=r'$s_{v_r}$',linewidth = 0)
            ax.scatter(f_mean, vel_mean, marker="s", s=30, c='w', edgecolor='k',
                       linewidth=1.5, zorder=2, label=r'$\overline{v_r}$')

            ax.legend(loc='upper right', edgecolor = 'k', frameon = True)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ax.set_xlabel('frequency (Hz)')
            ax.set_ylabel('phase velocity (m/s)')
            ax.grid(True, linestyle=':')

            plt.locator_params(nbins=4)
            plt.show()

        return dc_mean
