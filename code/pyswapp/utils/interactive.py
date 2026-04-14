from contextlib import suppress

import warnings
import numbers

import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from matplotlib.backend_bases import MouseEvent

from .utils import *

from .physics import lorentzian_err, wavenumber, phase_velocity

import numpy as np
from collections.abc import Iterable

class DraggablePoints:
    def __init__(self, ax, points=None, **kwargs):
        if ax is None:
            raise ValueError("Argument 'ax' must be a valid Matplotlib Axes instance.")

        self.ax = ax
        self.canvas = self.ax.figure.canvas

        self.distance_threshold = 0.1
        self._dragging_point = None
        self._points = {}   # Dict: x -> y
        self._line = None

        self.logger = create_logging(name='Interactive')

        # Extract plotting kwargs
        color = kwargs.pop('color', 'r')
        ls = kwargs.pop('linestyle', '--')
        lw = kwargs.pop('linewidth', 1)
        ms = kwargs.pop('markersize', 7)
        marker = kwargs.pop('marker', 'x')
        self._plot_kwargs = {
            'color': color,
            'linestyle': ls,
            'linewidth': lw,
            'markersize': ms,
            'marker': marker
        }

        self._kwargs = kwargs

        # Initialize points if provided
        if points:
            try:
                for x, y in points.items():
                    self._points[x] = y
            except Exception as e:
                self.logger.warning(f"Failed to load initial points: {e}")

            # remove last line
            try:
                if self.ax.lines:
                    self.ax.lines[-1].remove()
            except Exception:
                pass

            self._update_plot()

        self._init_plot()

    def _init_plot(self):
        """Connect matplotlib events."""
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)

    def _update_plot(self, kwargs=None):
        """Update line data and redraw."""
        if self._line and (self._line in self.ax.lines):
            self._line.remove()
            self._line = None

        try:
            if self._points:
                x, y = zip(*sorted(self._points.items()))
                self._line, = self.ax.plot(x, y, **self._plot_kwargs)
        except Exception as e:
            self.logger.warning(f"Failed to update plot: {e}")

        self.canvas.draw_idle()

    def _add_point(self, x, y=None):
        """add a new point"""
        self._points[x] = y
        return x, y

    def _remove_point(self, x, _):
        """Remove a point by its x coordinate."""
        if x in self._points:
            self._points.pop(x)

    def _find_neighbor_point(self, event):
        """Find point around mouse position."""
        try:
            if not self._points or event.xdata is None or event.ydata is None:
                return None

            points = []
            for x, y in self._points.items():
                try:
                    if isinstance(y, Iterable):
                        y = y[0]
                    points.append((x, y))
                except Exception:
                    continue

            mouse_pos = np.array([event.xdata, event.ydata])

            distances = np.sum((np.asarray(points) - mouse_pos) ** 2, axis=1)
            nearest_idx = np.argmin(distances)

            if distances[nearest_idx] < self.distance_threshold:
                return points[nearest_idx]
        except Exception:
            pass

        return None

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Left click
        if event.button == 1:
            point = self._find_neighbor_point(event)
            if point:
                self._dragging_point = point
            else:
                if event.xdata is not None and event.ydata is not None:
                    self._add_point(event.xdata, event.ydata)
                    self._update_plot()

        # Right click
        elif event.button == 3:
            point = self._find_neighbor_point(event)
            if point:
                self._remove_point(*point)
                self._update_plot()

    def _on_release(self, event):
        if event.button == 1 and self._dragging_point:
            self._dragging_point = None
            self._update_plot()

    def _on_motion(self, event):
        if not self._dragging_point or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        self._remove_point(*self._dragging_point)
        self._dragging_point = self._add_point(event.xdata, event.ydata)
        self._update_plot()

    @property
    def points(self):
        return self._points


class TXInteractive(DraggablePoints):

    def __init__(self, ax, receiver = None, points=None, data  = None, picks = None, **kwargs):

        super().__init__(ax,points,**kwargs)

        self.data = data

        if receiver is None:
            receiver = self.data.receiver

        self.receiver = receiver
        self.x = np.arange(0,len(self.receiver),1)
        self._init_param()
        self.canvas.mpl_connect('key_press_event', self._on_key)

    def _init_param(self):

        self._key = 't'
        self._terminate = False
        self._reset = False
        self._reset_last = False
        self.slope = None
        self.tbox = None
        self.plot_slope = None

    def _add_point(self, x, y=None):

        # snap to data
        indx = np.searchsorted(self.x, [x])[0]

        try:
            x = self.x[indx]
        except IndexError:
            if indx > 0:
                x = self.x[indx-1]
            else:
                x = self.x[0]

        if len(self._points) == 2:
            id = sorted(self._points.keys())

            lxo = id[0]
            rxo = id[1]
            if x <= lxo:
                self._points.pop(lxo)
            else:
                self._points.pop(rxo)

        self._points[x] = y
        return x, y

    def _estimate_velocity(self):
        """calculate the slope"""

        if (len(self._points) == 2) & (self.receiver is not None):
            idx, t = zip(*sorted(self._points.items()))

            x1 = self.receiver[int(idx[0])]
            x2 = self.receiver[int(idx[1])]

            t1 = t[0]
            t2 = t[1]

            self.slope = (t2-t1)/(x2-x1)

            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            self.tbox = self.ax.text(np.mean(idx), np.mean(t),f'v={round(1/self.slope,2)} m/s', bbox = props)

    def _on_key(self, event):

        if event.key == 'enter':
            plt.close()

        if event.key == 'e':
            #print(f'You pressed {event.key}. Process stopped.')
            self._terminate = True
            plt.close()

        if event.key == 'r':
            #print(f'You pressed {event.key}. Reset mute')
            if self.tbox is not None:
                self.tbox.remove()

            self._reset = True
            plt.close()

        if event.key == 'ctrl+z':
            #print(f'You pressed {event.key}. Reset last step')
            self._reset_last = True
            plt.close()

        if event.key == 't':
            #print(f'You pressed {event.key}.')
            if self.tbox is not None:
                with suppress(ValueError):
                    self.tbox.remove()

            self._key = 't'

            self.ax.set_title('Top mute active', fontweight='bold')
            self.canvas.draw_idle()

        if event.key == 'b':
            #print(f'You pressed {event.key}.')
            if self.tbox is not None:
                with suppress(ValueError):
                    self.tbox.remove()

            self._key = 'b'

            self.ax.set_title('Bottom mute active', fontweight='bold')
            self.canvas.draw_idle()

        if event.key == 'v':
            #print(f'You pressed {event.key}.')
            if self.tbox is not None:
                with suppress(ValueError):
                    self.tbox.remove()

            self._estimate_velocity()
            self.ax.set_title('Velocity estimation', fontweight='bold')
            self.canvas.draw_idle()

    def update_plot(self):

        if self._points:

            # reset plot
            self._points = {}
            self._update_plot()

    # def filter(self):
    #
    #     kwargs = self._kwargs
    #     taper_type = kwargs.pop('taper_type', 'tukey')
    #     taper = kwargs.pop('tapering', 'mild')
    #
    #     if self._points:
    #         self.data.linear_mute(self._points, key=self._key, taper = taper, taper_type = taper_type)
    #
    #         # reset plot
    #         self._points = {}
    #         self._update_plot()
    #
    #     return self.data

    @property
    def key(self):
        return self._key

    @property
    def terminate(self):
        return self._terminate

    @property
    def reset(self):
        return self._reset

    @property
    def reset_last(self):
        return self._reset_last

    @property
    def picks(self):
        return self._points


class FKFilterInteractive(DraggablePoints):

    def __init__(self, ax, points=None, data  = None, picks = None, **kwargs):

        super().__init__(ax, points, **kwargs)

        self.data = data
        self._init_param()
        self.canvas.mpl_connect('key_press_event', self._on_key)

    def _init_param(self):
        """initialize some parameters"""

        self._key = 't'
        self._terminate = False

    def update_plot(self):

        if self._points:

            # reset plot
            self._points = {}
            self._update_plot()

    def _on_key(self, event):
        """keyboard events"""
        if event.key == 'e':
            self._terminate = True
            plt.close()

        if event.key == 't':
            self._key = 't'
            self.ax.set_title('Top filter active', fontweight='bold') # change to textbox
            self.canvas.draw_idle()

        if event.key == 'b':
            self._key = 'b'
            self.ax.set_title('Bottom filter active', fontweight='bold') # change to textbox
            self.canvas.draw_idle()


    @property
    def key(self):
        return self._key

    @property
    def terminate(self):
        return self._terminate

    @property
    def picks(self):
        return self._points


class DCPickingInteractive(DraggablePoints):
    """class for drawing boundaries for dispersion curve extraction"""

    def __init__(self, ax, points=None,
                 data=None, freq=None, vel=None, power=None, offsets=None, picks=None, **kwargs):

        super().__init__(ax)

        self.domain = kwargs.pop('domain', 'FV')

        if self.domain == 'FV':
            self._err = 'lor'
        else:
            self._err = 5

        # load data
        try:
            if data:
                if self.domain == 'FK':
                    self._ydata = data.FK_data['freq']
                    self._xdata = data.FK_data['kw']
                    self._power = data.FK_data['FK_abs']
                else:
                    self._xdata = getattr(data, 'frequency', None)
                    self._ydata = getattr(data, 'velocity', None)
                    self._power = getattr(data, 'dispersive_energy', None)
                self._offsets = getattr(data, 'offset', None)
            else:
                self._xdata = freq
                self._ydata = vel
                self._power = power
                self._offsets = offsets
        except Exception as e:
            self.logger.warning(f"Failed to extract data attributes: {e}")
            self._xdata = self._ydata = self._power = self._offsets = None

        self._init_param()
        self._init_plot()

        # Load picks if provided
        if picks is not None:
            self._picks = picks
            self._update_plot(**kwargs)

        elif points:
            for x, y in points.items():
                self._points[x] = y

            # Line removal
            try:
                for line in list(ax.lines):
                    line.remove()
            except Exception:
                pass

            self._update_polygons()
            self._update_plot()

    def _init_plot(self):
        """event connections"""
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('key_press_event', self._on_key)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _init_param(self):
        """initialize parameters"""
        self._reset = False
        self._mode = 0
        self._polygons = {}
        self._picks = {}
        self._ppid = 0
        self._picks_prior = {}

        self._boundary_strength = 0.3
        self._minVelErr = 20
        self._boundary_min = self._minVelErr
        self._points_prior = {}

        self._pick_lines = {}
        self._line = None
        self._upper_bound_line = None
        self._lower_bound_line = None

    def _update_plot(self, **kwargs):
        """update plot after event"""

        try:
            show_legend = kwargs.pop('show_legend', True)
            marker_size = kwargs.pop('markersize', 3)
            alpha = kwargs.pop('alpha', 1)
        except Exception:
            show_legend = True
            marker_size = 3
            alpha = 1

        try:
            # case 1: no boundary drawn → show picks
            if not self._points:
                cmap = getattr(plt.cm, 'Greys', plt.cm.viridis)
                color = cmap(np.linspace(0, 1, 10))

                if self._mode in self._pick_lines:
                    try:
                        self._pick_lines[self._mode].remove()
                    except Exception:
                        pass
                    self._pick_lines.pop(self._mode, None)

                    try:
                        if self._pick_lines:
                            self.ax.legend(loc='upper right')
                        else:
                            lg = self.ax.get_legend()
                            if lg:
                                lg.remove()
                    except Exception:
                        pass

                if self._line:
                    try:
                        self._line.set_data([], [])
                    except Exception:
                        pass
                if self._upper_bound_line:
                    try:
                        self._upper_bound_line.set_data([], [])
                    except Exception:
                        pass
                if self._lower_bound_line:
                    try:
                        self._lower_bound_line.set_data([], [])
                    except Exception:
                        pass

                # draw picks
                for self._mode in self._picks:
                    if self._mode not in self._pick_lines and not self._reset:
                        try:
                            picks = self._picks[self._mode]
                            f = picks['frequency']
                            v = picks['velocity']

                            if self.domain == 'FK':
                                x_data = wavenumber(f,v)
                                y_data = f
                            else:
                                x_data = f
                                y_data = v

                            pick_line, = self.ax.plot(
                                x_data,
                                y_data,
                                marker="s",
                                markersize=marker_size,
                                markeredgecolor='k',
                                color=color[self._mode],
                                linewidth=0,
                                alpha=alpha,
                                label=f'Mode {self._mode}'
                            )
                            if show_legend:
                                self.ax.legend(loc='upper right')
                            self._pick_lines[self._mode] = pick_line
                        except Exception as e:
                            self.logger.warning(f"Failed plotting picks: {e}")

            # case 2: user is drawing boundaries
            else:
                try:
                    x, raw_y = zip(*sorted(self._points.items()))
                except Exception:
                    return

                try:
                    y_data = [val[0] for val in raw_y]
                    y_lower = [val[2] for val in raw_y]
                    y_upper = [val[1] for val in raw_y]
                except Exception:
                    return

                try:
                    if self._line is None:
                        self._line, = self.ax.plot(
                            x, y_data,
                            marker="o", markersize=7,
                            markeredgecolor='k',
                            color='white',
                            linewidth=0
                        )
                        marker_style = dict(color="r", marker="o",
                                            markersize=3, linestyle='--', linewidth=0.8)
                        self._upper_bound_line, = self.ax.plot(x, y_lower, **marker_style)
                        self._lower_bound_line, = self.ax.plot(x, y_upper, **marker_style)
                    else:
                        self._line.set_data(x, y_data)
                        self._upper_bound_line.set_data(x, y_lower)
                        self._lower_bound_line.set_data(x, y_upper)

                except Exception as e:
                    self.logger.warning(f"Failed updating boundary plot: {e}")

            self.canvas.draw_idle()
        except Exception as e:
            self.logger.error(f"Error in _update_plot: {e}")

    def _add_point(self, x, y=None):
        """add new point with upper/lower boundaries"""
        try:
            if isinstance(x, MouseEvent):
                x, y = float(x.xdata), float(x.ydata)
        except Exception:
            return None, None

        if (self._err == 'lor') and (self._offsets is not None):
            err = lorentzian_err(self._offsets, y, x)
        elif isinstance(self._err, numbers.Number):
            err = self._err
        else:
            err = 10

        self._points[x] = [y, y + err, y - err]

        return x, y

    def _update_bounds(self):
        try:
            xs, ys = zip(*sorted(self._points.items()))
        except Exception:
            return

        for i in range(len(ys)):
            x = xs[i]
            y = ys[i][0]

            if (self._err == 'lor') and (self._offsets is not None):
                err = lorentzian_err(
                    self._offsets, y, x,
                    a=self._boundary_strength,
                    minvelerr=self._boundary_min
                )
            elif isinstance(self._err, numbers.Number):
                err = self._err * self._boundary_strength
            else:
                err = 10 * self._boundary_strength

            self._points[x] = [y, y + err, y - err]

    def _update_polygons(self):
        """update polygons after event"""
        x, ys = zip(*sorted(self._points.items()))
        y_lo = [p[2] for p in ys]
        y_up = [p[1] for p in ys]

        bound_lo = np.vstack((np.array(x), np.array(y_lo))).T
        bound_up = np.flipud(np.vstack((np.array(x), np.array(y_up))).T)

        self._polygons = np.vstack((bound_up, bound_lo))

    def _extract_dccurve(self):
        """extract dispersion curve within boundary"""
        if self._xdata is None or self._ydata is None or self._power is None:
            self.logger.warning("Frequency, velocity, and power must be defined.")
            return

        fgrid, vgrid = np.meshgrid(self._xdata, self._ydata, indexing='xy')
        path = mpltPath.Path(self._polygons)

        # Test points inside polygon
        points = np.column_stack((fgrid.ravel(), vgrid.ravel()))
        flags = path.contains_points(points).reshape(self._power.shape)

        if not flags.any():
            return

        masked_power = np.where(flags, self._power, -np.inf)
        peaks_idx = masked_power.argmax(axis=0)
        cols = np.flatnonzero(flags.any(axis=0))

        x_pick = self._xdata[cols]
        y_pick = self._ydata[peaks_idx[cols]]

        if self.domain == 'FK':
            freq_pick, uniq_id = np.unique(y_pick, return_index=True)
            vel_pick = phase_velocity(y_pick,x_pick)[uniq_id]
        else:
            freq_pick = x_pick
            vel_pick = y_pick

        self._picks[self._mode] = {'frequency': freq_pick, 'velocity': vel_pick}
        self._picks_prior[self._ppid] = {'frequency': freq_pick, 'velocity': vel_pick}

        self._ppid = (self._ppid + 1) % 5

    def interact(self):
        """extract dispersion curve within boundary"""
        if self._points:
            self._extract_dccurve()
            self._points = {}
            self._update_plot()

    def _on_click(self, event):
        if event.button == 1 and event.inaxes in [self.ax]:
            point = self._find_neighbor_point(event)
            if point:
                self._dragging_point = point
            else:
                self._add_point(event)
            self._update_plot()

            if len(self._points) > 1:
                self._update_polygons()

        elif event.button == 3 and event.inaxes in [self.ax]:
            point = self._find_neighbor_point(event)
            if point:
                self._remove_point(*point)
                self._update_plot()

    def _on_scroll(self, event):
        if len(self._points) > 0:
            increment_strength = 0.05
            increment_min = 0.5

            if event.button == 'up':
                if self._boundary_strength <= 1:
                    self._boundary_strength += increment_strength
                elif self._boundary_min > 1 + increment_min:
                    self._boundary_min -= increment_min
            else:
                if self._boundary_strength > increment_strength:
                    self._boundary_strength -= increment_strength
                if self._boundary_min < self._minVelErr:
                    self._boundary_min += increment_min

            self._update_bounds()
            self._update_polygons()
            self._update_plot()

    def _on_key(self, event):

        if event.key.isnumeric():
            mode = int(event.key)
            if self._mode != mode:
                self._mode = mode

        if event.key == 'p':
            if self._points:
                self._extract_dccurve()
                self._points = {}
                self._update_plot()

        if event.key == 'r':
            self._reset = True
            if self._mode in self._picks:
                self._picks.pop(self._mode)
            self._polygons = {}
            self._points = {}
            self._update_plot()
            self._reset = False

        if event.key == 'd':
            self._points = {}
            self._update_plot()

        if event.key == 'e':
            self._terminate = True
            plt.close()

    @property
    def polygons(self):
        return self._polygons

    @property
    def picks(self):
        return self._picks


