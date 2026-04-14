import copy
import time
import sys

from .utils.utils import *
from .utils.interactive import *
from .utils.sql import *
from .curve import DispersionCurve
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
                             QLabel, QComboBox,QMessageBox,QStyle)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

class FigureSwitcher(QMainWindow):
    """User interface to view the seismic data"""

    def __init__(self, figures):
        super().__init__()
        self.setWindowTitle("Data Preview Mode")

        self.figures = figures
        self.current_index = 0

        self.init_ui()

    def init_ui(self):
        """interface design"""

        # Canvas for figure
        self.canvas = FigureCanvas(self.figures[self.current_index])

        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Layouts
        layout = QVBoxLayout()
        nav_layout = QHBoxLayout()
        layout.addLayout(nav_layout)
        central_widget.setLayout(layout)

        # Navigation buttons
        self.left_btn = QPushButton()
        self.left_btn.setIcon(self.style().standardIcon(self.style().SP_ArrowLeft))
        self.left_btn.setFixedSize(40, 40)

        self.right_btn = QPushButton()
        self.right_btn.setIcon(self.style().standardIcon(self.style().SP_ArrowRight))
        self.right_btn.setFixedSize(40, 40)

        nav_layout.addWidget(self.left_btn)
        nav_layout.addWidget(self.right_btn)
        nav_layout.addStretch()

        layout.addWidget(self.toolbar)

        self.left_btn.clicked.connect(self.show_previous_figure)
        self.right_btn.clicked.connect(self.show_next_figure)

        layout.addWidget(self.canvas)

    def show_previous_figure(self):
        """Switch to previous figure"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_figure()
        else:
            self.current_index = len(self.figures) - 1
            self.update_figure()

    def show_next_figure(self):
        """Switch to next figure"""
        if self.current_index < len(self.figures) - 1:
            self.current_index += 1
            self.update_figure()
        else:
            self.current_index = 0
            self.update_figure()

    def update_figure(self):
        """Update figure"""
        self.canvas.setParent(None)
        self.canvas = FigureCanvas(self.figures[self.current_index])
        self.centralWidget().layout().addWidget(self.canvas)


class DualDataSwitcher(QMainWindow):
    """User interface to display raw data and select subset of data (e.g., based on windowing) side by side"""
    def __init__(self, data, sql, plot = 'geom', DataSwitcher = None, procset = None,
                 procsets = None, window_title = "SWA Viewer",select_plot = True, **kwargs):
        super().__init__()

        self.setWindowTitle(window_title)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        if not DataSwitcher:
            DataSwitcher = DataSwitcherBase

        # Viewer 1: controls the index
        self.viewer1 = DataSwitcherBase(data, sql, plot = '', procset = procset, procsets = procsets,
                                        select_plot = select_plot, **kwargs)
        self.viewer1.setFixedSize(400, 600)

        # Viewer 2: shows groups of figures based on viewer1's index
        self.viewer2 = DataSwitcher(data, sql, plot = plot, use_windows=True, procset = procset, procsets = procsets,
                                        select_plot = select_plot,
                                    window_label = 'WINDOWS',**kwargs)
        self.viewer2.setFixedSize(800, 600)

        layout.addWidget(self.viewer1)
        layout.addWidget(self.viewer2)

        # Link left to right
        self.viewer1.index_changed.connect(self.viewer2.set_group)
        self.viewer1.procset_changed.connect(self.viewer2.set_procset)
        self.viewer1.plot_changed.connect(self.viewer2.set_plot)

    def clean(self):
        self.viewer1.clean()
        self.viewer2.clean()

    def closeEvent(self, event):
        self.viewer1.closeEvent(event)
        self.viewer2.closeEvent(event)


class DataSwitcherBase(QWidget):
    """Base user interface to manipulate seismic dataset"""

    index_changed = pyqtSignal(int)
    procset_changed = pyqtSignal(str)
    plot_changed = pyqtSignal(str)
    index_pair_changed = pyqtSignal(int, str)

    def __init__(self, data, sql, plot='TX',
                 use_windows=False, interaction_class=None,
                 procset=None, procsets=None,
                 btn_label='Click me',
                 window_label='SHOTFILES',
                 window_title='SWA Viewer',
                 select_plot=True,
                 **kwargs):
        super().__init__()

        self.logger = create_logging(name='GUI')

        # basic sanity check
        if data is None or not isinstance(data, dict):
            raise ValueError("data must be a dict of dicts (data[sin][rep]).")
        self.data = data

        # Select default plot if None
        self.plot = plot or 'TX'
        self.plots = ['TX', 'FX', 'spectra', 'SFR', 'FK', 'FV', 'DC']
        self.select_plot = select_plot

        self.stream = None
        self.canvas0 = FigureCanvas()
        self.canvas = FigureCanvas()

        self.setWindowTitle(window_title)
        self.window_label = window_label
        self._sql = SQL(database=sql)

        self.active_label = 'PLOT INACTIVE'

        # Interaction mode
        if interaction_class:
            procsets = list(procsets)
            procsets = [p for p in procsets if p != 'raw']
            self.active_label = 'PLOT ACTIVE'

        if procsets is None:
            raise ValueError("procsets must not be empty")
        self.procsets = procsets

        if procset not in procsets:
            procset = procsets[-1]
        self.procset = procset

        self.is_grouped = use_windows
        self.all_labels = self._get_all_labels()

        # warn if no labels found
        if not self.all_labels:
            self.logger.warning("No data labels found. GUI will be empty.")

        self.interaction_class = interaction_class

        self.group_index = 0
        self.current_index = 0
        self.interactor = None
        self.points = {}
        self.picks = {}

        # Available processing methods
        self.grouped_methods = {}
        for p in procsets:
            try:
                self.grouped_methods[p] = list(self._sql.get_trafo_labels(p, use_windows))
            except Exception as err:
                self.logger.error(f"Error retrieving trafo labels for '{p}': {err}")
                self.grouped_methods[p] = []

        self.methods = self.grouped_methods.get(procset, [])
        if not self.methods:
            self.method = "phaseshift"
            self.methods = [self.method, 'FK']
        else:
            self.methods.append('FK')
            self.method = self.methods[0]

        self.labels = self._get_current_labels()
        self.btn_label = btn_label

        self.kwargs = kwargs

        self._interact_kwargs = {
            'domain': 'FV',
            'taper_type': self.kwargs.pop('taper_type', 'tukey'),
            'tapering': self.kwargs.pop('tapering', 'mild'),
            'taper_length': self.kwargs.pop('taper_length', 5),
        }

        # Build UI
        try:
            self.init_ui()
        except Exception as err:
            self.logger.error(f"UI initialization failed: {err}")

        # Initial display
        try:
            self.update_display()
        except Exception as err:
            self.logger.error(f"Initial display update failed: {err}")

        self.canvas.setFocus()

    def init_ui(self):
        """interface design"""

        # canvas
        if self.labels:
            try:
                figure0 = self.create_figure(plot='geomShort')
                self.canvas0 = FigureCanvas(figure0) if figure0 else self.canvas0

                figure = self.create_figure()
                self.canvas = FigureCanvas(figure) if figure else self.canvas
            except Exception as err:
                self.logger.error(f"Error creating initial figures: {err}")

        # layout
        self.layout = QVBoxLayout(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setParent(self.canvas)

        # info bar
        self.info_layout = QHBoxLayout()
        label = QLabel(self.window_label)
        label.setStyleSheet("font-size: 14px; color: gray;")
        self.info_layout.addWidget(label)

        self.label = QLabel()
        self.label.setStyleSheet("font-size: 14px; color: gray;")
        self.info_layout.addStretch()
        self.info_layout.addWidget(self.label)
        self.layout.addLayout(self.info_layout)

        # Navigation buttons
        self.nav_layout = QHBoxLayout()
        self.left_btn = QPushButton()
        self.left_btn.setIcon(self.style().standardIcon(self.style().SP_ArrowLeft))
        self.left_btn.setFixedSize(40, 40)

        self.right_btn = QPushButton()
        self.right_btn.setIcon(self.style().standardIcon(self.style().SP_ArrowRight))
        self.right_btn.setFixedSize(40, 40)

        self.nav_layout.addWidget(self.left_btn)
        self.nav_layout.addWidget(self.right_btn)

        # Dropdowns
        if not self.is_grouped:
            try:
                self.combo = self.add_combobox(self.procsets, self.procset)
                self.combo.currentTextChanged.connect(self.set_procset)
                self.nav_layout.addWidget(self.combo)
            except Exception as err:
                self.logger.error(f"Error adding procset combobox: {err}")

        if not self.is_grouped and self.select_plot:
            try:
                self.combo_plots = self.add_combobox(self.plots, self.plot)
                self.combo_plots.currentTextChanged.connect(self.set_plot)
                self.nav_layout.addWidget(self.combo_plots)
            except Exception as err:
                self.logger.error(f"Error adding plot combobox: {err}")

        if not self.is_grouped:
            try:
                sins = np.unique(np.asarray(self.labels)[:, 0])
                sins = np.char.mod('%d', sins)
                self.combo_select_sin = self.add_combobox(
                    sins, str(self.current_index + 1), 'SIN:', 50
                )
                self.combo_select_sin.currentTextChanged.connect(self.set_index)
                self.nav_layout.addWidget(self.combo_select_sin)
            except Exception as err:
                self.logger.error(f"Error adding SIN combobox: {err}")

        # Interaction button
        if self.interaction_class:
            self.interact_btn = QPushButton(self.btn_label)
            self.interact_btn.setFixedSize(100, 40)
            self.interact_btn.clicked.connect(self.interact)
            self.nav_layout.addWidget(self.interact_btn)

        self.nav_layout.addStretch()
        self.layout.addLayout(self.nav_layout)

        # Toolbar and canvases
        self.left_btn.clicked.connect(self.show_previous_figure)
        self.right_btn.clicked.connect(self.show_next_figure)

        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas0)
        self.layout.addWidget(self.canvas)

    def _write_data(self, data, sin, rep, procset, wid=-1):
        self._sql.write_data(data, sin, rep, procset, wid)

    def _write_FV(self, data, sin, rep, procset, wid=-1):
        self._sql.write_FV(data, sin, rep, procset, wid)

    def _get_data(self, sin, rep, procset, wid=-1):
        try:
            return self._sql.read_data(sin, rep, procset=procset, wid=wid)
        except Exception as err:
            self.logger.error(f"_get_data failed: {err}")
            return None, None, None, None

    def _get_curve(self, sin, rep, procset, wid=-1, method='phaseshift'):
        """get the curve data"""
        params = {'procset': "'%s'" % procset, 'method': "'%s'" % method,
                  'sin': sin, 'rep': rep, 'wid': wid}
        return self._sql.read_curve(params)

    def _get_FV(self, sin, rep, procset, wid=-1, method='phaseshift'):
        try:
            return self._sql.read_FV(sin, rep, procset=procset, wid=wid, method=method)
        except Exception as err:
            self.logger.error(f"_get_FV failed: {err}")
            return None, None, None, None

    def _set_data(self, data, sin, rep, procset, wid=-1):
        """set processed data from database to current stream"""

        par, amps, recs, sht = self._get_data(sin, rep, procset=procset, wid=wid)

        if amps is not None:
            amps_ari = amps.transpose()
            data.update_pst(amps_ari, sht, recs, par)
            return True
        else:
            #self.logger.warning('No data.')
            return False

    def _set_FV(self, data, sin, rep, procset, wid=-1, method='phaseshift'):
        """set FV data from database to current stream"""

        vel, kw, freq, FV = self._get_FV(sin, rep, procset=procset, wid=wid, method=method)
        if FV is not None:
            data.update_FV(method, vel, kw, freq, FV)
            return True
        else:
            _, amps, _, _ = self._get_data(sin, rep, procset=procset, wid=wid)
            if amps is not None:
                self.logger.warning(f'Wave-field transformation not yet performed. Running {method} transformation.')
                data.transform(method = method)
                self._write_FV(data, sin, rep, procset, wid=wid)
                return True
            return False

    def create_figure(self, plot=None, procset=None, **kwargs):
        """Safely create the requested figure."""

        if not self.labels:
            return None

        procset = procset or self.procset
        plot = plot or self.plot

        try:
            label = self.labels[self.current_index]
        except Exception:
            #self.logger.error("Invalid current_index in create_figure.")
            return None

        try:
            self.stream = self.select_data(label[0], label[1])
            self.data_exists = self._set_data(self.stream, label[0], label[1], procset, label[2])
        except Exception as err:
            self.logger.error(f"Error setting data: {err}")
            return None

        # FV plots
        if plot in ['dispersionImage', 'dispersionImageComposite', 'FV', 'FVComposite']:
            try:
                FV_flag = self._set_FV(self.stream, label[0], label[1], procset,
                                       method=self.method, wid=label[2])
                self.data_exists = self.data_exists and FV_flag
            except Exception as err:
                self.logger.error(f"Error in FV setup: {err}")
                return None

        # DC plot
        if plot == 'DC':
            try:
                curves = self._get_curve(label[0], label[1], procset, wid=label[2],
                                         method=kwargs.pop('method', 'phaseshift'))
                if curves is None or curves.empty:
                    return None

                dc_modes = curves['dc_mode'].unique()
                cmap = getattr(plt.cm, 'Greys')
                color = cmap(np.linspace(0, 1, 10))

                fig = Figure(figsize=(8, 8), constrained_layout=True)
                ax = fig.add_subplot(111)

                for dc_mode in dc_modes:
                    curve = curves[curves['dc_mode'] == dc_mode]
                    dc = DispersionCurve()
                    dc.init_data(curve['frequency'], curve['velocity'], curve['error'])
                    ax = dc.plot(show=False, axes=ax, color=color[dc_mode],
                                 edgecolor='k', label=f'Mode {dc_mode}', **kwargs)
                return ax.figure
            except Exception as err:
                self.logger.error(f"DC plot failed: {err}")
                return None

        # Default stream plot
        if self.data_exists:
            try:
                return self.stream.plot(plot, show=False, gui=True, **self.kwargs, **kwargs)
            except Exception as err:
                self.logger.error(f"Error during stream plot: {err}")
                return None

        return None

    def update_display(self):
        """update the display"""

        # Save points from previous view
        if self.interactor and self.interactor.points:
            self.points[self.current_index] = self.interactor.points

        num_figures = len(self.labels)
        nav_enabled = num_figures > 1
        self.left_btn.setEnabled(nav_enabled)
        self.right_btn.setEnabled(nav_enabled)

        try:
            figure0 = self.create_figure(plot='geomShort')
            figure = self.create_figure()
        except Exception as err:
            self.logger.error(f"update_display figure creation failed: {err}")
            return

        # Update label
        try:
            if not self.labels:
                self.label.setText("No data")
            else:
                label = self.labels[self.current_index]
                if not self.is_grouped:
                    self.label.setText(
                        f"SIN {label[0]} | REP {label[1]}"
                        if figure0 else f"SIN {label[0]} | REP {label[1]}"
                    )
                else:
                    self.label.setText(
                        f"SIN {label[0]} | REP {label[1]} | WIN {label[2]+1}"
                        if figure0 else "No data"
                    )
        except Exception as err:
            self.logger.error(f"Error updating info label: {err}")

        # No data → placeholder
        if not figure0:
            self.canvas0 = self.create_placeholder(self.canvas0)
            self.canvas = self.create_placeholder(self.canvas)
            return

        # Replace canvases
        try:
            self.canvas0 = self.replace_widget(self.canvas0, figure0)
            self.canvas = self.replace_widget(self.canvas, figure)
        except Exception as err:
            self.logger.error(f"Canvas replacement failed: {err}")
            return

        # Reset toolbar
        try:
            self.layout.removeWidget(self.toolbar)
            self.toolbar.setParent(None)
            self.toolbar.deleteLater()
            self.toolbar = NavigationToolbar(self.canvas, self.canvas)
            self.layout.addWidget(self.toolbar)
        except Exception as err:
            self.logger.error(f"Toolbar update failed: {err}")

        # Add canvases
        self.layout.addWidget(self.canvas0)
        self.canvas.setEnabled(self.data_exists)
        self.layout.addWidget(self.canvas)

        # Interaction setup
        self.interactor = None
        if self.interaction_class and self.data_exists:
            try:
                if not figure.axes:
                    return
                ax = figure.axes[0]

                points = self.points.get(self.current_index, {})
                picks = self.get_picks()

                self.interactor = self.interaction_class(
                    ax, points=points, data=self.stream, picks=picks, **self._interact_kwargs
                )
                self.points[self.current_index] = self.interactor.points
                self.picks[self.current_index] = self.interactor.picks
            except Exception as err:
                self.logger.error(f"Interaction setup failed: {err}")

    def add_combobox(self, sets, set, label=None, size = 90):
        """Add a combo box"""

        combo = QComboBox()
        combo.addItems(sets)
        combo.setFixedSize(size, 40)

        if len(sets) > 0:
            try:
                combo.setCurrentIndex(list(sets).index(set))
            except ValueError:
                combo.setCurrentIndex(sets[0])

        #return dropdown_label, combo
        return combo

    def show_previous_figure(self):
        if not self.labels:
            return
        try:
            self.current_index = (self.current_index - 1) % len(self.labels)
            self.update_display()
            self.canvas.setFocus()

            if not self.is_grouped:
                self.index_changed.emit(self.current_index)
                self.combo_select_sin.blockSignals(True)
                self.combo_select_sin.setCurrentIndex(self.current_index)
                self.combo_select_sin.blockSignals(False)
        except Exception as err:
            self.logger.error(f"show_previous_figure failed: {err}")

    def show_next_figure(self):
        if not self.labels:
            return
        try:
            self.current_index = (self.current_index + 1) % len(self.labels)
            self.update_display()
            self.canvas.setFocus()

            if not self.is_grouped:
                self.index_changed.emit(self.current_index)
                self.combo_select_sin.blockSignals(True)
                self.combo_select_sin.setCurrentIndex(self.current_index)
                self.combo_select_sin.blockSignals(False)
        except Exception as err:
            self.logger.error(f"show_next_figure failed: {err}")

    def interact(self):
        return None

    def set_group(self, group_index):
        if not self.is_grouped:
            return
        if group_index < 0 or group_index >= len(self.all_labels):
            return

        self.group_index = group_index
        self.current_index = 0
        self.labels = self._get_current_labels()

        try:
            self.update_display()
            self.canvas.setFocus()
        except Exception as err:
            self.logger.error(f"set_group failed: {err}")

    def set_procset(self, procset):
        self.current_index = 0
        self.procset = procset
        self.procset_changed.emit(procset)

        try:
            self.all_labels = self._get_all_labels()
            self.labels = self._get_current_labels()
            self.update_display()

            if not self.is_grouped:
                self.combo_select_sin.blockSignals(True)
                self.combo_select_sin.setCurrentIndex(self.current_index)
                self.combo_select_sin.blockSignals(False)
        except Exception as err:
            self.logger.error(f"set_procset failed: {err}")

    def set_plot(self, plot):
        self.plot = plot
        self.plot_changed.emit(plot)
        try:
            self.update_display()
        except Exception as err:
            self.logger.error(f"set_plot failed: {err}")

    def set_index(self, current_index):
        try:
            _, indices = np.unique(np.asarray(self.labels)[:, 0], return_index=True)
            idx = int(current_index) - 1
            if idx < 0 or idx >= len(indices):
                return
            self.current_index = indices[idx]
            self.index_changed.emit(self.current_index)
            self.update_display()
        except Exception as err:
            self.logger.error(f"set_index failed: {err}")

    def create_placeholder(self, canvas):
        """create canvas placeholder"""
        placeholder = QWidget()
        placeholder.setFixedSize(canvas.size())
        self.layout.replaceWidget(canvas, placeholder)
        canvas.setParent(None)
        canvas.deleteLater()
        canvas = placeholder

        return canvas

    def replace_widget(self, canvas, figure):
        """remove widget"""
        self.layout.removeWidget(canvas)
        canvas.setParent(None)
        canvas.deleteLater()
        canvas = FigureCanvas(figure)
        return canvas

    # data selection
    def select_data(self, sin=1, rep=1):
        """Select one stream object based on source location and shot repetition indices"""

        data = copy.deepcopy(self.data)

        if sin in data.keys():
            if rep in data[sin].keys():
                return data[sin][rep]

        return None

    def _get_all_labels(self):
        """get all data indices for data selection"""

        labels = []

        if self.is_grouped:
            for sin in self.data.keys():
                for rep in self.data[sin].keys():
                    wids = sorted(self._sql.get_wids(sin, rep, self.procset))
                    labels.append([[sin,rep,x] for x in wids])
        else:
            for sin in self.data.keys():
                for rep in self.data[sin].keys():
                    labels.append([sin,rep,-1])

        return labels

    def _get_current_labels(self):
        """return data indices of current data"""
        return self.all_labels if not self.is_grouped else self.all_labels[self.group_index]

    def clean(self):
        pass

    def closeEvent(self, event):
        try:
            if hasattr(self, 'canvas'):
                self.canvas.setParent(None)
                self.canvas.close()
                del self.canvas
            plt.close('all')
            event.accept()
        except Exception as err:
            self.logger.error(f"closeEvent error: {err}")
            event.accept()


class DataSwitcherPick(DataSwitcherBase):
    """Manual dispersion curve picking interface"""

    def __init__(self, data, sql, plot='FV', use_windows=False,
                 procset=None, procsets=None, select_plot=True, **kwargs):

        interaction_class = DCPickingInteractive

        super().__init__(
            data, sql, plot=plot, use_windows=use_windows,
            interaction_class=interaction_class,
            procset=procset, procsets=procsets,
            btn_label='Extract Curve',
            select_plot=select_plot,
            **kwargs
        )

    def init_ui(self):
        """Builds the UI. Mostly identical to original, but safer & cleaner."""

        try:
            self.layout = QVBoxLayout(self)
            self.toolbar = NavigationToolbar(self.canvas, self)

            self.info_layout = QHBoxLayout()

            label = QLabel(self.window_label)
            label.setStyleSheet("font-size: 14px; color: gray;")
            self.info_layout.addWidget(label)

            self.label = QLabel()
            self.label.setStyleSheet("font-size: 14px; color: gray;")

            self.info_layout.addStretch()
            self.info_layout.addWidget(self.label)
            self.layout.addLayout(self.info_layout)

            self.nav_layout = QHBoxLayout()

            self.left_btn = QPushButton()
            self.left_btn.setIcon(self.style().standardIcon(self.style().SP_ArrowLeft))
            self.left_btn.setFixedSize(40, 40)

            self.right_btn = QPushButton()
            self.right_btn.setIcon(self.style().standardIcon(self.style().SP_ArrowRight))
            self.right_btn.setFixedSize(40, 40)

            self.nav_layout.addWidget(self.left_btn)
            self.nav_layout.addWidget(self.right_btn)

            if not self.is_grouped:
                self.combo = self.add_combobox(self.procsets, self.procset)
                self.combo.currentTextChanged.connect(self.set_procset)
                self.nav_layout.addWidget(self.combo)

            if self.procset is not None:
                self.combo2 = self.add_combobox(self.methods, self.method)
                self.combo2.currentTextChanged.connect(self.set_method)
                self.nav_layout.addWidget(self.combo2)

            if not self.is_grouped:
                _, indices = np.unique(np.asarray(self.labels)[:, 0], return_index=True)
                indices = np.char.mod('%d', indices + 1)

                self.combo_select_sin = self.add_combobox(
                    indices,
                    str(self.current_index + 1),
                    'SIN:',
                    50
                )
                self.combo_select_sin.currentTextChanged.connect(self.set_index)
                self.nav_layout.addWidget(self.combo_select_sin)

            if self.interaction_class:
                self.interact_btn = QPushButton(self.btn_label)
                self.interact_btn.setFixedSize(100, 40)
                self.nav_layout.addWidget(self.interact_btn)

            self.popup_btn = QPushButton()
            icon = QApplication.style().standardIcon(QStyle.SP_MessageBoxInformation)
            self.popup_btn.setIcon(icon)
            self.popup_btn.clicked.connect(self.show_popup)
            self.popup_btn.setFixedSize(40, 40)

            self.nav_layout.addStretch()
            self.nav_layout.addWidget(self.popup_btn)

            # Add bars
            self.layout.addLayout(self.nav_layout)
            self.layout.addWidget(self.toolbar)

            self.left_btn.clicked.connect(self.show_previous_figure)
            self.right_btn.clicked.connect(self.show_next_figure)

            if self.interaction_class:
                self.interact_btn.clicked.connect(self.interact)

            if self.labels:
                try:
                    figure0 = self.create_figure(plot='geomShort')
                    self.canvas0 = FigureCanvas(figure0)

                    figure = self.create_figure()
                    self.canvas = FigureCanvas(figure)
                except Exception as e:
                    self.canvas0 = FigureCanvas()
                    self.canvas = FigureCanvas()
                    self.logger.exception(f"Error creating initial figures: {e}")
            else:
                self.canvas0 = FigureCanvas()
                self.canvas = FigureCanvas()

            self.layout.addWidget(self.canvas0)
            self.layout.addWidget(self.canvas)

        except Exception as e:
            self.logger.exception(f"UI initialization failed: {e}")

    def interact(self):
        """Called when user performs picking interaction."""
        try:
            if not self.interactor:
                return

            self.canvas.setFocus()
            self.interactor.interact()
            self.picks[self.current_index] = self.interactor.picks

            self.write_data_to_sql()
            self.update_display()
            self.canvas.setFocus()
        except Exception as e:
            self.logger.exception(f"Interaction failed: {e}")


    def show_popup(self):
        """Help popup with keyboard shortcuts."""
        text = "\n".join((
            'Keyboard commands:',
            '- Left mouse click in dispersion image: add a point with a confidence interval.',
            '- Left mouse click and hold: Drag added point.',
            '- Right mouse click on point: delete point.',
            '- Press numbers 0–9 to set dispersion curve mode index.',
            '- Press d: Delete boundary.',
            '- Press r: Reset picks.',
            '- Mouse Scroll: Adjust boundary tightness.'
        ))

        msg = QMessageBox()
        msg.setWindowTitle('Information')
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def get_picks(self):
        """Load picks from SQL and return them as a dict."""
        try:
            label = self.labels[self.current_index]

            params = {
                'procset': f"'{self.procset}'",
                'method': f"'{self.method}'",
                'sin': label[0],
                'rep': label[1],
                'wid': label[2],
            }

            curves = self._sql.read_curve(params)
            if curves.empty:
                return {}

            picks = {
                mode: {
                    'frequency': group['frequency'].to_numpy(),
                    'velocity': group['velocity'].to_numpy()
                }
                for mode, group in curves.groupby('dc_mode')
            }

            return picks

        except Exception as e:
            self.logger.exception(f"Error loading picks: {e}")
            return {}

    def write_data_to_sql(self):
        """Writes edited picks to SQL."""
        if not self.interactor:
            return

        try:
            label = self.labels[self.current_index]

            # Confirm picks property exists on class
            if not isinstance(type(self.interactor).picks, property):
                return

            params = {
                'procset': f"'{self.procset}'",
                'method': f"'{self.method}'",
                'sin': label[0],
                'rep': label[1],
                'wid': label[2]
            }

            # Delete old picks
            self._sql.delete_data('curve', params)

            # Write new picks
            picks = self.picks.get(self.current_index, {})
            for dc_mode, pv in picks.items():
                dc = {
                    'xmid': self.stream.midpoint,
                    'method': self.method,
                    'dc_mode': int(dc_mode),
                    'frequency': pv['frequency'],
                    'velocity': pv['velocity'],
                    'error': np.zeros(len(pv['velocity'])),
                }

                self._sql.write_curve(
                    dc, label[0], label[1],
                    procset=self.procset,
                    wid=label[2],
                    xmid=self.stream.midpoint
                )

        except Exception as e:
            self.logger.exception(f"Error writing picks to SQL: {e}")

    def set_method(self, method):
        """Change FV/dc processing method and refresh display."""
        try:
            self.method = method

            if self.method == 'FK':
                self._interact_kwargs['domain'] = 'FK'
                self.plot = 'FK'
            else:
                self._interact_kwargs['domain'] = 'FV'
                self.plot = 'FV'

            self.update_display()
        except Exception as e:
            self.logger.exception(f"Error setting method {method}: {e}")


class DataSwitcherFilter(DataSwitcherBase):
    """Manual filter interface."""

    def __init__(self, data, sql, plot='FK', use_windows=False,
                 procset=None, procsets=None, select_plot=True,
                 interaction_class=None,**kwargs):

        self.ftype = plot

        self.flag = False

        self.canvas1 = FigureCanvas()
        self.seis_kwargs = {'show_xticks': False, 'title': None, 'figsize': (7.2, 8)}

        # apply the FK filter to the data after closing the window
        self._apply_filter = kwargs.pop('apply_filter', True)

        # unique filter exists for each rep
        self._uniq_per_rep = kwargs.pop('uniq_per_rep', True)

        # unique filter exists for each wid
        self._uniq_per_wid = kwargs.pop('uniq_per_wid', True)

        interaction_class = interaction_class

        super().__init__(
            data, sql, plot=plot, use_windows=use_windows,
            interaction_class=interaction_class,
            procset=procset, procsets=procsets,
            btn_label='Filter | Reset',
            select_plot=select_plot,
            **kwargs
        )

    def current_data(self):

        try:
            label = self.labels[self.current_index]
        except Exception:
            #self.logger.error("Invalid current_index in create_figure.")
            return None

        sin, rep, wid = label
        try:
            stream = self.select_data(sin, rep)
            self.data_exists = self._set_data(stream, sin, rep, self.procset, wid)
            self.stream,self.flag = self.apply_filter(stream, sin, rep, wid)

        except Exception as err:
            self.logger.error(f"Error setting data: {err}")
            return None

    def create_figure(self, plot=None,  **kwargs):
        """Safely create the requested figure."""

        if not self.labels:
            return None

        plot = plot or self.plot

        # Default stream plot
        if self.data_exists:
            try:
                return self.stream.plot(plot, show=False, gui=True, **self.kwargs, **kwargs)
            except Exception as err:
                self.logger.error(f"Error during stream plot: {err}")
                return None

        return None

    def setup_interaction(self, figure):

        self.interactor = None
        if (self.interaction_class is not None) and self.data_exists:
            ax = figure.axes[0]
            points = self.points.get(self.current_index, {})
            picks = self.picks.get(self.current_index, {})

            self.interactor = self.interaction_class(ax, points=points, data=self.stream,
                                                     picks=picks, **self._interact_kwargs)

            self.points[self.current_index] = self.interactor.points
            self.picks[self.current_index] = self.interactor.picks

    def interact_filter(self):
        """Apply filter based on interaction points."""
        try:
            if not self.interactor:
                return

            self.canvas.setFocus()
            points = self.points.get(self.current_index, {})
            label = self.labels[self.current_index]

            if points:
                # Save filter points
                self._sql.write_filter(points, label[0], label[1], self.interactor.key,
                                       procset=self.procset, wid=label[2], type=self.ftype)
                self.interactor.update_plot()
                self.points[self.current_index] = {}

            else:
                # delete filter
                params = {'sin': label[0], 'rep': label[1], 'procset': "'%s'" % self.procset, 'wid':label[2]}
                self._sql.delete_data('filter', params)

            self.update_display()
            self.canvas.setFocus()

        except Exception as e:
            self.logger.exception(f"Interaction failed: {e}")

    def interact_save(self):
        """Save filtered data to database."""

        def non_uniq(stream, sin, rep,wid):
            _ = self._set_data(stream, sin, rep, self.procset, wid)
            stream, flag = self.apply_filter(stream, sin, rep, wid)
            self.write_current_filtered_data(stream, flag, sin, rep, wid)

        label = self.labels[self.current_index]
        sin, rep, wid = label

        self.write_current_filtered_data(self.stream, self.flag, sin, rep, wid)

        for sin_, rep_, wid_ in self.labels:

            stream = self.select_data(sin, rep)

            if not self._uniq_per_rep and rep_ != rep:
                non_uniq(stream, sin, rep_, wid)

            if not self._uniq_per_wid and wid_ != wid:
                non_uniq(stream, sin, rep, wid_)

    def interact_delete(self):
        """Delete data from database."""

        label = self.labels[self.current_index]
        sin, rep, wid = label

        procset = self.procset

        params = {'sin': sin, 'rep': rep, 'wid': wid, 'procset': "'%s'" % procset}
        self._sql.delete_data('amplitudes', params)

        self.update_display()
        self.canvas.setFocus()

    def show_popup(self):
        """Information"""
        text = "\n".join((
            'Workflow:',
            '1.  Add points with left mouse click.',
            '2.a Press button "Filter|Reset" to mute data.',
            '2.b Optionally, press button "Filter|Reset" again to reset filter.',
            '3.  Press button "Save" to store filtered data in database.',
            '4.  Press button "Delete" to remove data from database.',
            '',
            'Keyboard commands:',
            f'- Left mouse click in {self.ftype} plot: add a point.',
            '- Left mouse click and hold: drag added point.',
            '- Right mouse click on point: delete point.',
            '- Press t for top or b for bottom filter.'
        ))
        msg = QMessageBox()
        msg.setWindowTitle('Information')
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def apply_filter(self, stream, sin, rep, wid):

        # Load FK filter
        params = {'sin': sin,
                  'procset': "'%s'" % self.procset,
                  'type': "'%s'" % self.ftype}

        if self._uniq_per_rep:
            params['rep'] = rep

        if self._uniq_per_wid:
            params['wid'] = wid

        points = self._sql.read_filter(params)

        if not points.empty:
            pt, pb = filter_df2dict(points)

            if self.ftype == 'TX':
                stream.apply_linear_mute([pt, pb], ["t", "b"],
                                         tapering=self._interact_kwargs['tapering'],
                                         taper_type=self._interact_kwargs['taper_type'])

            if self.ftype == 'FK':
                stream.apply_fk_filter([pt, pb], ["t", "b"])

            return stream, True

        return stream, False


    def write_current_filtered_data(self, stream, flag, sin, rep, wid):

        if flag:

            starttime = time.time()

            sys.stdout.write(
                f'\rApplying FK filter to (SIN,REP,WID) = ({sin}, {rep}, {wid}) and writing to database..... ')
            sys.stdout.flush()

            if self.ftype == 'FK':
                stream.tapered_amps = 1

            self._write_data(stream, sin, rep, self.procset, wid)

            endtime = time.time()
            print(f'{np.round(endtime - starttime, 2)} s')

        else:
            self.logger.info(f'No filter applied to (SIN,REP,WID) = ({sin}, {rep}, {wid}).')

    def write_filtered_data(self):

        starttime = time.time()

        if self.is_grouped:
            labels = [label for group in self.all_labels for label in group]
        else:
            labels = self.all_labels

        for sin, rep, wid in labels:

            stream = self.select_data(sin, rep)
            self.data_exists = self._set_data(stream, sin, rep, self.procset, wid)

            stream, flag = self.apply_filter(stream, sin, rep, wid)

            if flag:
                sys.stdout.write(
                    f'\rApplying FK filter to (SIN,REP,WID) = ({sin}, {rep}, {wid}) and writing to database..... ')
                sys.stdout.flush()

                if self.ftype == 'FK':
                    stream.tapered_amps = 1

                self._write_data(stream, sin, rep, self.procset, wid)

        endtime = time.time()
        print(f'{np.round(endtime - starttime, 2)} s')

    def add_button(self, label, width = 100):

        btn = QPushButton(label)
        btn.setFixedSize(width, 40)
        self.nav_layout.addWidget(btn)

        return btn

    # def closeEvent(self, event):
    #
    #     if self._apply_filter:
    #         self.write_filtered_data()
    #     event.accept()


class DataSwitcherFilterTX(DataSwitcherFilter):
    """Manual FK filter interface."""

    def __init__(self, data, sql, plot='TX', use_windows=False,
                 procset=None, procsets=None, select_plot=True, **kwargs):

        interaction_class = TXInteractive

        super().__init__(
            data, sql, plot=plot, use_windows=use_windows,
            interaction_class=interaction_class,
            procset=procset, procsets=procsets,
            select_plot=select_plot,
            **kwargs
        )

    def init_ui(self):
        """Build the FK filter interface."""

        try:
            self.layout = QVBoxLayout(self)

            self.toolbar1 = NavigationToolbar(self.canvas, self)

            self.toolbar_layout = QHBoxLayout()

            self.info_layout = QHBoxLayout()

            label = QLabel(self.window_label)
            label.setStyleSheet("font-size: 14px; color: gray;")
            self.info_layout.addWidget(label)

            # SIN/REP/WIN index
            self.label = QLabel()
            self.label.setStyleSheet("font-size: 14px; color: gray;")

            self.info_layout.addStretch()
            self.info_layout.addWidget(self.label)
            self.layout.addLayout(self.info_layout)

            # Navigation bar
            self.nav_layout = QHBoxLayout()
            self.left_btn = QPushButton()
            self.left_btn.setIcon(self.style().standardIcon(self.style().SP_ArrowLeft))
            self.left_btn.setFixedSize(40, 40)
            self.right_btn = QPushButton()
            self.right_btn.setIcon(self.style().standardIcon(self.style().SP_ArrowRight))
            self.right_btn.setFixedSize(40, 40)

            self.nav_layout.addWidget(self.left_btn)
            self.nav_layout.addWidget(self.right_btn)

            if not self.is_grouped:
                self.combo = self.add_combobox(self.procsets, self.procset)
                self.combo.currentTextChanged.connect(self.set_procset)
                self.nav_layout.addWidget(self.combo)

            if not self.is_grouped:
                _, indices = np.unique(np.asarray(self.labels)[:, 0], return_index=True)
                indices = np.char.mod('%d', indices + 1)

                self.combo_select_sin = self.add_combobox(indices,
                                                          str(self.current_index + 1),
                                                          'SIN:',
                                                          50)
                self.combo_select_sin.currentTextChanged.connect(self.set_index)
                self.nav_layout.addWidget(self.combo_select_sin)

            if self.interaction_class:
                self.interact_btn_filt = self.add_button(self.btn_label)
                self.interact_btn_save = self.add_button('Save', width = 60)
                self.interact_btn_del = self.add_button('Delete', width = 60)


            self.popup_btn = QPushButton()
            icon = QApplication.style().standardIcon(QStyle.SP_MessageBoxInformation)
            self.popup_btn.setIcon(icon)
            self.popup_btn.clicked.connect(self.show_popup)
            self.popup_btn.setFixedSize(40, 40)

            self.nav_layout.addStretch()

            self.nav_layout.addWidget(self.popup_btn)

            self.layout.addLayout(self.nav_layout)

            self.toolbar_layout.addWidget(self.toolbar1)
            self.layout.addLayout(self.toolbar_layout)

            self.left_btn.clicked.connect(self.show_previous_figure)
            self.right_btn.clicked.connect(self.show_next_figure)

            if self.interaction_class:
                self.interact_btn_filt.clicked.connect(self.interact_filter)
                self.interact_btn_save.clicked.connect(self.interact_save)
                self.interact_btn_del.clicked.connect(self.interact_delete)

            # Canvas
            self.fig_layout = QHBoxLayout()

            if self.labels:
                self.current_data()
                figure0 = self.create_figure(plot='geomShort')
                self.canvas0 = FigureCanvas(figure0)

                figure1 = self.create_figure(plot='seismogram', **self.seis_kwargs)
                self.canvas = FigureCanvas(figure1)
                self.canvas.setFocusPolicy(Qt.StrongFocus)
                self.canvas.setFocus()

            self.fig_layout.addWidget(self.canvas)

            self.layout.addWidget(self.canvas0)
            self.layout.addLayout(self.fig_layout)

        except Exception as e:
            self.logger.exception(f"UI initialization failed: {e}")

    def update_display(self):
        """Update all canvases, toolbars, labels, and interactor."""

        try:
            # Store points from previous session
            if self.interactor and getattr(self.interactor, 'points', None):
                self.points[self.current_index] = self.interactor.points

            # enable/disable navigation buttons
            num_figures = len(self.labels)
            is_navigation_enabled = num_figures > 1
            self.left_btn.setEnabled(is_navigation_enabled)
            self.right_btn.setEnabled(is_navigation_enabled)

            self.current_data()
            figure0 = self.create_figure(plot='geomShort')
            figure1 = self.create_figure(plot='seismogram', **self.seis_kwargs)

            # update label
            if not self.is_grouped:
                label = self.labels[self.current_index]
                if not figure0:
                    self.label.setText(f"SIN {label[0]} | REP {label[1]} | No data")
                else:
                    self.label.setText(f"SIN {label[0]} | REP {label[1]}")  # + " | " + self.active_label)
            else:
                if not figure0:
                    self.label.setText(f"No data")
                else:
                    label = self.labels[self.current_index]
                    self.label.setText(
                        f"SIN {label[0]} | REP {label[1]} | WIN {label[2] + 1}")  # + " | " + self.active_label)

            # figure0.tight_layout()
            if not figure0:
                self.canvas0 = self.create_placeholder(self.canvas0)
                self.canvas = self.create_placeholder(self.canvas)
                return

            # Replace canvas
            self.canvas0 = self.replace_widget(self.canvas0, figure0)
            self.canvas = self.replace_widget(self.canvas, figure1)

            # Reset toolbar
            for tb, c in [(self.toolbar1, self.canvas)]:
                self.layout.removeWidget(tb)
                tb.setParent(None)

            self.toolbar1 = NavigationToolbar(self.canvas, self.canvas)
            self.toolbar_layout = QHBoxLayout()
            self.toolbar_layout.addWidget(self.toolbar1)
            self.layout.addLayout(self.toolbar_layout)

            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setFocus()
            self.canvas.setEnabled(self.data_exists)

            self.fig_layout = QHBoxLayout()
            self.fig_layout.addWidget(self.canvas)
            self.layout.addWidget(self.canvas0)
            self.layout.addLayout(self.fig_layout)

            # Setup interaction
            self.setup_interaction(figure=figure1)

        except Exception as e:
            self.logger.exception(f"Error updating display: {e}")


class DataSwitcherFilterFK(DataSwitcherFilter):
    """Manual FK filter interface."""

    def __init__(self, data, sql, plot='FK', use_windows=False,
                 procset=None, procsets=None, select_plot=True, **kwargs):

        interaction_class = FKFilterInteractive

        super().__init__(
            data, sql, plot=plot, use_windows=use_windows,
            interaction_class=interaction_class,
            procset=procset, procsets=procsets,
            select_plot=select_plot,
            **kwargs
        )

    def init_ui(self):
        """Build the FK filter interface."""

        try:
            self.layout = QVBoxLayout(self)

            self.toolbar1 = NavigationToolbar(self.canvas, self)
            self.toolbar2 = NavigationToolbar(self.canvas1, self)

            self.toolbar_layout = QHBoxLayout()

            self.info_layout = QHBoxLayout()

            label = QLabel(self.window_label)
            label.setStyleSheet("font-size: 14px; color: gray;")
            self.info_layout.addWidget(label)

            # SIN/REP/WIN index
            self.label = QLabel()
            self.label.setStyleSheet("font-size: 14px; color: gray;")

            self.info_layout.addStretch()
            self.info_layout.addWidget(self.label)
            self.layout.addLayout(self.info_layout)

            # Navigation bar
            self.nav_layout = QHBoxLayout()
            self.left_btn = QPushButton()
            self.left_btn.setIcon(self.style().standardIcon(self.style().SP_ArrowLeft))
            self.left_btn.setFixedSize(40, 40)
            self.right_btn = QPushButton()
            self.right_btn.setIcon(self.style().standardIcon(self.style().SP_ArrowRight))
            self.right_btn.setFixedSize(40, 40)

            self.nav_layout.addWidget(self.left_btn)
            self.nav_layout.addWidget(self.right_btn)

            if not self.is_grouped:
                self.combo = self.add_combobox(self.procsets, self.procset)
                self.combo.currentTextChanged.connect(self.set_procset)
                self.nav_layout.addWidget(self.combo)

            if not self.is_grouped:
                _, indices = np.unique(np.asarray(self.labels)[:, 0], return_index=True)
                indices = np.char.mod('%d', indices + 1)

                self.combo_select_sin = self.add_combobox(indices,
                                                          str(self.current_index + 1),
                                                          'SIN:',
                                                          50)
                self.combo_select_sin.currentTextChanged.connect(self.set_index)
                self.nav_layout.addWidget(self.combo_select_sin)

            if self.interaction_class:
                self.interact_btn_filt = self.add_button(self.btn_label)
                self.interact_btn_save = self.add_button('Save', width = 60)
                self.interact_btn_del = self.add_button('Delete', width = 60)

            self.popup_btn = QPushButton()
            icon = QApplication.style().standardIcon(QStyle.SP_MessageBoxInformation)
            self.popup_btn.setIcon(icon)
            self.popup_btn.clicked.connect(self.show_popup)
            self.popup_btn.setFixedSize(40, 40)

            self.nav_layout.addStretch()

            self.nav_layout.addWidget(self.popup_btn)

            self.layout.addLayout(self.nav_layout)

            self.toolbar_layout.addWidget(self.toolbar1)
            self.toolbar_layout.addWidget(self.toolbar2)
            self.layout.addLayout(self.toolbar_layout)

            self.left_btn.clicked.connect(self.show_previous_figure)
            self.right_btn.clicked.connect(self.show_next_figure)

            if self.interaction_class:
                self.interact_btn_filt.clicked.connect(self.interact_filter)
                self.interact_btn_save.clicked.connect(self.interact_save)
                self.interact_btn_del.clicked.connect(self.interact_delete)

            # Canvas
            self.fig_layout = QHBoxLayout()

            if self.labels:
                self.current_data()
                figure0 = self.create_figure(plot='geomShort')
                self.canvas0 = FigureCanvas(figure0)

                figure1 = self.create_figure(plot='seismogram', **self.seis_kwargs)
                self.canvas1 = FigureCanvas(figure1)

                figure2 = self.create_figure()
                self.canvas = FigureCanvas(figure2)
                self.canvas.setFocusPolicy(Qt.StrongFocus)
                self.canvas.setFocus()

            self.fig_layout.addWidget(self.canvas)
            self.fig_layout.addWidget(self.canvas1)

            self.layout.addWidget(self.canvas0)
            self.layout.addLayout(self.fig_layout)

        except Exception as e:
            self.logger.exception(f"UI initialization failed: {e}")

    def update_display(self):
        """Update all canvases, toolbars, labels, and interactor."""

        try:
            # Store points from previous session
            if self.interactor and getattr(self.interactor, 'points', None):
                self.points[self.current_index] = self.interactor.points

            # enable/disable navigation buttons
            num_figures = len(self.labels)
            is_navigation_enabled = num_figures > 1
            self.left_btn.setEnabled(is_navigation_enabled)
            self.right_btn.setEnabled(is_navigation_enabled)

            self.current_data()
            figure0 = self.create_figure(plot='geomShort')
            figure1 = self.create_figure(plot='seismogram', **self.seis_kwargs)
            figure = self.create_figure()

            # update label
            if not self.is_grouped:
                label = self.labels[self.current_index]
                if not figure0:
                    self.label.setText(f"SIN {label[0]} | REP {label[1]} | No data")
                else:
                    self.label.setText(f"SIN {label[0]} | REP {label[1]}")  # + " | " + self.active_label)
            else:
                if not figure0:
                    self.label.setText(f"No data")
                else:
                    label = self.labels[self.current_index]
                    self.label.setText(
                        f"SIN {label[0]} | REP {label[1]} | WIN {label[2] + 1}")  # + " | " + self.active_label)

            # figure0.tight_layout()
            if not figure0:
                self.canvas0 = self.create_placeholder(self.canvas0)
                self.canvas1 = self.create_placeholder(self.canvas1)
                self.canvas = self.create_placeholder(self.canvas)
                return

            # Replace canvas
            self.canvas0 = self.replace_widget(self.canvas0, figure0)
            self.canvas1 = self.replace_widget(self.canvas1, figure1)
            self.canvas = self.replace_widget(self.canvas, figure)

            # Reset toolbar
            for tb, c in [(self.toolbar1, self.canvas), (self.toolbar2, self.canvas1)]:
                self.layout.removeWidget(tb)
                tb.setParent(None)

            self.toolbar1 = NavigationToolbar(self.canvas, self.canvas)
            self.toolbar2 = NavigationToolbar(self.canvas1, self.canvas1)
            self.toolbar_layout = QHBoxLayout()
            self.toolbar_layout.addWidget(self.toolbar1)
            self.toolbar_layout.addWidget(self.toolbar2)
            self.layout.addLayout(self.toolbar_layout)

            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setFocus()
            self.canvas.setEnabled(self.data_exists)

            self.fig_layout = QHBoxLayout()
            self.fig_layout.addWidget(self.canvas)
            self.fig_layout.addWidget(self.canvas1)
            self.layout.addWidget(self.canvas0)
            self.layout.addLayout(self.fig_layout)

            # Setup interaction
            self.setup_interaction(figure=figure)

        except Exception as e:
            self.logger.exception(f"Error updating display: {e}")


class CurveFilter(QWidget):
    """User interface to manually remove points from multiple curves"""

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.data_filtered = None
        self.keys = list(data.keys())
        self.current_index = 0

        history = []
        for key in self.keys:
            curves = data[key]
            history.append([np.ones(len(curves),dtype=bool)])
        self.history = history

        self.init_ui()

    def init_ui(self):
        """interface design"""

        # UI Setup
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.figure.add_subplot(111)
        self.selector = None

        # Buttons
        self.left_btn = QPushButton()
        self.left_btn.setIcon(self.style().standardIcon(self.style().SP_ArrowLeft))
        self.left_btn.setFixedSize(40, 40)
        self.right_btn = QPushButton()
        self.right_btn.setIcon(self.style().standardIcon(self.style().SP_ArrowRight))
        self.right_btn.setFixedSize(40, 40)
        self.undo_btn = QPushButton("Cancel")
        self.undo_btn.setFixedSize(60, 40)

        self.popup_btn = QPushButton()
        icon = QApplication.style().standardIcon(QStyle.SP_MessageBoxInformation)
        self.popup_btn.setIcon(icon)
        self.popup_btn.clicked.connect(self.show_popup)
        self.popup_btn.setFixedSize(60, 40)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.left_btn)
        btn_layout.addWidget(self.right_btn)

        btn_layout.addStretch()

        btn_layout.addWidget(self.undo_btn)
        btn_layout.addWidget(self.popup_btn)

        layout = QVBoxLayout()
        layout.addLayout(btn_layout)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Connect signals
        self.left_btn.clicked.connect(self.show_prev_data)
        self.right_btn.clicked.connect(self.show_next_data)
        self.undo_btn.clicked.connect(self.undo_filter)

        self.update_display()

    def current_mask(self):
        """Get the most recent mask for the active curve."""
        return self.history[self.current_index][-1].copy()

    def update_display(self):
        """Refresh the plot with current curve and mask."""
        self.ax.clear()

        curves = self.data[self.keys[self.current_index]]
        mask = self.current_mask()

        label1 = 'points to keep'
        label2 = 'points to remove'

        x, y = curves['frequency'].to_numpy(), curves['velocity'].to_numpy()

        self.ax.scatter(x, y, color="royalblue", label=label1, s = 20)
        self.ax.scatter(x[~mask], y[~mask], color="red", label=label2)

        self.ax.set_xlabel('frequency (Hz)')
        self.ax.set_ylabel('phase velocity (m/s)')

        self.ax.set_title(f"Curve {self.current_index + 1}/{len(self.data)} "
                          f"located at x = {self.keys[self.current_index]}", fontweight = 'bold', loc = 'left')
        self.ax.legend(loc='upper right', edgecolor='k', frameon=True)
        self.ax.grid(True, linestyle=':')
        self.canvas.draw_idle()

        # PolygonSelector
        if self.selector:
            self.selector.disconnect_events()
        self.selector = PolygonSelector(self.ax, self.onselect, useblit=True)

    def onselect(self, verts):
        """Handle polygon selection and update mask."""

        curves = self.data[self.keys[self.current_index]]
        x, y = curves['frequency'].to_numpy(), curves['velocity'].to_numpy()

        path = Path(verts)
        pts = np.column_stack((x, y))
        inside = path.contains_points(pts)

        # Save current mask state
        current = self.current_mask()
        self.history[self.current_index].append(current.copy())

        # Apply filtering (toggle: points inside polygon are excluded)
        new_mask = current & ~inside
        self.history[self.current_index][-1] = new_mask

        self.update_display()

    def undo_filter(self):
        """Undo last filtering operation for current curve."""

        curve_hist = self.history[self.current_index]
        if len(curve_hist) > 1:
            curve_hist.pop()  # Remove last mask

        self.update_display()

    def show_next_data(self):
        self.current_index = (self.current_index + 1) % len(self.data)
        self.update_display()

    def show_prev_data(self):
        self.current_index = (self.current_index - 1) % len(self.data)
        self.update_display()

    def show_popup(self):

        text = '\n'.join((
            '1. Draw a polygon by left click on the figure to add polygon edges.',
            '2. Once a closed polygon is drawn, points within will be marked.',
            '3. After marking all points, close App to remove marked points.',
            '4. Press button Cancel to cancel selection in current canvas.'))

        msg = QMessageBox()
        msg.setWindowTitle("Information")
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def remove_points(self):
        """Remove masked points"""

        data_flt = {}
        for j,key in enumerate(self.keys):

            curves = self.data[key]
            mask = self.history[j][-1]

            curves_flt = curves.loc[mask].reset_index()
            data_flt[key] = curves_flt

        return data_flt

    def closeEvent(self, event):

        self.data_filtered = self.remove_points()
        event.accept()