# type: ignore Unfortunatly, a lot of Qt stuff has incorrect type annotations :(

import itertools
from typing import Optional, Tuple, List

from PyQt5 import QtCore, QtGui
from PyQt5.Qt import Qt, QPixmap, QIcon
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QPoint
from PyQt5.QtWidgets import QApplication, QCheckBox, QMainWindow, QDockWidget, QLabel, QMdiArea, QMdiSubWindow, QProgressBar, \
    QFormLayout, QSpinBox, QDoubleSpinBox, QTabWidget, QWidget, QScrollArea, QPushButton, QFileDialog, QComboBox, QLineEdit, QBoxLayout
import pyqtgraph as pg
import numpy as np
import functools
import os

from . import simstate
from . import siunits
from . import simjob
from . import profiling
from . import sardata
from . import simscene


# Patch in the "jet" colormap used in the demo etc.
pg.graphicsItems.GradientEditorItem.Gradients['jet'] = {
    'ticks': [(0.0 / 63, (0, 0, 144)), (7.0 / 63, (0, 0, 255)), (23.0 / 63, (0, 255, 255)), (39.0 / 63, (255, 255, 0)),
              (55.0 / 63, (255, 0, 0)), (63.0 / 63, (127, 0, 0))], 'mode': 'rgb'}
pg.graphicsItems.GradientEditorItem.Gradients['paperjet'] = {
    'ticks': [
        (0.0 / 63, (0, 0, 26)),
        (7.0 / 63, (0, 0, 53)),
        (23.0 / 63, (0, 117, 117)),
        (39.0 / 63, (149, 149, 0)),
        (55.0 / 63, (229, 0, 0)),
        (63.0 / 63, (255, 229, 229))
    ], 'mode': 'rgb'}
pg.graphicsItems.GradientEditorItem.Gradients['invpaperjet'] = {
    'ticks': [
        (0.00, (214, 205, 255)),
        (0.11, (163, 186, 248)),
        (0.37, (0, 202, 202)),
        (0.62, (152, 152, 0)),
        (0.87, (111, 0, 0)),
        (1.00, (75, 25, 25)),
    ], 'mode': 'rgb'}

class SarGuiPlotWindowBase(QMdiSubWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(self.get_title())

        self._layout = pg.GraphicsLayoutWidget()

    def do_autorange(self):
        pass

    BG_STALE = (64, 0, 0)
    BG_NORMAL = (0, 0, 0)

    def mark_stale(self, stale: bool = True):
        if stale:
            self._layout.setBackground(SarGuiPlotWindowBase.BG_STALE)
        else:
            self._layout.setBackground(SarGuiPlotWindowBase.BG_NORMAL)

    def get_title(self) -> str:
        raise NotImplementedError()

class SarGuiImagePlotBase(SarGuiPlotWindowBase):
    def __init__(self, aspect_lock: bool, unit_x: str = 'm', unit_y: str = 'm') -> None:
        super().__init__()
        self._sim_image = None
        self._data = np.array([[]])
        self._data_tr = QtGui.QTransform()

        self._create_plot_widget()

        if aspect_lock:
            self._p1.setAspectLocked()
        self._p1.setLabel('left', 'Azimuth / X', unit_x)
        self._p1.setLabel('bottom', 'Range / Y', unit_y)
        self._unit_x = unit_x
        self._unit_y = unit_y

    @property
    def data(self) -> simstate.SimImage:
        return self._sim_image

    @data.setter
    def data(self, image: simstate.SimImage):
        self._sim_image = image
        magimg = 20 * np.log10(np.clip(np.abs(image.data), 1e-30, None))
        self._data = np.clip(magimg - np.max(magimg), -240, None)
        self.set_transform(image.y0, image.x0, image.dy, image.dx)
        self._update_data()

    def set_transform(self, x0: float, y0: float, dx: float, dy: float) -> None:
        tr = QtGui.QTransform()
        tr.translate(x0-dx/2, y0-dy/2).scale(dx, dy)
        self._data_tr = tr
        self._img.setTransform(self._data_tr)

    def _set_random_data(self):
        # Generate image data
        data = np.random.normal(size=(200, 100))
        data[20:80, 20:80] += 2.
        data = pg.gaussianFilter(data, (3, 3))
        data += np.clip(-30 + np.random.normal(size=(200, 100)) * 10, -90, 0)
        self.set_transform(0, 0, 1, 1)
        self.data = data

    def do_autorange(self) -> None:
        self._hist.setLevels(-90, 0)
        self._p1.autoRange()

    def _create_plot_widget(self):
        # Interpret image data as row-major instead of col-major

        isolines = False

        # A plot area (ViewBox + axes) for displaying the image
        self._p1: pg.PlotItem = self._layout.addPlot(row=1, col=1, title="")

        # Item for displaying image data
        self._img = pg.ImageItem()
        self._p1.addItem(self._img)

        # Isocurve drawing
        if isolines:
            self._iso = pg.IsocurveItem(level=0.8, pen='g')
            self._iso.setParentItem(self._img)
            self._iso.setZValue(5)
            self._iso2 = pg.IsocurveItem(level=0.8, pen='r')
            self._iso2.setParentItem(self._img)
            self._iso2.setZValue(6)
        else:
            self._iso = None
            self._iso2 = None

        # Contrast/color control
        self._hist = pg.HistogramLUTItem()
        self._hist.gradient.loadPreset('jet') #spectrum, viridis
        self._hist.setLevels(-90, 0)
        self._hist.setImageItem(self._img)
        self._hist.axis.setLabel('Magnitude', 'dB')
        self._layout.addItem(self._hist, row=1, col=2)

        # Draggable line for setting isocurve level
        if isolines:
            self._isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
            self._hist.vb.addItem(self._isoLine)
            self._hist.vb.setMouseEnabled(y=False)  # makes user interaction a little easier
            self._isoLine.setValue(-6)
            self._isoLine.setZValue(1000)  # bring iso line above contrast controls
            self._isoLine2 = pg.InfiniteLine(angle=0, movable=True, pen='r')
            self._hist.vb.addItem(self._isoLine2)
            self._isoLine2.setValue(-24)
            self._isoLine2.setZValue(1000)  # bring iso line above contrast controls

        self.setWidget(self._layout)

        if isolines:
            self._isoLine.sigDragged.connect(self._update_isocurve)
            self._isoLine2.sigDragged.connect(self._update_isocurve)
            self._update_isocurve()

        # Monkey-patch the image to use our custom hover function.
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this.
        self._img.hoverEvent = self._image_hover_event

    def _update_levels(self):
        if self._data is None:
            return
        lmin, lmax = self._hist.getLevels()
        self._range_cut_plot.vb.setYRange(lmin, lmax)
        self._azi_cut_plot.vb.setXRange(lmin, lmax)
            
    def _update_data(self):
        lmin, lmax = self._hist.getLevels()
        self._img.setImage(self._data)
        # hist.setLevels(data.min(), data.max())
        self._hist.setLevels(lmin, lmax)

        # build isocurves from smoothed data
        if self._iso is not None:
            # gfilter = pg.gaussianFilter(data, (2, 2))
            # gfilter = pg.gaussianFilter(data, (1, 1))
            gfilter = self._data
            self._iso.setData(gfilter)
            self._iso2.setData(gfilter)

        # set position and scale of image
        self._img.setTransform(self._data_tr)

    def _update_isocurve(self):
        self._iso.setLevel(self._isoLine.value())
        self._iso2.setLevel(self._isoLine2.value())

    def _image_hover_event(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self._p1.setTitle("")
            return
        pos = event.pos()
        i, j = pos.y(), pos.x()
        i = int(np.clip(i, 0, self._data.shape[0] - 1))
        j = int(np.clip(j, 0, self._data.shape[1] - 1))
        val = self._data[i, j]
        #ppos = self._img.mapToParent(pos)
        ppos = self._img.mapToParent(QPoint(i, j))
        x, y = ppos.x(), ppos.y()
        #self._p1.setTitle("pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %g" % (x, y, i, j, val))
        self._p1.setTitle(f"{siunits.format_si_unit(x, self._unit_x)}, {siunits.format_si_unit(y, self._unit_y)} : "
                          f"{val:.2f} dB")

    def set_color_preset(self, preset: str = 'jet'):
        self._hist.gradient.loadPreset(preset)  # spectrum, viridis

class SarGuiRawDataWindow(SarGuiImagePlotBase):
    def __init__(self) -> None:
        super().__init__(aspect_lock=False, unit_x='m', unit_y='s')

    def get_title(self) -> str:
        return "Raw FMCW Data"

class SarGuiRangeCompressionWindow(SarGuiImagePlotBase):
    def __init__(self) -> None:
        super().__init__(aspect_lock=False, unit_x='m', unit_y='m')

    def get_title(self) -> str:
        return "Range Compression"

class SarGuiAzimuthCompressionWindow(SarGuiImagePlotBase):
    def __init__(self) -> None:
        super().__init__(aspect_lock=True, unit_x='m', unit_y='m')

    def get_title(self) -> str:
        return "Azimuth Compression"

    def set_transform(self, x0: float, y0: float, dx: float, dy: float) -> None:
        super().set_transform(x0, y0, dx, dy)
        # Update data cuts
        self._update_cuts()

    def do_autorange(self) -> None:
        super().do_autorange()
        self._range_line.setValue(0)
        self._azi_line.setValue(0)

    def _create_plot_widget(self):
        super()._create_plot_widget()
        # Two movable lines for "cuts" trough the data
        self._range_line = pg.InfiniteLine(pos=0, movable=True, angle=0)
        self._azi_line = pg.InfiniteLine(pos=0, movable=True, angle=90)
        self._p1.addItem(self._range_line)
        self._p1.addItem(self._azi_line)
        self._range_line.setZValue(10)  # make sure line is drawn above image
        self._azi_line.setZValue(10)
        self._range_line.sigDragged.connect(self._update_cuts)
        self._azi_line.sigDragged.connect(self._update_cuts)
        self._hist.sigLevelsChanged.connect(self._update_levels)

        # Line plot of the cuts
        self._range_cut_plot: pg.PlotItem = self._layout.addPlot(row=0, col=1)
        self._azi_cut_plot: pg.PlotItem = self._layout.addPlot(row=1, col=0)
        self._azi_cut_plot.setFixedWidth(130)
        self._range_cut_plot.setFixedHeight(130)
        # self._layout.ci.layout.setRowMaximumHeight(0, 130) # set width/height of cut-plots
        # self._layout.ci.layout.setColumnMaximumWidth(0, 130)
        self._range_cut_plot.plot()
        self._azi_cut_plot.plot()
        self._range_cut_plot.setLabel('left', 'Magnitude', 'dB')
        self._azi_cut_plot.setLabel('bottom', 'Magnitude', 'dB')

    def _update_cuts(self):
        if self._data is None:
            return
        range_curve = self._range_cut_plot.listDataItems()[0]
        azi_curve = self._azi_cut_plot.listDataItems()[0]
        inv_transform, _ = self._data_tr.inverted()
        azi_pos, range_pos = inv_transform.map(self._azi_line.value(), self._range_line.value())
        # get the real-world coordinates for the data start/end
        rg_start, az_start = self._data_tr.map(0.5, 0.5)
        rg_end, az_end = self._data_tr.map(self._data.shape[1]-0.5, self._data.shape[0]-0.5)
        az_cnt, rg_cnt = self._data.shape
        if int(range_pos) in range(self._data.shape[0]):
            range_curve.setData(y=self._data[int(range_pos), :], x=np.linspace(rg_start, rg_end, rg_cnt))
        if int(azi_pos) in range(self._data.shape[1]):
            azi_curve.setData(x=self._data[:, int(azi_pos)], y=np.linspace(az_start, az_end, az_cnt)) # get real-world coordinates of data 0,0

    def _update_data(self):
        super()._update_data()
        # Update data cuts
        self._update_cuts()

class SarGuiAutofocusResultWindow(SarGuiAzimuthCompressionWindow):
    def __init__(self) -> None:
        super().__init__()

    def get_title(self) -> str:
        return "Autofocus Result"

class SarGuiFlightPathWindow(SarGuiPlotWindowBase):
    def __init__(self, state: simstate.SarSimParameterState):
        super().__init__()

        self._state = state

        self._data_exact: Optional[np.ndarray] = None
        self._data_distorted: Optional[np.ndarray] = None

        # Create plot items
        self._plots_xyz: pg.PlotItem = self._layout.addPlot(row=1, col=1)
        self._layout.ci.layout.setRowStretchFactor(1, 2)
        self._plots_xyz.setLabel('left', 'Position', 'm')
        self._plots_xyz.setLabel('bottom', '# Aperture') # this is not the position, but the aperture number
        legend = self._plots_xyz.addLegend(offset=(50,-30), colCount=6)
        legend.setParentItem(self._plots_xyz)
        self._plot_x_exact = self._plots_xyz.plot(name='X (exact)', pen='y')
        self._plot_y_exact = self._plots_xyz.plot(name='Y (exact)', pen='g')
        self._plot_z_exact = self._plots_xyz.plot(name='Z (exact)', pen='c')
        self._plot_x_distorted = self._plots_xyz.plot(name='X (distorted)', pen=pg.mkPen('y', style=Qt.DashLine))
        self._plot_y_distorted = self._plots_xyz.plot(name='Y (distorted)', pen=pg.mkPen('g', style=Qt.DashLine))
        self._plot_z_distorted = self._plots_xyz.plot(name='Z (distorted)', pen=pg.mkPen('c', style=Qt.DashLine))

        self._plots_dist: pg.PlotItem = self._layout.addPlot(row=2, col=1)
        self._layout.ci.layout.setRowStretchFactor(2, 1)
        self._plots_dist.setLabel('left', 'Distance', 'λ')
        self._plots_dist.setLabel('bottom', '# Aperture')
        self._plot_dist = self._plots_dist.plot(name='Distance')

        self._plots_dist.getViewBox().setXLink(self._plots_xyz.getViewBox())
        
        self.setWidget(self._layout)

    def get_title(self) -> str:
        return "Flight path"

    @property
    def data_exact(self) -> Optional[np.ndarray]:
        return self._data_exact

    @data_exact.setter
    def data_exact(self, data: np.ndarray):
        self._data_exact = data
        self._update_plots()

    @property
    def data_distorted(self) -> Optional[np.ndarray]:
        return self._data_distorted

    @data_distorted.setter
    def data_distorted(self, data: np.ndarray):
        self._data_distorted = data
        self._update_plots()

    def _update_plots(self):
        if self._data_exact is not None:
            self._plot_x_exact.setData(self._data_exact[:,0])
            self._plot_y_exact.setData(self._data_exact[:,1])
            self._plot_z_exact.setData(self._data_exact[:,2])
        
        if self._data_distorted is not None:
            self._plot_x_distorted.setData(self._data_distorted[:,0])
            self._plot_y_distorted.setData(self._data_distorted[:,1])
            self._plot_z_distorted.setData(self._data_distorted[:,2])

        if self._data_exact is not None and self._data_distorted is not None and self._data_exact.shape == self._data_distorted.shape:
            dist = np.linalg.norm(self._data_exact - self.data_distorted, axis=1)
            # scale to multiples of wavelength
            dist = dist / simjob.SIGNAL_SPEED * self._state.fmcw_start_frequency
            self._plot_dist.setData(dist)

    def do_autorange(self):
        self._plots_xyz.autoRange()
        self._plots_dist.autoRange()

class SarGuiSimWorker(QObject):
    finished = pyqtSignal(simjob.SimResult)
    progress = pyqtSignal(float, str)
    sim_state: simstate.SarSimParameterState
    loaded_data: Optional[sardata.SarData] = None
    sim_scene: simscene.SimulationScene
    gpu_id: int = 0

    def run(self):
        ts = profiling.TimeStamper()

        def cb(p, m):
            self.progress.emit(p, m)

        images = simjob.run_sim(self.sim_state, self.sim_scene, ts, cb, self.loaded_data, self.gpu_id)
        self.finished.emit(images)


class SarGuiParameterDock():
    def __init__(self, win: 'SarGuiMainFrame'):
        # We pass in the window, not the state itself: The state could be exchanged by a new instance, when
        # a capture is loaded. When the state is passed directly, it will be captured by the callbacks, so the
        # controls would still update the old instance.
        self.widgets = {} # Save all widget in a dict, so that we can update them from update_from_state()

        self.tab_control = QTabWidget()
        self.tab_control.setTabPosition(QTabWidget.TabPosition.West)

        def get_category(x: simstate.SimParameter):
            if x.category is not None:
                return x.category
            else:
                return "Uncategorized"

        params_by_catergory = itertools.groupby(sorted(win._state.get_parameters(), key=get_category), key=get_category)
        for category, parameters in params_by_catergory:
            form = QFormLayout()
            form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            widget = QWidget(self.tab_control)
            widget.setLayout(form)

            self.tab_control.addTab(widget, category)

            for parameter in parameters:
                box = None
                if parameter.type.type is float:
                    factor, unit = siunits.choose_si_scale(
                        parameter.default or win._state.get_value(parameter),
                        parameter.type.unit)
                    box = QDoubleSpinBox()
                    box.setDecimals(3)
                    box.setSingleStep(1)
                    if parameter.symbol is not None:
                        box.setPrefix(f'{parameter.symbol} = ')
                    if unit is not None and len(unit) > 0:
                        box.setSuffix(f' {unit}')
                    box.setProperty('si_factor', factor)
                    if parameter.type.min is not None:
                        box.setMinimum(parameter.type.min / factor)
                    if parameter.type.max is not None:
                        box.setMaximum(parameter.type.max / factor)
                    box.setValue(win._state.get_value(parameter) / factor)
                    box.valueChanged.connect(lambda v, p=parameter, f=factor: win._state.set_value(p, v*f))

                elif parameter.type.type is int:
                    box = QSpinBox()
                    if parameter.symbol is not None:
                        box.setPrefix(f'{parameter.symbol} = ')
                    if parameter.type.unit is not None:
                        box.setSuffix(f' {parameter.type.unit}')
                    box.setProperty('si_factor', 1)
                    if parameter.type.min is not None:
                        box.setMinimum(parameter.type.min)
                    if parameter.type.max is not None:
                        box.setMaximum(parameter.type.max)
                    box.setValue(win._state.get_value(parameter))
                    box.valueChanged.connect(lambda v, p=parameter: win._state.set_value(p, v))

                elif parameter.type.type is str or parameter.type.choices is not None:
                    value = win._state.get_value(parameter)
                    choices = parameter.type.choices
                    if value is None:
                        value = ''
                    if choices is not None: # enum-style parameter
                        box = QComboBox()
                        for choice in choices.keys():
                            box.addItem(choice)
                        box.setCurrentText(self._get_dict_key_by_value(choices, value))
                        box.currentTextChanged.connect(lambda v, p=parameter: win._state.set_value(p, choices[v]))
                    else: # normal string parameter
                        box = QLineEdit()
                        box.setText(value)
                        box.textChanged.connect(lambda v, p=parameter: win._state.set_value(p, v))
                    tool_tip = []
                    if parameter.symbol is not None:
                        tool_tip.append(f'{parameter.symbol}')
                    if parameter.type.unit is not None:
                        tool_tip.append(f'[{parameter.type.unit}]')
                    if len(tool_tip) > 0:
                        box.setToolTip(' '.join(tool_tip))

                elif parameter.type.type is bool:
                    box = QCheckBox()
                    box.setChecked(win._state.get_value(parameter))
                    box.stateChanged.connect(lambda v, p=parameter: win._state.set_value(p, v == QtCore.Qt.CheckState.Checked))

                else:
                    print(f'WARNING: Unsupported type in GUI for state parameter "{parameter.name}"!')
                    continue  # others not supported for now in GUI

                form.addRow(parameter.human_name(), box)
                self.widgets[parameter.name] = box

    @classmethod
    def _get_dict_key_by_value(cls, d, v):
        return list(d.keys())[list(d.values()).index(v)]

    def get_widget(self) -> QTabWidget:
        return self.tab_control

    def update_from_state(self, state: simstate.SarSimParameterState) -> None:
        for parameter in state.get_parameters():
            if parameter.name not in self.widgets:
                print(f"Note: Ignoring unknown parameter {parameter.name}")
                continue
            box = self.widgets[parameter.name]
            value = state.get_value(parameter)
            if parameter.type.type in [float, int]:
                factor = box.property('si_factor')
                box.setValue(value / factor)
            elif parameter.type.type is str or parameter.type.choices is not None:
                if isinstance(box, QComboBox):
                    box.setCurrentText(self._get_dict_key_by_value(parameter.type.choices, value))
                else:
                    box.setText(value)
            elif parameter.type.type is bool:
                box.setChecked(value)
            else:
                raise NotImplementedError("Update routine missing!")

class SarGuiMainFrame(QMainWindow):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args # command line args

        self._state = simstate.create_state()
        self._scene = simscene.create_default_scene()

        self._loaded_data: Optional[sardata.SarData] = None
        self._use_loaded_data: bool = False

        self._color_preset = 'jet'

        self._create_menu()
        self._params_control = self._create_parameter_dock()

        self._create_mdi()
        self._create_status_bar()

        self._plot_window_raw = SarGuiRawDataWindow()
        self._plot_window_rc = SarGuiRangeCompressionWindow()
        self._plot_window_ac = SarGuiAzimuthCompressionWindow()
        self._plot_window_af = SarGuiAutofocusResultWindow()
        self._plot_fpath = SarGuiFlightPathWindow(self._state)

        # Connect the level controls of AC and AF together
        self._plot_window_ac._hist.sigLevelsChanged.connect(lambda h: self._plot_window_af._hist.setLevels(*h.getLevels()))
        self._plot_window_af._hist.sigLevelsChanged.connect(lambda h: self._plot_window_ac._hist.setLevels(*h.getLevels()))

        # Add in order of processing
        self._windows: List[SarGuiPlotWindowBase] = [
            self._plot_window_raw,
            self._plot_window_rc,
            self._plot_window_ac,
            self._plot_fpath,
            self._plot_window_af,
        ]
        for win in self._windows:
            self.mdi.addSubWindow(win)

        # Thread for worker
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[SarGuiSimWorker] = None

        # See if we had any sim so far (for initial auto-ranging feature etc)
        self._first_run = True
        for win in self._windows:
            win.mark_stale()
        
        # Default view
        self.mdi.tileSubWindows()
        self.showMaximized()

    def _create_menu(self):
        bar = self.menuBar()
        file = bar.addMenu('&File')
        file.addAction('Load parameters').triggered.connect(self._load_param_file)
        file.addAction('Save parameters').triggered.connect(self._save_param_file)
        file.addAction('Load demo capture').triggered.connect(self._load_demo_capture)
        file.addAction('Unload demo capture').triggered.connect(self._unload_demo_capture)
        file.addAction('E&xit').triggered.connect(lambda: self.close())

        view = bar.addMenu('&View')
        view.addAction('&Tiled').triggered.connect(lambda: self.mdi.tileSubWindows())
        view.addAction('&Cascade').triggered.connect(lambda: self.mdi.cascadeSubWindows())
        view.addAction('&Maximized').triggered.connect(lambda: self.mdi.currentSubWindow().showMaximized())

        _spacing = 0.0375*4
        _count = 3
        scenes = bar.addMenu('&Scenes')
        scenes.addAction('Single Reflector at 0,0').triggered.connect(lambda: self.set_scene(
            simscene.create_default_scene()))
        scenes.addAction('Reflector Row in Azimuth').triggered.connect(lambda: self.set_scene(
            simscene.create_reflector_array_scene(count_x=_count, spacing_x=_spacing)))
        scenes.addAction('Reflector Row in Range').triggered.connect(lambda: self.set_scene(
            simscene.create_reflector_array_scene(count_y=_count, spacing_y=_spacing)))
        scenes.addAction('Reflector Grid arount center').triggered.connect(lambda: self.set_scene(
            simscene.create_reflector_array_scene(count_x=_count, count_y=_count,
                                                  spacing_x=_spacing, spacing_y=_spacing,
                                                  start_x=_spacing*(1-_count)/2,
                                                  start_y=_spacing*(1-_count)/2)))
        colors = bar.addMenu('&Colors')
        self._color_preset_menu_children = {}
        icon_size = 64
        # For icon painting, see pyqtgraph/graphicsItems/GradientEditorItem.py:460 and following
        temp_gradient_editor = pg.GradientEditorItem()
        temp_gradient_editor.length = icon_size
        for preset in pg.graphicsItems.GradientEditorItem.Gradients.keys():
            action = colors.addAction(preset)
            action.triggered.connect(functools.partial(self.set_color_preset, preset))
            action.setCheckable(True)
            action.setChecked(preset == self._color_preset)
            # Make Preview
            temp_gradient_editor.loadPreset(preset)
            pixmap = QtGui.QPixmap(icon_size, icon_size)
            painter = QtGui.QPainter(pixmap)
            gradient = temp_gradient_editor.getGradient()
            brush = QtGui.QBrush(gradient)
            painter.fillRect(QtCore.QRect(0, 0, icon_size, icon_size), brush)
            painter.end()
            action.setIcon(QIcon(pixmap))
            self._color_preset_menu_children[preset] = action

        export = bar.addMenu('&Export')
        export.addAction('Azimuth comp. as image (PNG)').triggered.connect(self._export_png)
        export.addAction('Azimuth comp. as NumPy array').triggered.connect(self._export_npy)

    def set_scene(self, scene: simscene.SimulationScene):
        self._scene = scene

    def set_color_preset(self, preset: str):
        for p, a in self._color_preset_menu_children.items():
            a.setChecked(p == preset)
        self._color_preset = preset
        self._plot_window_raw.set_color_preset(self._color_preset)
        self._plot_window_rc.set_color_preset(self._color_preset)
        self._plot_window_ac.set_color_preset(self._color_preset)
        self._plot_window_af.set_color_preset(self._color_preset)
        pass

    def _load_param_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select parameter file", filter="*.ini")
        if filename:
            self._state = simstate.SarSimParameterState.read_from_file(filename)

        self._update_gui_values_from_state()

    def _load_demo_capture(self):
        dirname = QFileDialog.getExistingDirectory(self, "Select captured \".sardata\" directory",
                                                   options=QFileDialog.ShowDirsOnly | QFileDialog.ReadOnly)
        if dirname:
            print(dirname)
            sd = sardata.SarData.import_from_directory(dirname)
            self._state = sd.sim_state
            self._use_loaded_data = True
            self._loaded_data = sd
            self._label_loaded_dataset.setText(sd.name)

            print(f'Loaded SarData from: {dirname}')
            print(f'Raw FMCW: {len(sd.fmcw_lines)} lines of {len(sd.fmcw_lines[0])} samples')

        self._update_gui_values_from_state()

    def _unload_demo_capture(self):
        self._use_loaded_data = False
        self._loaded_data = None
        self._label_loaded_dataset.setText('(none)')

    def _export_png(self):
        if self._plot_window_ac._data is None:
            return

        levels = self._plot_window_ac._hist.getLevels()
        suggested_name = f"export_{round(levels[1])}dB_to_{round(levels[0])}dB.png"

        filename, _ = QFileDialog.getSaveFileName(self, "Select image file", suggested_name, filter="*.png")
        if not filename:
            return
        if not filename.endswith(".png"):
            filename = filename + ".png"
        
        # for some reason we need to flip the image, otherwise the exported image does not look like the graph
        self._plot_window_ac._img.qimage.mirrored(horizontal=False, vertical=True).save(filename)

    def _export_npy(self):
        if self._plot_window_ac._data is None:
            return

        suggested_name = "azi_comp.npy"
        filename, _ = QFileDialog.getSaveFileName(self, "Select numpy file", suggested_name, filter="*.npy")

        if not filename:
            return

        # np.save will append to the end of the file if it alreday exists. This is usually not expected
        # so we delete the file beforehand, if it exists
        if os.path.exists(filename):
            os.unlink(filename)
        np.save(filename, self._plot_window_ac._sim_image.data)

    def _update_gui_values_from_state(self):
        self._params_control.update_from_state(self._state)

    def _save_param_file(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Select parameter file", filter="*.ini")
        if filename:
            if not filename.endswith(".ini"):
                filename = filename + ".ini"
            self._state.write_to_file(filename)

    # def on_parameter_spinbox_change(self, symbol: str, value: float):
    #     print(f'{symbol} = {value}')

    def _create_parameter_dock(self):
        dock = QDockWidget("Parameters", self)
        dock.setFeatures(QDockWidget.DockWidgetFloatable|QDockWidget.DockWidgetMovable)

        stack = QBoxLayout(QBoxLayout.Direction.TopToBottom)

        param_dock = SarGuiParameterDock(self)

        scroll = QScrollArea(self)
        scroll.setWidget(param_dock.get_widget())

        stack.addWidget(scroll)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)

        btn = QPushButton('RUN SIM')
        btn.clicked.connect(self._rerun_sim)
        form.addRow('Start Simulation', btn)

        btn = QPushButton('ZOOM FIT')
        btn.clicked.connect(self._autorange_plots)
        form.addRow('Auto Range Plots', btn)

        lbl = QLabel('(none)')
        form.addRow('Using External Dataset', lbl)
        self._label_loaded_dataset = lbl
        
        widget = QWidget(self)
        widget.setLayout(form)
        
        stack.addWidget(widget)

        widget = QWidget(self)
        widget.setLayout(stack)
        dock.setWidget(widget)

        self._parameter_dock = dock
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.resizeDocks([dock], [320], QtCore.Qt.Orientation.Horizontal)
        
        return param_dock

    def _autorange_plots(self):
        for win in self._windows:
            win.do_autorange()

    def _show_progress(self, progress: float, message: str):
        self._progress_bar.setValue(progress*1000)
        self._progress_label.setText(message)

    def _show_results(self, result: simjob.SimResult):
        print('GUI: Updating Plots')

        self._plot_window_raw.data = result.raw
        self._plot_window_rc.data = result.rc
        self._plot_window_ac.data = result.ac
        self._plot_window_af.data = result.af
        self._plot_fpath.data_exact = result.fpath_exact
        self._plot_fpath.data_distorted = result.fpath_distorted
        for win in self._windows:
            win.mark_stale(False)

        if self._first_run:
            self._autorange_plots()
        self._first_run = False

    def _create_worker(self):
        # See: https://realpython.com/python-pyqt-qthread/
        self._worker_thread = QThread()
        self._worker = SarGuiSimWorker()
        self._worker.moveToThread(self._worker_thread)

        self._worker.progress.connect(self._show_progress)
        self._worker.finished.connect(self._show_results)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._worker_thread.quit)

        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(self._worker_done)

    def _worker_done(self):
        self._worker = None
        self._worker_thread = None

    def _rerun_sim(self):
        if self._worker_thread is not None and self._worker_thread.isRunning():
            print('GUI: Cannot start Simulation, already running')
            return
        print('GUI: Starting Simulation in Thread')
        self._create_worker()
        assert self._worker is not None and self._worker_thread is not None
        self._worker.sim_state = self._state
        self._worker.sim_scene = self._scene
        self._worker.gpu_id = self.args.gpu
        if self._use_loaded_data:
            self._worker.loaded_data = self._loaded_data
        else:
            self._worker.loaded_data = None
        for win in self._windows:
            win.mark_stale(True)
        self._worker_thread.start()

    def _create_mdi(self):
        self.mdi = QMdiArea()
        # Todo: Maybe add logic to switch from and to tabbed view only if window is maximized, dbl-click tab to return
        self.mdi.setViewMode(QMdiArea.TabbedView)
        self.setCentralWidget(self.mdi)

    def _create_status_bar(self):
        status = self.statusBar()
        label = QLabel('Press RUN to simulate')
        label.setAlignment(Qt.AlignRight)
        progress = QProgressBar()
        progress.setMinimum(0)
        progress.setMaximum(1000)
        progress.setValue(0)
        self._progress_bar = progress
        self._progress_label = label
        status.addPermanentWidget(label, stretch=6)
        status.addPermanentWidget(progress, stretch=4)
        self.setStatusBar(status)



def run_gui(args):
    QApplication.setStyle('fusion')

    pg.setConfigOptions(imageAxisOrder='row-major')

    app: QApplication = pg.mkQApp()
    app.setApplicationName('sarsim')
    app.setApplicationDisplayName('SAR-Sim GUI')
    app.setOrganizationName('IMS')

    wnd = SarGuiMainFrame(args)
    wnd.show()

    app.exec_()
