import typing

from PyQt5 import QtCore, QtGui
from PyQt5.Qt import Qt
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QPoint
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QLabel, QMdiArea, QMdiSubWindow, QProgressBar, \
    QFormLayout, QSpinBox, QDoubleSpinBox, QWidget, QScrollArea, QPushButton, QFileDialog, QComboBox, QLineEdit
import pyqtgraph as pg
import numpy as np

from . import simstate
from . import siunits
from . import simjob
from . import profiling
from . import sardata


class SarGuiPlotSubWindow(QMdiSubWindow):
    def __init__(self, aspect_lock: bool = True, unit_x: str = 'm', unit_y: str = 'm', variant: str = '') -> None:
        super().__init__()
        self.setWindowTitle('Plot')
        self._data = np.array([[]])
        self._data_tr = QtGui.QTransform()
        self._variant = variant

        self._create_plot_widget()

        if aspect_lock:
            self._p1.setAspectLocked()
        self._p1.setLabel('left', 'Azimuth', unit_x)
        self._p1.setLabel('bottom', 'Range', unit_y)
        self._unit_x = unit_x
        self._unit_y = unit_y

        # self._set_random_data()

    @property
    def data(self) -> np.array:
        return self._data

    @data.setter
    def data(self, data: np.array):
        self._data = data
        self._update_data()

    def set_transform(self, x0, y0, dx, dy):
        tr = QtGui.QTransform()
        tr.translate(x0-dx/2, y0-dy/2).scale(dx, dy)
        self._data_tr = tr
        self._img.setTransform(self._data_tr)

        # Update data cuts
        if self._variant == 'azimuth_comp':
            self._update_cuts()

    def _set_random_data(self):
        # Generate image data
        data = np.random.normal(size=(200, 100))
        data[20:80, 20:80] += 2.
        data = pg.gaussianFilter(data, (3, 3))
        data += np.clip(-30 + np.random.normal(size=(200, 100)) * 10, -90, 0)
        self.set_transform(0, 0, 1, 1)
        self.data = data

    def do_autorange(self):
        self._hist.setLevels(-90, 0)
        self._p1.autoRange()
        if self._variant == 'azimuth_comp':
            self._range_line.setValue(0)
            self._azi_line.setValue(0)

    def _create_plot_widget(self):
        # Interpret image data as row-major instead of col-major

        isolines = False

        self._layout = pg.GraphicsLayoutWidget()

        # A plot area (ViewBox + axes) for displaying the image
        self._p1: pg.PlotItem = self._layout.addPlot(row=1, col=1, title="")

        # Item for displaying image data
        self._img = pg.ImageItem()
        self._p1.addItem(self._img)

        # Two movable lines for "cuts" trough the data
        if self._variant == 'azimuth_comp':
            self._range_line = pg.InfiniteLine(pos=0, movable=True, angle=0)
            self._azi_line = pg.InfiniteLine(pos=0, movable=True, angle=90)
            self._p1.addItem(self._range_line)
            self._p1.addItem(self._azi_line)
            self._range_line.setZValue(10)  # make sure line is drawn above image
            self._azi_line.setZValue(10)
            self._range_line.sigDragged.connect(self._update_cuts)
            self._azi_line.sigDragged.connect(self._update_cuts)

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

        # Patch in the "jet" colormap used in the demo etc.
        pg.graphicsItems.GradientEditorItem.Gradients['jet'] = {'ticks' :[(0.0 / 63, (0, 0, 144)), (7.0 / 63, (0, 0, 255)), (23.0 / 63, (0, 255, 255)), (39.0 / 63, (255, 255, 0)), (55.0 / 63, (255, 0, 0)), (63.0 / 63, (127, 0, 0))], 'mode': 'rgb'}

        # Contrast/color control
        self._hist = pg.HistogramLUTItem()
        self._hist.gradient.loadPreset('jet') #spectrum, viridis
        self._hist.setLevels(-90, 0)
        self._hist.setImageItem(self._img)
        self._hist.axis.setLabel('Magnitude', 'dB')
        self._layout.addItem(self._hist, row=1, col=2)

        if self._variant == 'azimuth_comp':
            self._hist.sigLevelsChanged.connect(self._update_levels)

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
        
        # Update data cuts
        if self._variant == 'azimuth_comp':
            self._update_cuts()

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

    BG_STALE = (64, 0, 0)
    BG_NORMAL = (0, 0, 0)

    def mark_stale(self, stale: bool = True):
        if stale:
            self._layout.setBackground(SarGuiPlotSubWindow.BG_STALE)
        else:
            self._layout.setBackground(SarGuiPlotSubWindow.BG_NORMAL)


class SarGuiSimWorker(QObject):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(float, str)
    sim_state: simstate.SarSimParameterState = None
    loaded_data: sardata.SarData = None

    def run(self):
        ts = profiling.TimeStamper()

        def cb(p, m):
            self.progress.emit(p, m)

        images = simjob.run_sim(self.sim_state, ts, cb, self.loaded_data)
        self.finished.emit(images)


class SarGuiMainFrame(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self._state = simstate.create_state()

        self._loaded_data = None  # sardata.SarData
        self._use_loaded_data: bool = False

        self._create_menu()
        self._param_widgets = self._create_parameter_dock()

        self._create_mdi()
        self._create_status_bar()

        self._plot_window_raw = SarGuiPlotSubWindow(False, 'm', 's', variant='raw')
        self._plot_window_rc = SarGuiPlotSubWindow(False, variant='range_comp')
        self._plot_window_ac = SarGuiPlotSubWindow(variant='azimuth_comp')

        self._plot_window_raw.setWindowTitle('Raw FMCW Data')
        self._plot_window_rc.setWindowTitle('Range Compression')
        self._plot_window_ac.setWindowTitle('Azimuth Compression')

        # Add in order of processing, last will be double size on left if 3 MDIs
        self.mdi.addSubWindow(self._plot_window_raw)
        self.mdi.addSubWindow(self._plot_window_rc)
        self.mdi.addSubWindow(self._plot_window_ac)

        # Thread for worker
        self._worker_thread = None
        self._worker = None

        # See if we had any sim so far (for initial auto-ranging feature etc)
        self._first_run = True
        self._plot_window_raw.mark_stale()
        self._plot_window_rc.mark_stale()
        self._plot_window_ac.mark_stale()

        # Default view
        self.mdi.tileSubWindows()

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

    def _update_gui_values_from_state(self):
        for parameter in self._state.get_parameters():
            if parameter.name not in self._param_widgets:
                continue
            box = self._param_widgets[parameter.name]
            value = self._state.get_value(parameter)
            if parameter.type.type in [float, int]:
                factor = box.property('si_factor')
                box.setValue(value / factor)
            elif parameter.type.type is str:
                if isinstance(box, QComboBox):
                    box.setCurrentText(self._state.get_value(parameter))
                else:
                    box.setText(self._state.get_value(parameter))
            else:
                raise NotImplementedError("Update routine missing!")

    def _save_param_file(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Select parameter file", filter="*.ini")
        if filename:
            if not filename.endswith(".ini"):
                filename = filename + ".ini"
            self._state.write_to_file(filename)

    def on_parameter_spinbox_change(self, symbol: str, value: float):
        print(f'{symbol} = {value}')

    def _create_parameter_dock(self):
        dock = QDockWidget("Parameters", self)
        dock.setFeatures(QDockWidget.DockWidgetFloatable|QDockWidget.DockWidgetMovable)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)

        widgets = {} # Save all widget in a dict, so that we can update them from _load_param_file()
        # TODO: Refactor this into a class?

        for parameter in self._state.get_parameters():
            box = None
            if parameter.type.type is float:
                factor, unit = siunits.choose_si_scale(
                    parameter.default or self._state.get_value(parameter),
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
                box.setValue(self._state.get_value(parameter) / factor)
                box.valueChanged.connect(lambda v, p=parameter, f=factor: self._state.set_value(p, v*f))

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
                box.setValue(self._state.get_value(parameter))
                box.valueChanged.connect(lambda v, p=parameter: self._state.set_value(p, v))

            elif parameter.type.type is str:
                value = self._state.get_value(parameter)
                if value is None:
                    value = ''
                if parameter.type.choices is not None:
                    box = QComboBox()
                    for choice in parameter.type.choices:
                        box.addItem(choice)
                    box.setCurrentText(value)
                    box.currentTextChanged.connect(lambda v, p=parameter: self._state.set_value(p, v))
                else:
                    box = QLineEdit()
                    box.setText(value)
                    box.textChanged.connect(lambda v, p=parameter: self._state.set_value(p, v))
                tool_tip = []
                if parameter.symbol is not None:
                    tool_tip.append(f'{parameter.symbol}')
                if parameter.type.unit is not None:
                    tool_tip.append(f'[{parameter.type.unit}]')
                if len(tool_tip) > 0:
                    box.setToolTip(' '.join(tool_tip))

            else:
                print(f'WARNING: Unsupported type in GUI for state parameter "{parameter.name}"!')
                continue  # others not supported for now in GUI

            form.addRow(parameter.human_name(), box)
            widgets[parameter.name] = box

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
        scroll = QScrollArea(self)
        scroll.setWidget(widget)
        dock.setWidget(scroll)

        self._parameter_dock = dock
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        return widgets

    def _autorange_plots(self):
        self._plot_window_raw.do_autorange()
        self._plot_window_rc.do_autorange()
        self._plot_window_ac.do_autorange()

    def _show_progress(self, progress: float, message: str):
        self._progress_bar.setValue(progress*1000)
        self._progress_label.setText(message)

    def _show_results(self, images: dict):
        print('GUI: Updating Plots')

        def _assign(win: SarGuiPlotSubWindow, img: simstate.SimImage):
            magimg = 20 * np.log10(np.clip(np.abs(img.data), 1e-30, None))
            win.data = np.clip(magimg - np.max(magimg), -240, None)
            #win.set_transform(img.x0, img.y0, img.dx, img.dy)
            win.set_transform(img.y0, img.x0, img.dy, img.dx)

        _assign(self._plot_window_raw, images['raw'])
        _assign(self._plot_window_rc, images['rc'])
        _assign(self._plot_window_ac, images['ac'])
        self._plot_window_raw.mark_stale(False)
        self._plot_window_rc.mark_stale(False)
        self._plot_window_ac.mark_stale(False)

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
        self._worker.sim_state = self._state
        if self._use_loaded_data:
            self._worker.loaded_data = self._loaded_data
        else:
            self._worker.loaded_data = None
        self._plot_window_raw.mark_stale(True)
        self._plot_window_rc.mark_stale(True)
        self._plot_window_ac.mark_stale(True)
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



def run_gui():
    QApplication.setStyle('fusion')

    pg.setConfigOptions(imageAxisOrder='row-major')

    app: QApplication = pg.mkQApp()
    app.setApplicationName('sarsim')
    app.setApplicationDisplayName('SAR-Sim GUI')
    app.setOrganizationName('IMS')

    wnd = SarGuiMainFrame()
    wnd.show()

    app.exec_()