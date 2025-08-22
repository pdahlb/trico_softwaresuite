from PyQt5 import QtWidgets
from AndorUIv4 import Ui_MainWindow
from AndorImageWindow import Ui_CapturedImage
from ROIAnalysis import Ui_ROIAnalysis
import numpy as np
import sys
import math
from PyQt5.QtCore import *
import pandas as pd
import datetime, time
import pathlib
from odemis import model
from odemis.util import fluo, conversion
import traceback
import os
from PIL import Image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.optimize import least_squares, curve_fit
import SymGaussFitting as GF
import pyqtgraph as pg
from timeout import timeout
import errno
from scipy.signal import find_peaks

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        self.is_paused = False
        self.is_killed = False
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['result'] = self.signals.result


    @pyqtSlot()
    def run(self):
        #worker thread checks status of acquire button while emiting rawimag data to contimag
        while self.is_killed == False:
            #allows for pausing of threading
            while self.is_paused:
                time.sleep(0)

            if self.is_killed:
                break

            try:
                 result = self.fn(*self.args, **self.kwargs)
            except:
                traceback.print_exc()
                exctype, value = sys.exc_info()[:2]
                self.signals.error.emit((exctype, value, traceback.format_exc()))
            else:
                a=0
                #self.signals.result.emit(result)  # Return the result of the processing
            finally:
                self.signals.finished.emit()  # Will pass the signal to fn that plots image
                a= 1

    def pause(self):
        self.is_paused = True
        pass

    def resume(self):
        self.is_paused = False
        pass

    def kill(self):
        self.is_killed = True
        pass

#Main Code that established GUI from file ImageWindow
class ImageWindow(QtWidgets.QMainWindow):
    def __init__(self, data, parent=None):
        super().__init__()
        self.ui = Ui_CapturedImage()
        self.ui.setupUi(self, data)

class DiffWindow(QtWidgets.QMainWindow):
    def __init__(self,  parent=None):
        super().__init__()
        self.ui = Ui_ROIAnalysis()
        self.ui.setupUi(self)

class MplCanvas(FigureCanvas):
#Opens Figure that can be used to view derivative
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        #Sets up variables for all light related Enzel Components
        self.FileCt = 0
        self.FOCUS_RANGE = (-0.25e-03, 0.35e-03)
        self.opt_stage = model.getComponent(role="align")
        self.focus = model.getComponent(role="focus")
        self.ccd = model.getComponent(role="ccd")
        self.light = model.getComponent(role="light")
        self.em_filter = model.getComponent(role="filter")

        #starts with all LEDS off
        self.laserstate= [0, 0, 0, 0]
        self.laserstart = [0, 0, 0, 0]
        self.contframes= 0

        super(MainWindow, self).__init__(*args, **kwargs)

        self.ccd.binning.value = (1, 1)
        self.ccd.resolution.value= (2048, 2048)
        self.resolution = self.ccd.resolution.value
        em_choices = self.em_filter.axes["band"].choices.copy()

        # convert any list into tuple, as lists cannot be put in a set
        for k, v in em_choices.items():
            em_choices[k] = conversion.ensure_tuple(v)

        # User friendly dict to select filter positions
        # (4 FM, 1 RLM)
        self.center_wavelengths = {int(1e9 * fluo.get_one_center(v)): k for k, v in em_choices.items()}
        print(self.center_wavelengths)
        exporter = None

        self.setupUi(self)
        self.light.power.value = [self.PwSlider_385.value(), self.PwSlider_470.value(), self.PwSlider_555.value(),
                            self.PwSlider_625.value()]
        self._plot_ref = None

        #set up threading for live image capture
        self.threadpool = QThreadPool()
        self.updatePlot()
        self.AccProgg.setValue(0)

        #declare data variables
        self.MillRateChange()
        self.selected_ROI = self.roi.getArrayRegion(self.data, self.img)
        self.ydata= np.repeat(1, 2)
        self.fulldata= np.repeat(1,2)
        self.ydatatosave= self.ydata
        self.Run_Time = np.linspace(0, 1, self.ydata.size)
        self.ydata_Fit = np.linspace(0, 1, self.ydata.size)
        self.Run_Time_Fit = np.linspace(0, 1, self.ydata.size)
        self.diffdata = np.linspace(0, 1, self.ydata.size+1)
        self.Frame_Times = np.linspace(0, 1, self.ydata.size)

        #Predefines all Plot Objects
        self.AlgFitting_line = self.AlgFitting.plot( self.Run_Time_Fit, self.ydata_Fit )
        self.PSF_Fit_Line = self.AlgFitting.plot(self.Run_Time_Fit, self.ydata_Fit)
        self.RealData_line = self.AlgFitting.plot(self.Run_Time, self.ydatatosave)
        pen = pg.mkPen(color=(255, 255, 255))
        self.Stop_Here = self.AlgFitting.plot(self.Run_Time,self.ydatatosave, pen=pen)

        #Defines all event cases each one willactivate a fn
        self.roi.sigRegionChanged.connect(self.updatePlot)
        self.Acquire.clicked.connect(self.Collect)
        self.Bins.valueChanged.connect(self.CamBin)

        #LEDS
        self.Filters.currentTextChanged.connect(self.filter_change)
        self.LaserOff.clicked.connect(self.laserOffstate)
        self.Laser385.stateChanged.connect(self.laser385State)
        self.Laser470.stateChanged.connect(self.Laser470State)
        self.Laser555.stateChanged.connect(self.Laser555State)
        self.Laser625.stateChanged.connect(self.Laser625State)
        self.PwSlider_385.valueChanged.connect(self.laser385State)
        self.PwSlider_470.valueChanged.connect(self.Laser470State)
        self.PwSlider_555.valueChanged.connect(self.Laser555State)
        self.PwSlider_625.valueChanged.connect(self.Laser625State)
        #objective
        self.StepSize.valueChanged.connect(self.StepSizeState)
        self.PosX.clicked.connect(self.PosXChange)
        self.PosY.clicked.connect(self.PosYChange)
        self.NegX.clicked.connect(self.NegPosXChange)
        self.NegY.clicked.connect(self.NegPosYChange)
        self.PosFocus.clicked.connect(self.PosFocusChange)
        self.NegFocus.clicked.connect(self.NegFocusChange)
        #Meta Data
        self.Save.clicked.connect(self.savedata)
        self.TopDown.clicked.connect(self.TopDownChg)
        # Fitting Params
        self.Fit_Button.clicked.connect(self.FitPSF)
        self.MillTime.valueChanged.connect(self.MillRateChange)
        self.Wave.valueChanged.connect(self.MillRateChange)
        self.YSize.valueChanged.connect(self.MillRateChange)
        self.MillTime.valueChanged.connect(self.MillRateChange)
        self.PSF_X_Y.clicked.connect(self.ChangePSFPlot)
        self.Clear.clicked.connect(self.cleargraph)

    #Optical Adjustments of Frequency seen in Mill Algorithm
    def MillRateChange(self):
        Time = (math.trunc(self.MillTime.value()) * 60) + (100 * (self.MillTime.value() % 1))
        n = 1.25  # index of refraction Kofman, Vincent, et al. "The refractive index of amorphous and crystalline water ice in the UV–vis." The Astrophysical Journal 875.2 (2019): 131.
        theta = 10  # angle in degrees for milling
        self.MR = (self.YSize.value() * 1000) / Time  # milling rate in nm per second
        self.MillingRate_LCD.display(self.MR)
        pf = 2 * math.pi
        ActWave = self.Wave.value()
        fr = self.IntTime.value()  # frame rate as measured from time stamps
        self.ExpFrequency = ActWave / ((pf / fr) * self.MR * n * (1 / math.cos(math.radians(theta))) * 2)
        self.ExpFreq_LCD.display(self.ExpFrequency)

    def TopDownChg(self):
        self.BottomUp.setChecked(False)

    #Objective move events
    def PosFocusChange(self):
        focusmove= np.multiply(self.StepSize.value(), 1e-9)
        f= self.focus.moveRel({"z": focusmove})
        f.result()

    def Y1WindowSizeChange(self):
        self.Y1WindowSizeLabel.display(self.Y1WindowSize.value())
    def Y2WindowSizeChange(self):
        self.Y2WindowSizeLabel.display(self.Y2WindowSize.value())

    def WindowSizeChange(self):
        self.WindowSizeLabel.display(self.WindowSize.value())
    def NegFocusChange(self):
        focusmove = np.multiply(self.StepSize.value(), 1e-9)
        f = self.focus.moveRel({"z": -focusmove})
        f.result()
    def PosXChange(self):
        xmove= np.multiply(self.StepSize.value(), 1e-9)
        f = self.opt_stage.moveRel({"x": xmove, "y": 0})
        f.result()

    def NegPosXChange(self):
        xmove= np.multiply(self.StepSize.value(), 1e-9)
        f = self.opt_stage.moveRel({"x": -xmove, "y": 0})
        f.result()

    def PosYChange(self):
        ymove = np.multiply(self.StepSize.value(), 1e-9)
        f = self.opt_stage.moveRel({"x": 0, "y": ymove})
        f.result()

    def NegPosYChange(self):
        ymove = np.multiply(self.StepSize.value(), 1e-9)
        f = self.opt_stage.moveRel({"x": 0, "y": -ymove})
        f.result()

    #Clear ROI graph
    def cleargraph(self):
        self.ydatatosave= self.ydata
        self.Run_Time= np.linspace(0, 1, self.ydata.size)
        self.diffdata = np.linspace(0, 1, self.ydata.size+1)
        self.contframes= 0
        self.AlgFitting.clear()
        self.AlgFitting_line = self.AlgFitting.plot(self.Run_Time_Fit, self.ydata_Fit)
        self.PSF_Fit_Line = self.AlgFitting.plot(self.Run_Time_Fit, self.ydata_Fit)
        self.RealData_line = self.AlgFitting.plot(self.Run_Time, self.ydatatosave)
        pen = pg.mkPen(color=(255, 255, 255))
        self.Stop_Here = self.AlgFitting.plot(self.Run_Time, self.ydatatosave, pen=pen)


    #Changes Camera Binning
    def CamBin(self):
        a = self.Bins.value()
        self.ccd.binning.value = (a, a)
        self.img.setScaledMode()
        a= self.ccd.resolution.value
        self.Res.display(str(a[0]))

    #Step size slider display value
    def StepSizeState(self):
        x= self.StepSize.value()
        self.StepSizeInd.display(x)

    #LED Events (on/off of LED and power)
    #multiplicaion rough guess of power, needs to be readjusted
    def laser385State(self, checked):
        x = self.PwSlider_385.value() / 100
        a = np.multiply(-62.53, x**2) + np.multiply(387.49, x)
        self.PowerNumber385.display(a)
        self.PowerNumberAc385.display(x)
        if self.Laser385.isChecked():
            self.light.power.value[0] = self.PwSlider_385.value()/100

        else:
            self.light.power.value[0] = 0

    def Laser470State(self, checked):
        x = self.PwSlider_470.value() / 100
        a = np.multiply(661.07, x**3) + np.multiply(-693.70,  x**2) + np.multiply(448.99, x)
        self.PowerNumberAc470.display(x)
        self.PowerNumber470.display(a)
        if self.Laser470.isChecked():
            self.light.power.value[1] = self.PwSlider_470.value()/100

        else:
            self.light.power.value[1] = 0
    def Laser555State(self, checked):
        x = self.PwSlider_555.value() / 100
        a = np.multiply(-57.34, x**3) + np.multiply(67.90, x**2) + np.multiply(92.47, x)
        self.PowerNumber_555.display(a)
        self.PowerNumberAc555.display(x)
        if self.Laser555.isChecked():
            self.light.power.value[2] = self.PwSlider_555.value()/100

        else:
            self.light.power.value[2] = 0


    def Laser625State(self, checked):
        x = self.PwSlider_625.value() / 100
        a = np.multiply(299.33, x**3) + np.multiply(-87.78, x**2) + np.multiply(255.50, x)
        self.PowerNumber_625.display(a)
        self.PowerNumberAc625.display(x)
        if self.Laser625.isChecked():
            self.light.power.value[3] = self.PwSlider_625.value()/100

        else:
            self.light.power.value[3] = 0

    def laserOffstate(self):
        self.light.power.value= [0, 0, 0, 0]
        self.Laser385.setChecked(False)
        self.Laser470.setChecked(False)
        self.Laser555.setChecked(False)
        self.Laser625.setChecked(False)


    def filter_change(self, index):
        filt= self.Filters.currentText()
        filt= filt[:3]
        if filt == 'Pas':
            print(self.center_wavelengths[5000])
            f = self.em_filter.moveAbs({"band": self.center_wavelengths[5000]})
            f.result()
        else:
            print(self.center_wavelengths[int(filt)])
            f = self.em_filter.moveAbs({"band": self.center_wavelengths[int(filt)]})
            f.result()

    #ROI meaning live sampling
    def plotsample(self):
        # Drop off the first y element, append a new one.
        self.selected_ROI = self.roi.getArrayRegion(self.data, self.img)
        self.Frame_Times = np.append(self.Frame_Times, (self.Timer.elapsed() / 1000))

        #Allows buffer to form before fitting 5 can be any number
        if self.imagenumb >5 and self.Begin_Fit.isChecked():
            #Wont fit unless an LED is on (gets rid of null frames)
            if self.Laser385.isChecked() or self.Laser470.isChecked() or self.Laser555.isChecked() or self.Laser625.isChecked():
                #Will try to fit the PSF if failed add to the mean plot instead
                try:
                    self.Run_Time_Fit = np.append(self.Run_Time_Fit, (self.Timer.elapsed() / 1000))
                    self.Fit = self.Gauss2DFit(self.selected_ROI, 0)
                    a = np.mean(self.Fit)
                    b = np.mean(a)
                    self.PSFFitting.setImage(self.Fit)
                    self.ydata_Fit = np.append(self.ydata_Fit, b)
                    line = pg.mkPen(color=(0, 255, 0), width=2)
                    self.PSF_Fit_Line.setData(self.Run_Time_Fit, self.ydata_Fit, pen=line)

                except:
                    a = self.selected_ROI.mean(axis=0)
                    b = a.mean(axis=0)
                    self.ydata_Fit = np.append(self.ydata_Fit, b)
                    line = pg.mkPen(color=(0, 255, 0), width=2)
                    self.PSF_Fit_Line.setData(self.Run_Time_Fit, self.ydata_Fit, pen=line)

                if np.size(self.ydata_Fit) == 151:
                    print('start')

              #starts fitting for milling alg
                if np.size(self.ydata_Fit) >= 151:
                    self.Analyze()

        else:
            a = self.selected_ROI.mean(axis=0)
            b = a.mean(axis=0)
            self.ydatatosave = np.append(self.ydatatosave, b)
            line = pg.mkPen(color=(255, 0, 0), width=2)
            self.Run_Time= np.append(self.Run_Time, (self.Timer.elapsed()/1000))
            self.RealData_line.setData(self.Run_Time, self.ydatatosave, pen=line)
            if self.imagenumb > 30:
                self.AlgFitting.setYRange(np.min(self.ydatatosave[-30:-1]), np.max(self.ydatatosave[-30:-1]))

        self.AlgFitting.setYRange(self.Alg_YLimMin.value(), self.Alg_YLim.value())
    ##@timeout(1, os.strerror(errno.ETIMEDOUT))
    def Analyze(self):

        #Marks window of fitting, going to eventually make this adjustable
        sub = 150
        i = np.size(self.ydata_Fit) - sub
        Subregion = self.ydata_Fit[i:-1]
        #(xo, t, w, phi, r, offset) and limits
        Lower_1 = [-50, -np.inf, self.ExpFrequency - (self.ExpFrequency * 0.1), 0, -0.5, 0]
        Upper_1 = [np.inf, np.inf, self.ExpFrequency + (self.ExpFrequency * 0.1), 2 * np.pi, 0.5, 1000]

        self.OscGuesses = [self.Run_Time_Fit[-1] - 149, 200, self.ExpFrequency, 0, np.mean(np.diff(Subregion,5)), np.min(Subregion)]

        # Attempts fit
        try:

            self.StopFit_1, pconv = curve_fit(GF.SymGaussFitting.MillAlg, self.Run_Time_Fit[i:-1],
                                              Subregion, p0=self.OscGuesses, method='trf',
                                              bounds=(Lower_1, Upper_1), maxfev=1200)
            Fit1 = GF.SymGaussFitting.MillAlg(self.Run_Time_Fit[i:-1], self.StopFit_1[0],
                                                self.StopFit_1[1], self.StopFit_1[2], self.StopFit_1[3],
                                                self.StopFit_1[4], self.StopFit_1[5])

            residuals_1 = Subregion - Fit1
            #Recovers Goodness of Fit
            try:
                ss_res = np.sum(residuals_1 ** 2)
                ss_tot = np.sum((Subregion - np.mean(Subregion)) ** 2)
                r_squared_1 = 1 - (ss_res / ss_tot)
            except:
                r_squared_1 = 0

            # phase shift
            self.OscGuesses[3] = self.OscGuesses[3] + np.pi
            self.StopFit_2, pconv = curve_fit(GF.SymGaussFitting.MillAlg, self.Run_Time_Fit[i:-1],
                                              Subregion, bounds=(Lower_1, Upper_1), method='trf',
                                              p0=self.OscGuesses, maxfev=1200)
            Fit2 = GF.SymGaussFitting.MillAlg(self.Run_Time_Fit[i:-1], self.StopFit_2[0],
                                                self.StopFit_2[1], self.StopFit_2[2], self.StopFit_2[3],
                                                self.StopFit_2[4], self.StopFit_2[5])

            residuals_2 = Subregion - Fit2
            try:
                ss_res = np.sum(residuals_2 ** 2)
                ss_tot = np.sum((Subregion - np.mean(Subregion)) ** 2)
                r_squared_2 = 1 - (ss_res / ss_tot)
            except:
                r_squared_2 = 0

            #Has option to ignore previous fitting results to start again at original guesses
            if self.Reset_Fit.isChecked():
                self.OscGuesses = [self.Run_Time_Fit[-1] - 10, 200, 0, self.ExpFrequency,
                                  np.mean(np.diff(Subregion, 5)), np.min(Subregion)]
            else:
                #Compares Fit 1 and Fit 2 (phase shifts) and accepts the one with better fit determined by r^2
                if 0.1 < r_squared_1 < 1 or 0.1 < r_squared_2 < 1:
                    if r_squared_1 > r_squared_2:
                        self.OscGuesses = self.StopFit_1
                        r_squared = r_squared_1
                    else:
                        self.OscGuesses = self.StopFit_2
                        r_squared = r_squared_2
                else:
                    self.OscGuesses = [self.Run_Time_Fit[-1] - 149, 200, self.ExpFrequency, 0,
                                       np.mean(np.diff(Subregion, 5)), np.min(Subregion)]
                    r_squared = 0

        #Uses fitting param to get a line
        except:
            self.OscGuesses = [self.Run_Time_Fit[-1] - 10, 200, self.ExpFrequency, 0,
                               np.mean(np.diff(Subregion)), np.min(Subregion)]
            CurrentFit = GF.SymGaussFitting.MillAlg(self.Run_Time_Fit[i:-1], self.OscGuesses[0],
                                                  self.OscGuesses[1], self.OscGuesses[2],
                                                  self.OscGuesses[3], self.OscGuesses[4],
                                                  self.OscGuesses[5])
            r_squared = 0
        else:
            line = pg.mkPen(color=(255, 255, 255))
            CurrentFit = GF.SymGaussFitting.MillAlg(self.Run_Time_Fit[i:-1], self.OscGuesses[0],
                                                  self.OscGuesses[1], self.OscGuesses[2],
                                                  self.OscGuesses[3], self.OscGuesses[4],
                                                  self.OscGuesses[5])
            self.AlgFitting_line.setData(self.Run_Time_Fit[i:-1], CurrentFit, pen=line)
            self.AlgFitting.setYRange(0,  self.Alg_YLim.value())
        #Takes Amplitude/Avg to get number to threshold alg criteria
        MillCheck = (np.max(CurrentFit[10:-1]) - np.min(CurrentFit[10:-1])) / np.mean(CurrentFit)

        #Number semi arbituary but determined from Fresnel, while maintaining that it was done with a semi "decent" fit
        if MillCheck > 0.14 and r_squared > 0.4:
            Start = self.Run_Time_Fit[i]
            Spacing = np.mean(np.diff(self.Run_Time_Fit))
            #Extends fit to guess where the next trough will be to stop fitting adds 150 pts again can be adjusted
            End = self.Run_Time_Fit[-1] + Spacing * 150


            Time_Extend = np.linspace(Start, End,
                                      int(np.floor(End - (self.Run_Time_Fit[i]) / Spacing)))
            CurrentFit = GF.SymGaussFitting.MillAlg(Time_Extend, self.OscGuesses[0], self.OscGuesses[1],
                                                      self.OscGuesses[2], self.OscGuesses[3],

                                                      self.OscGuesses[4], self.OscGuesses[5])
            peaks, properties = find_peaks(CurrentFit)
            line = pg.mkPen(color=(255, 255, 255))
            self.AlgFitting_line.setData(Time_Extend, CurrentFit, pen=line)
            self.Stop_Here.setData(Time_Extend[peaks], CurrentFit[peaks], symbol='x', symbolSize=4,
                                   symbolBrush='b')
            print('Stop Milling at')
            #prints estimate of distance to mill left (not robust yet)
            dist = np.multiply(self.Run_Time_Fit[-1], self.MR) - np.multiply(Time_Extend[peaks], self.MR)
            print(dist)
            self.Mill_Goal.display(dist[3])

    #Check X,Y dimension fits
    def ChangePSFPlot(self):
        if self.PSF_X_Y.isChecked() == True:
            self.PSFFit.clear()
            line = pg.mkPen(color=(255, 0, 0))
            B = np.mean(self.Fit_PSF, 1)
            self.PSFFit.plot(B, pen = line)
            line  = pg.mkPen(color=(0, 0, 255))
            B = np.mean(self.selected_ROI, 0)
            self.PSFFit.plot(B, pen = line)
        else:
            self.PSFFit.clear()
            line = pg.mkPen(color=(0, 255, 0))
            B = np.mean(self.Fit_PSF, 0)
            self.PSFFit.plot(B, pen = line)
            line = pg.mkPen(color=(0, 0, 0))
            B = np.mean(self.selected_ROI, 1)
            self.PSFFit.plot(B, pen = line)

    #Fits PSF determined by ROI on plot
    def FitPSF(self):
        self.PSFFit.clear()
        self.Fit_PSF = self.Gauss2DFit(self.selected_ROI, 1)
        if self.PSF_X_Y.isChecked() == True:
            line = pg.mkPen(color=(255, 0, 0))
            B = np.mean(self.Fit_PSF, 1)
            self.PSFFit.plot(B, pen = line)
            line  = pg.mkPen(color=(0, 0, 255))
            B = np.mean(self.selected_ROI, 0)
            self.PSFFit.plot(B, pen = line)
        else:
            self.PSFFit.clear()
            line = pg.mkPen(color=(0, 255, 0))
            B = np.mean(self.Fit_PSF, 0)
            self.PSFFit.plot(B, pen = line)
            line = pg.mkPen(color=(0, 0, 0))
            B = np.mean(self.selected_ROI, 1)
            self.PSFFit.plot(B, pen = line)

    #Fitting PSF of ROI optional timeout decorator in case you need to break function
    #@timeout(0.8, os.strerror(errno.ETIMEDOUT))
    def Gauss2DFit(self, data, offset):
        x = np.linspace(0, np.size(data, 0), num=np.size(data, 0))
        y = np.linspace(0, np.size(data, 1), num=np.size(data, 1))
        x, y = np.meshgrid(x, y)
        #Making the 2D data 1D
        fitdata = data.ravel()

        #Guesses for the fit (A, X0 , Y0, sig_x, C)
        self.InitialGuesses = (self.Gauss_A.value(), self.Gauss_X0.value(), self.Gauss_Y0.value(), self.GaussSig_X.value() , self.Gauss_C.value())
        LowBound = [0, 1, 1, 0.5, 0]
        Upbound = [np.inf, 7, 7, 8, np.inf]
        try:
            popt, pcov = curve_fit(GF.SymGaussFitting.SymGaussFn, (x, y), fitdata, bounds=(LowBound, Upbound), method ='trf', p0=self.InitialGuesses, maxfev=800)
        except:
            print('PSF no Fit')
        else:

            #if fit is successful defines the results as new values
            self.Gauss_A.setValue(popt[0])
            self.Gauss_X0.setValue(popt[1])
            self.Gauss_Y0.setValue(popt[2])
            self.GaussSig_X.setValue(popt[3])
            self.Gauss_C.setValue(popt[4])
            self.InitialGuesses = (self.Gauss_A.value(), self.Gauss_X0.value(), self.Gauss_Y0.value(), self.GaussSig_X.value(),self.Gauss_C.value())
            final_fit = GF.SymGaussFitting.SymGaussFn((x, y), self.Gauss_A.value(), self.Gauss_X0.value(), self.Gauss_Y0.value(), self.GaussSig_X.value(), self.Gauss_C.value()*offset)

            #Rewraps the data to be 2D
            return final_fit.reshape(np.size(x, 0), np.size(x, 1))
    def savedata(self):
        self.CollectMeta()
        path = str(pathlib.Path(__file__).parent.resolve())+ '/SavedData'
        currentDateTime = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%p")
        DF = pd.DataFrame(self.ydatatosave)
        os.makedirs(path +  self.fileloc.text())
        a =  os.path.join(path, self.fileloc.text() )
        DF.to_csv(a + currentDateTime + "IMAGE.tiff", index=False, date_format= '%d/%m/%y')
        DF = pd.DataFrame(list(zip(self.headers, self.MetaData, self.On)))
        DF.to_csv(a + currentDateTime+ "Meta.csv", index=False, date_format= '%d/%m/%y')

    def CollectMeta(self):
        #collects metadate from capture
        a = self.ccd.resolution.value
        self.On= [self.Laser385.isChecked(), self.Laser470.isChecked(),self.Laser555.isChecked(),self.Laser625.isChecked(), 'LP nm', 'pxs', 's', '']
        self.MetaData= [str(self.PowerNumber385.value()), str(self.PowerNumber470.value()), str(self.PowerNumber_555.value()), str(self.PowerNumber_625.value()), self.Filters.currentText(), str(a[0])+ 'x'+ str(a[1]), str( self.ccd.exposureTime.value), str(self.contframes), str(self.MR)]
        self.headers = ['385 nm', '470 nm', '555 nm', '625 nm', 'Filter',  'Resolution', 'Integration Time', 'Frames', 'Mill Rate' ]

    def updatePlot(self):
        # getting the selected region by the roi
            self.selected_ROI = self.roi.getArrayRegion(self.data, self.img)
            self.ROIGraph.plot(self.selected_ROI.mean(axis=0), clear=True)

    def ExpTime(self):
        self.cam.ExposureTime = self.IntTime.value()

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def progress_fn(self, n):
        print("%d%% done" % n)

    def Collect(self):
        self.imagenumb = 1
        if self.savecheck.isChecked():
            #Opens INterface to select folder to save to must have file name in the text box already typed
            self.Frame_Times = np.linspace(0, 1, self.ydata.size)
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            self.dirname = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose", "", options=options)
            directory = QDir(self.dirname)

            if self.FileCt == 0:
                QDir.mkdir(directory, self.fileloc.text())
                self.dirname =  self.dirname, self.fileloc.text()
            else:
                QDir.mkdir(directory, self.fileloc.text()+ '_' + str(self.FileCt))
                self.dirname =  self.dirname, self.fileloc.text()+ '_' + str(self.FileCt)
            self.FileCt = self.FileCt + 1
            dir = self.dirname[0] + '/' + self.dirname[1]+ '/' + str(self.fileloc.text())
            self.CollectMeta()
            DF = pd.DataFrame(list(zip(self.headers, self.MetaData, self.On)))
            DF.to_csv(dir + '_' + str(self.FileCt) +'_Optical_Meta.csv')       
        a= (self.IntTime.value())
        self.ccd.exposureTime.value = a
        
        if self.Cont.isChecked():
            data = {'MillOn': [0],
                    'Direction': [0],
                    'Time': [0.0],
                    'MillingRate': [0.0]}
            self.MetaToSave = pd.DataFrame(data)
            #self.MetaToSave.to_csv(r'self.dirname' + 'Meta.csv')
            self.Timer = QElapsedTimer()
            self.Timer.start()
            #Sets threading and uses worker fn (above) to set its task to contrun fn
            worker = Worker(self.contrun)
            #Will send the worker results once complete to function contimage
            worker.signals.result.connect(self.contimage)
            #Stopping threading
            self.Stop.pressed.connect(worker.kill)
            #starts threading
            self.threadpool.start(worker)
        else:
            print('a')
            self.AquireSeries(self)
            #self.ImgViewer.ui.CapturePlot.addItem(rawimg)

    #Thread Fn that acquires data
    def contrun(self, result):
        self.im = self.ccd.data.get()
        result.emit(np.rot90(self.im, k = 3))

    #Non-realtime acquisition
    def AquireSeries(self, result):
        frames  = self.Frames.value()
        AqFr = np.rot90(self.ccd.data.get(), k =3)
        c = self.ccd.resolution.value
        for i in range(frames):
            im = np.rot90(self.ccd.data.get(), k =3)
            AqFr = np.append(AqFr, im)
            ##
            a = Image.fromarray(im)
            a.save(self.dirname[0] + '/' + self.dirname[1] + '/' + str(self.fileloc.text()) + '_' + str(
                self.imagenumb) + '.tiff', format="tiff")
            self.imagenumb = self.imagenumb+1
            prog = ((i+1)/frames)*100
            self.AccProgg.setValue(int(prog))
        self.data= np.reshape(AqFr, [-1, c[0], c[1]])
        self.ImgViewer = ImageWindow((self.data))

        self.ImgViewer.show()


    def contimage(self, rawimg):
        
        if self.savecheck.isChecked():
            if self.TopDown.isChecked():
                Direction = -1
            else:
                Direction = 1
            if self.Laser385.isChecked() or self.Laser470.isChecked() or self.Laser555.isChecked() or self.Laser625.isChecked():
                self.imagenumb = self.imagenumb + 1
                a =  Image.fromarray(rawimg)
                dir = self.dirname[0] + '/' + self.dirname[1]+ '/' + str(self.fileloc.text())
                a.save( dir + '_' + str(self.imagenumb) + '.tiff' , format = "tiff")
                data = {'MillOn': [self.isMilling.isChecked()],
                        'Direction': [Direction],
                        'Time': [(self.Timer.elapsed() / 1000)],
                        'MillingRate': [self.MR]}
                #data  = [self.isMilling.isChecked(), Direction, (self.Timer.elapsed() / 1000),self.MR]
                data = pd.DataFrame(data)
                self.MetaToSave = pd.concat([self.MetaToSave, data], axis=0)
                self.MetaToSave.to_csv(dir + '_' + str(self.FileCt) +'_Meta.csv')             
            else:
                print('empty')
                
        self.contframes = self.contframes + 1
        self.img.setImage(rawimg)
        self.data= rawimg
        self.plotsample()





if __name__ == "__main__":
    # Create the application
    app = QtWidgets.QApplication(sys.argv)
    # Create and show the application's main window

    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())
    cam.startcam()
