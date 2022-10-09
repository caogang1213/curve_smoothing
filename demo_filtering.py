#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import math, random
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

__appname__ = '曲线平滑滤波工具'

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle(__appname__)

        self.filename = None
        self.loadedDataFrame = None
        self.selected_column = 'deltaz'
        self.algorithm = 'lowpass'

        ### svagol
        self.savgolfilterChecklist = QCheckBox(u'Savgol-Golay滤波')
        self.savgolfilterChecklist.setChecked(False)
        self.savgolfilterChecklist.stateChanged.connect(self.onStateChangeforAlgorithm)

        self.windowlengthLabel = QLabel('滤波窗口长度')
        self.windowlengthSpinbox = QSpinBox()     
        self.windowlengthSpinbox.setEnabled(False)
        self.windowlengthSpinbox.setRange(1, 151)
        self.windowlengthSpinbox.setSingleStep(2)
        self.windowlengthSpinbox.setValue(5)
        # self.windowlengthSpinbox.valueChanged.connect()

        self.polyorderLabel = QLabel('多项式阶数')
        self.polyorderSpinbox = QSpinBox()     
        self.polyorderSpinbox.setEnabled(False)
        self.polyorderSpinbox.setRange(1, 7)
        self.polyorderSpinbox.setSingleStep(1)
        self.polyorderSpinbox.setValue(2)
        # self.polyorderSpinbox.valueChanged.connect()
        
        ### rolling average
        self.convolveChecklist = QCheckBox(u'滑动平均滤波')
        self.convolveChecklist.setChecked(False)
        self.convolveChecklist.stateChanged.connect(self.onStateChangeforAlgorithm)

        self.kernelsizeLabel = QLabel('卷积核大小')
        self.kernelsizeSpinbox = QSpinBox()     
        self.kernelsizeSpinbox.setEnabled(False)
        self.kernelsizeSpinbox.setRange(1, 200)
        self.kernelsizeSpinbox.setSingleStep(1)
        self.kernelsizeSpinbox.setValue(10)
        # self.kernelsizeSpinbox.valueChanged.connect()

        self.convolvemodeLabel = QLabel('卷积核模式')
        self.convolvemodeCombobox= QComboBox()
        self.convolvemodeCombobox.setEnabled(False)
        self.convolvemodeCombobox.addItems(['等长', '全长', '有效'])
        # self.convolvemodeCombobox.currentIndexChanged.connect()

        ### lowpass filter
        self.lowpassChecklist = QCheckBox(u'低通滤波')
        self.lowpassChecklist.setChecked(True)
        self.lowpassChecklist.stateChanged.connect(self.onStateChangeforAlgorithm)

        self.cutoffLabel = QLabel('截止频率')
        self.cutoffSpinbox = QSpinBox()     
        self.cutoffSpinbox.setEnabled(True)
        self.cutoffSpinbox.setRange(10, 500)
        self.cutoffSpinbox.setSingleStep(1)
        self.cutoffSpinbox.setValue(10)
        # self.cutoffSpinbox.valueChanged.connect()

        self.scalarLabel = QLabel('标量系数')
        self.scalarSpinbox= QSpinBox()   
        self.scalarSpinbox.setRange(1, 1000)
        self.scalarSpinbox.setSingleStep(1)
        self.scalarSpinbox.setValue(1000)
        self.scalarSpinbox.setEnabled(False)
        
        lowpassLayout = QVBoxLayout()
        lowpassLayout.addWidget(self.lowpassChecklist)
        lowpassLayout.addWidget(self.cutoffLabel)
        lowpassLayout.addWidget(self.cutoffSpinbox)
        lowpassLayout.addWidget(self.scalarLabel)
        lowpassLayout.addWidget(self.scalarSpinbox)

        lowpassContainer = QWidget()
        lowpassContainer.setLayout(lowpassLayout)

        self.lowpassDock = QDockWidget(u'', self)
        self.lowpassDock.setObjectName(u'')
        self.lowpassDock.setWidget(lowpassContainer)

        svagolLayout = QVBoxLayout()
        svagolLayout.addWidget(self.savgolfilterChecklist)
        svagolLayout.addWidget(self.windowlengthLabel)
        svagolLayout.addWidget(self.windowlengthSpinbox)
        svagolLayout.addWidget(self.polyorderLabel)
        svagolLayout.addWidget(self.polyorderSpinbox)

        svagolContainer = QWidget()
        svagolContainer.setLayout(svagolLayout)

        self.svagolDock = QDockWidget(u'', self)
        self.svagolDock.setObjectName(u'')
        self.svagolDock.setWidget(svagolContainer)

        convolveLayout = QVBoxLayout()
        convolveLayout.addWidget(self.convolveChecklist)
        convolveLayout.addWidget(self.kernelsizeLabel)
        convolveLayout.addWidget(self.kernelsizeSpinbox)
        convolveLayout.addWidget(self.convolvemodeLabel)
        convolveLayout.addWidget(self.convolvemodeCombobox)

        convolveContainer = QWidget()
        convolveContainer.setLayout(convolveLayout)

        self.convolveDock = QDockWidget(u'', self)
        self.convolveDock.setObjectName(u'')
        self.convolveDock.setWidget(convolveContainer)

        ### checklist 
        self.t1Checklist = QCheckBox(u'温度T1')
        self.t1Checklist.setChecked(False)
        self.t1Checklist.stateChanged.connect(self.onStateChangeforDataColumn)

        self.t2Checklist = QCheckBox(u'温度T2')
        self.t2Checklist.setChecked(False)
        self.t2Checklist.stateChanged.connect(self.onStateChangeforDataColumn)

        self.t3Checklist = QCheckBox(u'温度T3')
        self.t3Checklist.setChecked(False)
        self.t3Checklist.stateChanged.connect(self.onStateChangeforDataColumn)

        self.deltazChecklist = QCheckBox(u'纵向补偿量')
        self.deltazChecklist.setChecked(True)
        self.deltazChecklist.stateChanged.connect(self.onStateChangeforDataColumn)

        datacolumnLayout = QVBoxLayout()
        datacolumnLayout.addWidget(self.t1Checklist)
        datacolumnLayout.addWidget(self.t2Checklist)
        datacolumnLayout.addWidget(self.t3Checklist)
        datacolumnLayout.addWidget(self.deltazChecklist)

        datacolumnContainer = QWidget()
        datacolumnContainer.setLayout(datacolumnLayout)

        self.datacolumnDock = QDockWidget(u'选择数据列', self)
        self.datacolumnDock.setObjectName(u'')
        self.datacolumnDock.setWidget(datacolumnContainer)

        ### textbox
        self.datalistPlaintext = QPlainTextEdit()
        self.datalistPlaintext.setReadOnly(True)

        datalistLayout = QVBoxLayout()
        datalistLayout.addWidget(self.datalistPlaintext)

        datalistContainer = QWidget()
        datalistContainer.setLayout(datalistLayout)

        self.datalistDock = QDockWidget(u'温度补偿数据', self)
        self.datalistDock.setObjectName(u'')
        self.datalistDock.setWidget(datalistContainer)

        ### subplots layout type
        self.verticalsubplotsChecklist = QCheckBox(u'纵向排列')
        self.verticalsubplotsChecklist.setChecked(False)
        self.verticalsubplotsChecklist.stateChanged.connect(self.onStateChangeforSubplotType)

        self.matsubplotsChecklist = QCheckBox(u'矩阵排列')
        self.matsubplotsChecklist.setChecked(True)
        self.matsubplotsChecklist.stateChanged.connect(self.onStateChangeforSubplotType)

        subplotypeLayout = QHBoxLayout()
        subplotypeLayout.addWidget(self.verticalsubplotsChecklist)
        subplotypeLayout.addWidget(self.matsubplotsChecklist)

        ### button
        self.filteringButton = QPushButton('滤波', self)    
        self.filteringButton.clicked.connect(self.filtering)

        filteringLayout = QVBoxLayout()
        filteringLayout.addLayout(subplotypeLayout)
        filteringLayout.addWidget(self.filteringButton)

        filteringContainer = QWidget()
        filteringContainer.setLayout(filteringLayout)

        self.filteringDock = QDockWidget(u'图像排列模式', self)
        self.filteringDock.setObjectName(u'')
        self.filteringDock.setWidget(filteringContainer)

        ### axes paramters
        self.xlimLabel = QLabel('x轴上限')
        self.xlimSpinbox = QSpinBox()     
        self.xlimSpinbox.setEnabled(True)
        self.xlimSpinbox.setRange(-1000, 1000)
        self.xlimSpinbox.setSingleStep(1)
        self.xlimSpinbox.setValue(500)
        self.xlimSpinbox.valueChanged.connect(self.updateAxesLimits)

        self.ylimLabel = QLabel('y轴上限')
        self.ylimSpinbox = QSpinBox()     
        self.ylimSpinbox.setEnabled(True)
        self.ylimSpinbox.setRange(0, 100000)
        self.ylimSpinbox.setSingleStep(1000)
        self.ylimSpinbox.setValue(50000)
        self.ylimSpinbox.valueChanged.connect(self.updateAxesLimits)

        xylimLayout = QVBoxLayout()
        xylimLayout.addWidget(self.xlimLabel)
        xylimLayout.addWidget(self.xlimSpinbox)
        xylimLayout.addWidget(self.ylimLabel)
        xylimLayout.addWidget(self.ylimSpinbox)

        xylimContainer = QWidget()
        xylimContainer.setLayout(xylimLayout)

        self.xylimDock = QDockWidget(u'坐标轴显示范围', self)
        self.xylimDock.setObjectName(u'')
        self.xylimDock.setWidget(xylimContainer)

        self.create_subplots_canvas()

        unicanvasLayout = QVBoxLayout()
        unicanvasLayout.addWidget(self.uniCanvas)

        unicanvasContainer = QWidget()
        unicanvasContainer.setLayout(unicanvasLayout)

        self.setCentralWidget(unicanvasContainer)
        self.addDockWidget(Qt.TopDockWidgetArea, self.lowpassDock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.svagolDock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.convolveDock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.datacolumnDock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.xylimDock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.filteringDock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.datalistDock)

        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        # self.DisplayDock.setFeatures(self.DisplayDock.features() ^ self.dockFeatures)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        """
        Drop files directly onto the widget
        File locations are stored in self.fname
        :param e:
        :return:
        """
        if e.mimeData().hasUrls:
            e.setDropAction(Qt.CopyAction)
            e.accept()
            # Workaround for OSx dragging and dropping
            for url in e.mimeData().urls():
               self.filename = url.toLocalFile()
               self.load_data_into_dataframe()
        else:
            e.ignore()

    def create_subplots_canvas(self):
        self.uniFigure = plt.figure()
        self.uniCanvas = FigureCanvas(self.uniFigure)

    def update_subplots_canvas(self):
        self.uniFigure.clear()
        if self.verticalsubplotsChecklist.isChecked() == True:
            self.axRegression = self.uniFigure.add_subplot(411)
            self.axDistribution = self.uniFigure.add_subplot(412)
            self.axFftBefore = self.uniFigure.add_subplot(413)
            self.axFftafter = self.uniFigure.add_subplot(414)
        elif self.matsubplotsChecklist.isChecked() == True:
            self.axRegression = self.uniFigure.add_subplot(221)
            self.axDistribution = self.uniFigure.add_subplot(222)
            self.axFftBefore = self.uniFigure.add_subplot(223)
            self.axFftafter = self.uniFigure.add_subplot(224)

    def updateAxesLimits(self):
        xlimvalue = int(self.xlimSpinbox.value())
        self.axFftBefore.set_xlim(-xlimvalue, xlimvalue)
        ylimvalue = int(self.ylimSpinbox.value())
        self.axFftBefore.set_ylim(0, ylimvalue)

        xlimvalue = int(self.xlimSpinbox.value())
        self.axFftafter.set_xlim(-xlimvalue, xlimvalue)
        ylimvalue = int(self.ylimSpinbox.value())
        self.axFftafter.set_ylim(0, ylimvalue)

        self.uniCanvas.draw()

    def onStateChangeforSubplotType(self, state):
        if state == Qt.Checked:
            if self.sender() == self.verticalsubplotsChecklist:
                self.verticalsubplotsChecklist.setChecked(True)
                self.matsubplotsChecklist.setChecked(False)
            elif self.sender() == self.matsubplotsChecklist:
                self.verticalsubplotsChecklist.setChecked(False)
                self.matsubplotsChecklist.setChecked(True)

    def onStateChangeforAlgorithm(self, state):
        if state == Qt.Checked:
            if self.sender() == self.savgolfilterChecklist:
                self.algorithm = 'Savgol'

                self.windowlengthSpinbox.setEnabled(True)
                self.polyorderSpinbox.setEnabled(True)

                self.convolveChecklist.setChecked(False)
                self.kernelsizeSpinbox.setEnabled(False)
                self.convolvemodeCombobox.setEnabled(False)

                self.lowpassChecklist.setChecked(False)
                self.cutoffSpinbox.setEnabled(False)

            elif self.sender() == self.convolveChecklist:
                self.algorithm = 'RollingAverage'

                self.kernelsizeSpinbox.setEnabled(True)
                self.convolvemodeCombobox.setEnabled(True)

                self.savgolfilterChecklist.setChecked(False)
                self.windowlengthSpinbox.setEnabled(False)
                self.polyorderSpinbox.setEnabled(False)

                self.lowpassChecklist.setChecked(False)
                self.lowpassChecklist.setChecked(False)
                self.cutoffSpinbox.setEnabled(False)
            elif self.sender() == self.lowpassChecklist:
                self.algorithm = 'Lowpass'

                self.cutoffSpinbox.setEnabled(True)

                self.convolveChecklist.setChecked(False)
                self.kernelsizeSpinbox.setEnabled(False)
                self.convolvemodeCombobox.setEnabled(False)

                self.savgolfilterChecklist.setChecked(False)
                self.windowlengthSpinbox.setEnabled(False)
                self.polyorderSpinbox.setEnabled(False)

    def onStateChangeforDataColumn(self, state):
        if state == Qt.Checked:
            if self.sender() == self.t1Checklist:
                self.selected_column = 't1'
                self.t2Checklist.setChecked(False)
                self.t3Checklist.setChecked(False)
                self.deltazChecklist.setChecked(False)

            elif self.sender() == self.t2Checklist:
                self.selected_column = 't2'
                self.t1Checklist.setChecked(False)
                self.t3Checklist.setChecked(False)
                self.deltazChecklist.setChecked(False)

            elif self.sender() == self.t3Checklist:
                self.selected_column = 't3'
                self.t1Checklist.setChecked(False)
                self.t2Checklist.setChecked(False)
                self.deltazChecklist.setChecked(False)

            elif self.sender() == self.deltazChecklist:
                self.selected_column = 'deltaz'
                self.t1Checklist.setChecked(False)
                self.t2Checklist.setChecked(False)
                self.t3Checklist.setChecked(False)

    def filtering(self):
        if self.loadedDataFrame is not None and self.selected_column is not None:
            unfiltered_data = self.loadedDataFrame[self.selected_column]
            filtered_data = None
            freq = None
            sig_fft = None
            sig_fft_filtered = None
            if self.savgolfilterChecklist.isChecked() == True:
                windowlength = int(self.windowlengthSpinbox.value())
                polyorder = int(self.polyorderSpinbox.value())
                filtered_data =  filtering_savgol(unfiltered_data, windowlength, polyorder)
            elif self.convolveChecklist.isChecked() == True:
                kernelsize = int(self.kernelsizeSpinbox.value())
                convolvemode = self.convolvemodeCombobox.currentText()
                filtered_data =  filtering_convolve(unfiltered_data, kernelsize, convolvemode)
            elif self.lowpassChecklist.isChecked() == True:
                cutoff = float(self.cutoffSpinbox.value())
                scalar = int(self.scalarSpinbox.value())
                filtered_data, sig_fft, sig_fft_filtered, freq  =  filtering_lowpass(unfiltered_data, cutoff, scalar)
            self.painting_canvas(unfiltered_data, filtered_data, sig_fft, sig_fft_filtered, freq)

    def painting_canvas(self, unfiltered_data, filtered_data, fft_unfiltered_data, fft_filtered_data,  freq):
        self.update_subplots_canvas()
        self.clearing_canvas()
        self.plot_regression(unfiltered_data, filtered_data)
        self.plot_distribution(unfiltered_data,filtered_data )
        self.plot_sig_fft_before(fft_unfiltered_data, freq)
        self.plot_sig_fft_after(fft_filtered_data, freq)
        self.uniCanvas.draw()

    def plot_regression(self, unfiltered_data, filtered_data):
        xx = range(0, len(unfiltered_data))
        ### regression
        self.axRegression.clear()
        self.axRegression.plot(xx, unfiltered_data, 'g:', label='Before filtering')
        self.axRegression.plot(xx, filtered_data, 'b-', label='After filtering')
        self.axRegression.set_xlabel('Samples', fontsize=8)
        self.axRegression.set_title('%s (Curve Smoothing/Noise Reduction using <%s>)'%(self.selected_column,self.algorithm), fontsize=8)
        if self.selected_column == 'deltaz':
            self.axRegression.set_ylabel('μm', fontsize=8)
        else:
            self.axRegression.set_ylabel('°C', fontsize=8)
        self.axRegression.legend(fontsize=8)

    def plot_distribution(self, unfiltered_data, filtered_data):
        self.axDistribution.hist(unfiltered_data, density=True, histtype='stepfilled', bins=10, alpha=0.6, label='Before Filtering')
        self.axDistribution.hist(filtered_data, density=True, histtype='stepfilled', bins=10, alpha=0.6, label='After Filtering')
        self.axDistribution.legend(fontsize=8)
        self.uniCanvas.draw()

    def plot_sig_fft_before(self, data, freq):
        if freq is not None and data is not None:
            self.axFftBefore.stem(freq, np.abs(data), 'b', markerfmt=" ", basefmt="-b")
            self.axFftBefore.set_xlabel('Freq (Hz, scalar=%s)'%self.scalarSpinbox.value(), fontsize=8)
            self.axFftBefore.set_ylabel('FFT Amplitude |X(freq)|', fontsize=8)
            self.axFftBefore.set_title('Before Filtering', fontsize=8)

            xlimvalue = int(self.xlimSpinbox.value())
            self.axFftBefore.set_xlim(-xlimvalue, xlimvalue)
            ylimvalue = int(self.ylimSpinbox.value())
            self.axFftBefore.set_ylim(0, ylimvalue)

    def plot_sig_fft_after(self, data, freq):
        if freq is not None and data is not None:
            self.axFftafter.stem(freq, np.abs(data), 'b', markerfmt=" ", basefmt="-b")
            self.axFftafter.set_xlabel('Freq (Hz, scalar=%s)'%self.scalarSpinbox.value(), fontsize=8)
            self.axFftafter.set_ylabel('FFT Amplitude |X(freq)|', fontsize=8)
            self.axFftafter.set_title('After Filtering', fontsize=8)

            xlimvalue = int(self.xlimSpinbox.value())
            self.axFftafter.set_xlim(-xlimvalue, xlimvalue)
            ylimvalue = int(self.ylimSpinbox.value())
            self.axFftafter.set_ylim(0, ylimvalue)

    def clearing_canvas(self):
        self.axFftafter.clear()
        self.axFftBefore.clear()
        self.axDistribution.clear()
        self.axRegression.clear()

    def load_data_into_dataframe(self):
        # filepath = '/home/gangcao/source/temperature_analysis/data/data_2.txt'
        if self.filename is not None:
            self.datalistPlaintext.clear()
            self.loadedDataFrame = pd.DataFrame(read_textfiledata_into_list(self.filename), columns=['t1','t2','t3','deltaz'])
            self.populate_datalist_into_plaintext()
            self.filtering()

    def populate_datalist_into_plaintext(self):
        if self.loadedDataFrame is not None:  
            for ind in self.loadedDataFrame.index:
                t1 = self.loadedDataFrame['t1'][ind]
                t2 = self.loadedDataFrame['t2'][ind]
                t3 = self.loadedDataFrame['t3'][ind]
                deltaz = self.loadedDataFrame['deltaz'][ind]
                self.datalistPlaintext.appendPlainText('温度T1(°c):%.2f, 温度T2(°c):%.2f, 温度T3(°c):%.2f, 纵向补偿量(μm):%.2f' %(t1, t2, t3, deltaz))
            self.filtering()

def filtering_lowpass(data, cutoff, scalar):
    from numpy.fft import fft, ifft, fftfreq
    sig_fft = fft(data)
    freq = fftfreq(len(data), d=1./scalar)
    sig_fft_filtered = sig_fft.copy()
    
    # high-pass filter by assign zeros to the 
    # FFT amplitudes where the absolute 
    # frequencies smaller than the cut-off 
    sig_fft_filtered[np.abs(freq) > cutoff] = 0
    filtered_data = ifft(sig_fft_filtered)
    return filtered_data, sig_fft, sig_fft_filtered, freq

def filtering_savgol(data, windowlength, polyorder):
    from scipy.signal import savgol_filter
    return savgol_filter(data, windowlength, polyorder)

def filtering_convolve(data, kernelsize, convolvemode):
    kernel = np.ones(kernelsize) / kernelsize
    return np.convolve(data, kernel, mode=convolvemode)

### read temperature data(txt file) as a list
def read_textfiledata_into_list(filepath):
    fdata = np.loadtxt(filepath, dtype=str, delimiter=',',usecols=(0,1,2,5))
    merge_float_data = []
    for strdata in fdata:
        merge_float_data.append(convertarr_str_to_float(strdata))
    return merge_float_data

### remove prefix and suffix for each column
def convertarr_str_to_float(strarr):
    T1 = float(strarr[0][3:-1])
    T2 = float(strarr[1][3:-1])
    T3 = float(strarr[2][3:-1])
    deltaZ = float(strarr[-1][3:-25])
    return [T1, T2, T3, deltaZ]

def get_main_app(argv=[]):
    app = QApplication(argv)
    win = MainWindow()
    win.show()
    return app, win

def main():
    app, _win = get_main_app(sys.argv)
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main())