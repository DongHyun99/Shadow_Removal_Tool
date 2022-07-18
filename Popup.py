import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class ShadowRemovalPopup(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi('ShadowRemovalPopup.ui', self)
        self.setWindowTitle('ShadowRemoval Popup')
        
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.fig.tight_layout()
        self.fig.subplots_adjust(wspace=0.05, hspace = 0)
        self.shadowRemovalArea.addWidget(self.canvas)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.shadowRemovalArea.addWidget(self.toolbar)
        self.canvas.flush_events()
        
class PredictPopup(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi('PredictPopup.ui', self)
        self.setWindowTitle('Predict Popup')
        
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.fig.tight_layout()
        self.fig.subplots_adjust(wspace=0.05, hspace = 0)
        self.inpaintingArea.addWidget(self.canvas)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.inpaintingArea.addWidget(self.toolbar)
        self.canvas.flush_events()
