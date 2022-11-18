# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import uic
from PyQt5.QtWidgets import *


class ShadowRemovalPopup(QDialog):
    """Popup 창 Dialog Class (그림자 제거시 inpainted)
    """
    def __init__(self):
        """UI에서 값을 불러와 사용
        """
        
        super().__init__()
        self.ui = uic.loadUi('UI/ShadowRemovalPopup.ui', self)
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
    """Popup 창 Dialog Class (그림자 제거시 predicted)
    """
    
    def __init__(self):
        """UI에서 값을 불러와 사용
        """
        
        super().__init__()
        self.ui = uic.loadUi('UI/PredictPopup.ui', self)
        self.setWindowTitle('Predict Popup')
        
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.fig.tight_layout()
        self.fig.subplots_adjust(wspace=0.05, hspace = 0)
        self.inpaintingArea.addWidget(self.canvas)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.inpaintingArea.addWidget(self.toolbar)
        self.canvas.flush_events()
