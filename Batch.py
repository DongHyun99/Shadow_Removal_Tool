import os
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

class Batch(QMainWindow):
    def __init__(self) :
        super().__init__()
        self.ui = uic.loadUi('UI/batch.ui', self)
        self.actionInteractive.triggered.connect(self.change_ui)
        
    def change_ui(self):
        from Interactive import Interact
        self.interact = Interact()
        self.interact.show()
        self.close()
