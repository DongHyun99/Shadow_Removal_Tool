import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PIL import ImageColor

class Masking(QDialog) :
    def __init__(self) :
        super().__init__()
        self.ui = uic.loadUi('UI/Masking.ui', self)
        self.setWindowTitle('Masking Setting')
        self.maskCol = (255, 255, 180)
        self.segmentCol = (255, 180, 180)
        
        self.maskColorButton.clicked.connect(self.maskColorFunc)
        self.boundaryColorButton.clicked.connect(self.segmentColorFunc)
        self.segmentationCheckbox.clicked.connect(self.segmentationChecked)
        self.slic.clicked.connect(self.slicToggle)
        self.felzenswalb.clicked.connect(self.felzenswalbToggle)
        self.quickShift.clicked.connect(self.quickShiftToggle)
        self.waterShed.clicked.connect(self.waterShedToggle)
        
        self.slic.setEnabled(False)
        self.felzenswalb.setEnabled(False)
        self.quickShift.setEnabled(False)
        self.waterShed.setEnabled(False)
        
        self.maskColorButton.setStyleSheet(
            'background-color:rgb{};'.format(self.maskCol)
        )
        
        self.boundaryColorButton.setStyleSheet(
            'background-color:rgb{};'.format(self.segmentCol)
        )
        

    def maskColorFunc(self):
        self.maskCol = QColorDialog.getColor().name()
        self.maskCol = ImageColor.getcolor(self.maskCol, 'RGB')
        self.maskColorButton.setStyleSheet(
            'background-color:rgb{};'.format(self.maskCol)
        )
        
    def segmentColorFunc(self):
        self.segmentCol = QColorDialog.getColor().name()
        self.segmentCol = ImageColor.getcolor(self.segmentCol, 'RGB')
        self.boundaryColorButton.setStyleSheet(
            'background-color:rgb{};'.format(self.segmentCol)
        )
        
    def segmentationChecked(self):
        if self.segmentationCheckbox.isChecked():
            self.slic.setEnabled(True)
            self.felzenswalb.setEnabled(True)
            self.quickShift.setEnabled(True)
            self.waterShed.setEnabled(True)
            
        else:
            self.slic.setEnabled(False)
            self.felzenswalb.setEnabled(False)
            self.quickShift.setEnabled(False)
            self.waterShed.setEnabled(False)
            
    def slicToggle(self):
        if self.slic.isChecked():
            if self.felzenswalb.isChecked():
                self.felzenswalb.toggle()
            if self.quickShift.isChecked():
                self.quickShift.toggle()
            if self.waterShed.isChecked():
                self.waterShed.toggle()
                
        elif not self.felzenswalb.isChecked() and not self.quickShift.isChecked() and not self.waterShed.isChecked():
            self.slic.toggle()
            
    def felzenswalbToggle(self):
        if self.felzenswalb.isChecked():
            if self.slic.isChecked():
                self.slic.toggle()
            if self.quickShift.isChecked():
                self.quickShift.toggle()
            if self.waterShed.isChecked():
                self.waterShed.toggle()
                
        elif not self.slic.isChecked() and not self.quickShift.isChecked() and not self.waterShed.isChecked():
            self.slic.toggle()
            
    def quickShiftToggle(self):
        if self.quickShift.isChecked():
            if self.felzenswalb.isChecked():
                self.felzenswalb.toggle()
            if self.slic.isChecked():
                self.slic.toggle()
            if self.waterShed.isChecked():
                self.waterShed.toggle()
                
        elif not self.felzenswalb.isChecked() and not self.slic.isChecked() and not self.waterShed.isChecked():
            self.slic.toggle()
        
    def waterShedToggle(self):
        if self.waterShed.isChecked():
            if self.felzenswalb.isChecked():
                self.felzenswalb.toggle()
            if self.quickShift.isChecked():
                self.quickShift.toggle()
            if self.slic.isChecked():
                self.slic.toggle()
                
        elif not self.felzenswalb.isChecked() and not self.quickShift.isChecked() and not self.slic.isChecked():
            self.slic.toggle()
        
class Visualization(QDialog) :
    def __init__(self) :
        super().__init__()
        self.ui = uic.loadUi('UI/Visualization.ui', self)
        self.setWindowTitle('Visualization Setting')
        
        self.shadowRemovalCheckbox.clicked.connect(self.shadowRemovalToggle)
        self.predictAndInpaintingCheckbox.clicked.connect(self.removalAndInpaintingToggle)
        self.edgeDetectionCheckbox.clicked.connect(self.edgeDetectionToggle)
        
        
    def shadowRemovalToggle(self):
        if self.shadowRemovalCheckbox.isChecked():
            if self.predictAndInpaintingCheckbox.isChecked():
                self.predictAndInpaintingCheckbox.toggle()
            if self.edgeDetectionCheckbox.isChecked():
                self.edgeDetectionCheckbox.toggle()
                
        elif not self.predictAndInpaintingCheckbox.isChecked() and not self.edgeDetectionCheckbox.isChecked():
            self.shadowRemovalCheckbox.toggle()
                         
    def removalAndInpaintingToggle(self):
        if self.predictAndInpaintingCheckbox.isChecked():
            if self.shadowRemovalCheckbox.isChecked():
                self.shadowRemovalCheckbox.toggle()
            if self.edgeDetectionCheckbox.isChecked():
                self.edgeDetectionCheckbox.toggle()
                
        elif not self.shadowRemovalCheckbox.isChecked() and not self.edgeDetectionCheckbox.isChecked():
            self.shadowRemovalCheckbox.toggle()
                
    def edgeDetectionToggle(self):
        if self.edgeDetectionCheckbox.isChecked():
            if self.shadowRemovalCheckbox.isChecked():
                self.shadowRemovalCheckbox.toggle()
            if self.predictAndInpaintingCheckbox.isChecked():
                self.predictAndInpaintingCheckbox.toggle()
        
        elif not self.shadowRemovalCheckbox.isChecked() and not self.predictAndInpaintingCheckbox.isChecked():
            self.shadowRemovalCheckbox.toggle()
            
class Server(QDialog) :
    def __init__(self) :
        super().__init__()
        self.ui = uic.loadUi('UI/Server.ui', self)
        self.setWindowTitle('Server Setting')