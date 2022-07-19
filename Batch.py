import os
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import json
import requests
import cv2

class Batch(QMainWindow):
    def __init__(self) :
        super().__init__()
        self.ui = uic.loadUi('UI/batch.ui', self)
        self.setWindowTitle('ShadowRemoval (Batch mode) Demo')

        # api로 전송할 이미지 directory    
        self.shadowImageDir = 'None'
        self.maskImageDir= 'None'
        self.GTDir = 'None'
        self.saveDir = 'None'
        self.shadowImage = None
        self.maskImage = None
        self.GTImage = None
        self.predictImage = None
        self.inpaintImage = None
        self.shadowImageList = []
        self.maskImageList = []
        self.GTImageList = []
        self.num = 1
        self.ip = 'http://129.254.81.18:50000/'
        self.maskColor = (255, 255, 180)
        self.ax = None
        
        self.loadModels()
        self.menuBar()
        self.selectDirectory()
    
    def menuBar(self):
        self.actionInteractive.triggered.connect(self.change_ui)
        
    def selectDirectory(self):
        self.shadowImageButton.clicked.connect(self.selectShadow)
        self.maskImageButton.clicked.connect(self.selectMask)
        self.GTImageButton.clicked.connect(self.selectGT)
         
    def loadModels(self):
        res = requests.get(self.ip + 'modelList')
        models = json.loads(res.json())
        
        for model in models['detection_model']:
            self.selectDetectionModel.addItem(model)
            
        for model in models['removal_model']:
            self.selectRemovalModel.addItem(model)
    
    # ===================================================================

    def selectShadow(self):
        self.shadowImageDir = QFileDialog.getExistingDirectory(self, 'Select Directory', './')
        
        if self.shadowImageDir != 'None' and self.shadowImageDir != '':
            self.shadowImageName.setText(self.shadowImageDir)
            self.shadowImageList = os.listdir(self.shadowImageDir)
            self.shadowScrollArea.setText(self.shadowImageList)
        
    def selectMask(self):
        self.maskImageDir = QFileDialog.getExistingDirectory(self, 'Select Directory', './')
        
        if self.maskImageDir != 'None' and self.maskImageDir != '':
            self.maskImageName.setText(self.maskImageDir)
            self.maskImageList = os.listdir(self.maskImageDir)
            
    def selectGT(self):
        self.GTDir = QFileDialog.getExistingDirectory(self, 'Select Directory', './')
        
        if self.GTDir != 'None' and self.GTDir != '':
            self.GTImageName.setText(self.GTDir)
            self.GTImageList = os.listdir(self.GTDir)
    
    def change_ui(self):
        from Interactive import Interact
        self.interact = Interact()
        self.interact.show()
        self.close()

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = Batch()
    myWindow.show()
    sys.exit(app.exec())