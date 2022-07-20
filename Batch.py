import os
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import json
import requests
import cv2
import numpy as np
import glob
import zipfile
import base64

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_interactions import zoom_factory

class Batch(QMainWindow):
    def __init__(self) :
        super().__init__()
        self.ui = uic.loadUi('UI/batch.ui', self)
        self.setWindowTitle('ShadowRemoval (Batch mode) Demo')

        # api로 전송할 이미지 directory    
        self.shadowImageDir = 'None'
        self.maskImageDir= 'None'
        self.GTImageDir = 'None'
        self.resultImageDir = 'None'
        
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
        self.checkbox()
        self.selectButton()
        
        self.maskImageCheckbox.toggle()
        
        self.shadowFig = plt.Figure()
        self.maskFig = plt.Figure()
        self.GTFig = plt.Figure()
        self.resultFig = plt.Figure()
        
        self.shadowCanvas = FigureCanvas(self.shadowFig)
        self.maskCanvas = FigureCanvas(self.maskFig)
        self.GTCanvas = FigureCanvas(self.GTFig)
        self.resultCanvas = FigureCanvas(self.resultFig)
        
        self.shadowFig.tight_layout()
        self.maskFig.tight_layout()
        self.GTFig.tight_layout()
        self.resultFig.tight_layout()
        
        self.shadowFig.subplots_adjust(wspace=0, hspace = 0)
        self.maskFig.subplots_adjust(wspace=0, hspace = 0)
        self.GTFig.subplots_adjust(wspace=0, hspace = 0)
        self.resultFig.subplots_adjust(wspace=0, hspace = 0)
        
        self.shadowLayout.addWidget(self.shadowCanvas)
        self.maskLayout.addWidget(self.maskCanvas)
        self.GTLayout.addWidget(self.GTCanvas)
        self.resultLayout.addWidget(self.resultCanvas)
        
        self.shadowToolbar = NavigationToolbar(self.shadowCanvas, self)
        self.maskToolbar = NavigationToolbar(self.maskCanvas, self)
        self.GTToolbar = NavigationToolbar(self.GTCanvas, self)
        self.resultToolbar = NavigationToolbar(self.resultCanvas, self)
        
        self.shadowLayout.addWidget(self.shadowToolbar)
        self.maskLayout.addWidget(self.maskToolbar)
        self.GTLayout.addWidget(self.GTToolbar)
        self.resultLayout.addWidget(self.resultToolbar)
        
        self.shadowCanvas.flush_events()
        self.maskCanvas.flush_events()
        self.GTCanvas.flush_events()
        self.resultCanvas.flush_events()
    
    def menuBar(self):
        self.actionInteractive.triggered.connect(self.change_ui)
        
    def selectDirectory(self):
        self.shadowImageButton.clicked.connect(self.selectShadow)
        self.maskImageButton.clicked.connect(self.selectMask)
        self.GTImageButton.clicked.connect(self.selectGT)
        
    def checkbox(self):
        self.maskImageCheckbox.clicked.connect(self.maskImageToggle)
        self.detectionModelCheckbox.clicked.connect(self.detectionModelToggle)        
         
    def loadModels(self):
        res = requests.get(self.ip + 'modelList')
        models = json.loads(res.json())
        
        for model in models['detection_model']:
            self.selectDetectionModel.addItem(model)
            
        for model in models['removal_model']:
            self.selectRemovalModel.addItem(model)
            
    def selectButton(self):
        self.run.clicked.connect(self.shadowRemovalUpload)
        
    
    # ===================================================================

    def selectShadow(self):
        self.shadowImageDir = QFileDialog.getExistingDirectory(self, 'Select shadowImage Directory', './')
        
        if self.shadowImageDir != 'None' and self.shadowImageDir != '':
            self.shadowImageName.setText(self.shadowImageDir)
            self.shadowImageList = os.listdir(self.shadowImageDir)
            self.shadowImageSelect = QListWidget()
            for idx, l in enumerate(self.shadowImageList):
                if '.png' in l or '.jpg' in l or '.JPG' in l:
                    self.shadowImageSelect.insertItem(idx, l)
                    
            self.shadowScrollArea.setWidget(self.shadowImageSelect)
            self.shadowImageSelect.itemClicked.connect(self.showShadow)
        
    def selectMask(self):
        self.maskImageDir = QFileDialog.getExistingDirectory(self, 'Select maskImage Directory', './')
        
        if self.maskImageDir != 'None' and self.maskImageDir != '':
            self.maskImageName.setText(self.maskImageDir)
            self.maskImageList = os.listdir(self.maskImageDir)
            self.maskImageSelect = QListWidget()
            for idx, l in enumerate(self.maskImageList):
                if '.png' in l or '.jpg' in l or '.JPG' in l:
                    self.maskImageSelect.insertItem(idx, l)
                    
            self.maskScrollArea.setWidget(self.maskImageSelect)
            self.maskImageSelect.itemClicked.connect(self.showMask)
            
    def selectGT(self):
        self.GTImageDir = QFileDialog.getExistingDirectory(self, 'Select GTImage Directory', './')
        
        if self.GTImageDir != 'None' and self.GTImageDir != '' and self.GTImageCheckbox.isChecked():
            self.GTImageName.setText(self.GTImageDir)
            self.GTImageList = os.listdir(self.GTImageDir)
            self.GTImageSelect = QListWidget()
            for idx, l in enumerate(self.GTImageList):
                if '.png' in l or '.jpg' in l or '.JPG' in l:
                    self.GTImageSelect.insertItem(idx, l)
                    
            self.GTScrollArea.setWidget(self.GTImageSelect)
            self.GTImageSelect.itemClicked.connect(self.showGT)
    
    def showShadow(self):
        if self.shadowImageSelect.currentItem().text() is not None:
            name = self.shadowImageSelect.currentItem().text()
            imgDir = os.path.join(self.shadowImageDir, name)
            self.shadowImage = cv2.imread(imgDir)
            
            self.setImageFunc()
            
    def showMask(self):
        if self.maskImageSelect.currentItem().text() is not None:
            name = self.maskImageSelect.currentItem().text()
            imgDir = os.path.join(self.maskImageDir, name)
            self.maskImage = cv2.imread(imgDir)
            
            self.setImageFunc()
            
    def showGT(self):
        if self.GTImageSelect.currentItem().text() is not None:
            name = self.GTImageSelect.currentItem().text()
            imgDir = os.path.join(self.GTImageDir, name)
            self.GTImage = cv2.imread(imgDir)
            
            self.setImageFunc()
    
    def maskImageToggle(self):
        # mask를 이미 가지고 있는 이미지를 사용할지, Detection model을 사용할 지 선택
        if self.maskImageCheckbox.isChecked():
            if self.detectionModelCheckbox.isChecked():
                self.detectionModelCheckbox.toggle()
        else:
            if not self.detectionModelCheckbox.isChecked():
                self.detectionModelCheckbox.toggle()
            
    def detectionModelToggle(self):
        # mask를 이미 가지고 있는 이미지를 사용할지, Detection model을 사용할 지 선택
        if self.detectionModelCheckbox.isChecked():
            if self.maskImageCheckbox.isChecked():
                self.maskImageCheckbox.toggle()
        else:
            if not self.maskImageCheckbox.isChecked():
                self.maskImageCheckbox.toggle()
                
        if self.maskImageDir != 'None' and self.maskImageDir != '':
            self.maskImageSelect.clear()
            ax = self.maskFig.add_subplot(1, 1, 1)
            ax.axis('off')
            ax.cla()
            self.maskCanvas.draw()
           
    # matplotlib를 통한 이미지 plot 알고리즘    
    def setImageFunc(self):
        # 크기 설정
        print('set Image')
        
        if self.shadowImage is not None:
            self.shadowFig.clear()
            ax = self.shadowFig.add_subplot(1, 1, 1)
            ax.axis('off')
            ax.title.set_text('shadow')
            shadow = cv2.cvtColor(self.shadowImage, cv2.COLOR_BGR2RGB)
            ax.imshow(shadow)
            disconnect_zoom = zoom_factory(ax)
            self.shadowFig.tight_layout()
            self.shadowCanvas.draw()


        if self.maskImage is not None:
            self.maskFig.clear()
            ax = self.maskFig.add_subplot(1, 1, 1)
            ax.axis('off')
            ax.title.set_text('mask')
            mask = cv2.cvtColor(self.maskImage, cv2.COLOR_BGR2RGB)
            ax.imshow(mask)
            disconnect_zoom = zoom_factory(ax)
            self.maskFig.tight_layout()
            self.maskCanvas.draw()
            
        if self.GTImage is not None:
            self.GTFig.clear()
            ax = self.GTFig.add_subplot(1, 1, 1)
            ax.axis('off')
            ax.title.set_text('mask')
            gt = cv2.cvtColor(self.GTImage, cv2.COLOR_BGR2RGB)
            ax.imshow(gt)
            disconnect_zoom = zoom_factory(ax)
            self.GTFig.tight_layout()
            self.GTCanvas.draw()

    # def runFunc(self):
        # if self.shadowImageDir != 'None' and self.maskImageDir != 'None':
            
    def shadowRemovalUpload(self):
        removalModel = self.selectRemovalModel.currentText()
        shadowList = os.listdir(self.shadowImageDir)
        dir = os.getcwd()
        os.chdir(self.shadowImageDir)
        
        shadowZip = zipfile.ZipFile(os.path.join(dir, 'data/shadow.zip'), 'w')
        
        for shadow in shadowList:
            shadowZip.write(shadow)
        shadowZip.close()
        
        if self.detectionModelCheckbox.isChecked():
            detectionModel = self.selectDetectionModel.currentText()
            maskZip = 'None'
            
        else:
            detectionModel = 'None'
            maskList = os.listdir(self.maskImageDir)
            os.chdir(self.maskImageDir)
        
            maskZip = zipfile.ZipFile(os.path.join(dir, 'data/mask.zip'), 'w')
        
            for mask in maskList:
                maskZip.write(mask)
            maskZip.close()
            
            os.chdir(dir)
            with open("data/mask.zip", "rb") as f:
                bytes = f.read()
                maskZip = base64.b64encode(bytes)
            
        
        os.chdir(dir)
        
        with open("data/shadow.zip", "rb") as f:
            bytes = f.read()
            shadowZip = base64.b64encode(bytes)
            
        upload = {'mode': 'Batch', 'removal': True, 'detection': self.detectionModelCheckbox.isChecked(), 'perf': False,
                      'removalModel': removalModel, 'detectionModel': detectionModel, 'shadowImage': shadowZip, 'maskImage': maskZip, 'GTImage': 'None',
                      'dilateValue': 'None'}
            
        res = requests.post(self.ip + 'predict', data=upload)
        res = json.loads(res.json())
        img_predicted = res['predictImage']
        img_inpainted = res['inpaintingImage']
        img_mask = res['maskImage']
        perf = res['perf']
        
        predictZip = base64.b64decode(img_predicted)
        with open(os.path.join('result','predicted.zip'), 'wb') as f:
                f.write(predictZip)   
        predictZip = zipfile.ZipFile(os.path.join('result', 'predicted.zip'))
        os.makedirs(os.path.join('result', 'predicted'), exist_ok=True)
        for file in predictZip.namelist():
                predictZip.extract(file, os.path.join('result', 'predicted'))
                
        inpaintZip = base64.b64decode(img_inpainted)
        with open(os.path.join('result','inpainted.zip'), 'wb') as f:
                f.write(inpaintZip)   
        inpaintZip = zipfile.ZipFile(os.path.join('result', 'inpainted.zip'))
        os.makedirs(os.path.join('result', 'inpainted'), exist_ok=True)
        for file in inpaintZip.namelist():
                inpaintZip.extract(file, os.path.join('result', 'inpainted'))
                
        maskZip = base64.b64decode(img_mask)
        with open(os.path.join('result','masked.zip'), 'wb') as f:
                f.write(maskZip)   
        maskZip = zipfile.ZipFile(os.path.join('result', 'masked.zip'))
        os.makedirs(os.path.join('result', 'masked'), exist_ok=True)
        for file in maskZip.namelist():
                maskZip.extract(file, os.path.join('result', 'masked'))
        
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