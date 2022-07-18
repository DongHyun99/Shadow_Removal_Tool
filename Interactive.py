import os
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import requests
import json
import base64
import cv2
import numpy as np
from utils import Masking, Visualization
from Popup import ShadowRemovalPopup, InpaintingPopup

class Interact(QMainWindow) :
    def __init__(self) :
        super().__init__()
        self.ui = uic.loadUi('Interactive.ui', self)
        self.setWindowTitle('ShadowRemoval (Interactive mode) Demo')
        
        self.maskRetriev = False # MaskRetrieval 이후 run이 활성화 됨
        self.runed = False # run 이후 overlay enabled가 활성화 됨
        
        # api로 전송할 이미지 directory    
        self.shadowImageDir = 'None'
        self.maskImageDir= 'None'
        self.shadowImage = None
        self.maskImage = None
        # self.ip = 'http://203.249.22.163:50000/'
        self.ip = 'http://129.254.81.18:50000/'
        
        # Visualization
        self.popupCheckbox = False
        self.shadowRemovalCheckbox = True
        self.removalAndInpaintingCheckbox = False
        self.edgeDetectionCheckbox = False
        
        self.loadModels()
        self.menuBar()
        self.selectDirectory()
        self.checkbox()
        self.slider()
        self.hide()
        self.selectButton()
        
        self.maskImageCheckbox.toggle()
        
        self.maskSetting = Masking()
        self.visualization = Visualization()
        self.ShadowRemovalPopup = ShadowRemovalPopup()
        self.InpaintingPopup = InpaintingPopup()
        
        
        
    def menuBar(self):
        self.actionBatch.triggered.connect(self.change_ui)
        self.actionMasking.triggered.connect(self.maskSetting_ui)
        self.actionVisualization.triggered.connect(self.visualization_ui)
        
    def selectDirectory(self):
        self.shadowImageButton.clicked.connect(self.selectShadow)
        self.maskImageButton.clicked.connect(self.selectMask)
        
    def checkbox(self):
        self.maskOverlayCheckbox.clicked.connect(self.overlayCheck)
        self.maskOverlayCheckbox.stateChanged.connect(self.overlayToggle)
        self.maskImageCheckbox.clicked.connect(self.maskImageToggle)
        self.detectionModelCheckbox.clicked.connect(self.detectionModelToggle)
        #self.actionMasking.triggered.connect(self.actionMaskingToggle)
    
    def slider(self):
        self.maskOpacitySlider.setValue(10)
        self.maskOpacitySlider.valueChanged.connect(self.overlayOpacity)
    
    def hide(self):
        self.maskImageArea.hide()
        self.inpaintingArea.hide()
        self.maskOverlayArea.hide()
        self.maskOpacitySlider.hide()
        
    def selectButton(self):
        self.run.clicked.connect(self.runFunc)
        self.maskRetrieval.clicked.connect(self.MaskRetrievalFunc)
        
    def loadModels(self):
        res = requests.get(self.ip + 'modelList')
        models = json.loads(res.json())
        
        for model in models['detection_model']:
            self.selectDetectionModel.addItem(model)
            
        for model in models['removal_model']:
            self.selectRemovalModel.addItem(model)
        
        
    # ===================================================================
    
    def selectImage(self, area, fileOpen):
        
        # 불러오기 취소 시 종료되는 현상 방지
        if fileOpen[0] == '':
            return 'None'
        
        else:
            image = QPixmap(fileOpen[0])
        
            # 비율이 달라 crop되는 현상 방지 
            if image.height()/image.width() <= 0.75:
                ImageResize = image.scaledToWidth(area.width()-10)
                area.setPixmap(ImageResize)
                area.setAlignment(Qt.AlignCenter)

            else:
                ImageResize = image.scaledToHeight(area.height()-10)
                area.setPixmap(ImageResize)
                area.setAlignment(Qt.AlignCenter)

            return fileOpen[0]

    def selectShadow(self):
        FileOpen = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.shadowImageDir = self.selectImage(self.shadowRemovalArea, FileOpen)
        #print('shadowImageName: {}'.format(self.shadowImageName))
        self.shadowImageName.setText(self.shadowImageDir)
        
        if self.shadowImageDir != 'None':
            self.shadowImage = cv2.imread(self.shadowImageDir)
        
    def selectMask(self):
        FileOpen = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.maskImageDir = self.selectImage(self.maskImageArea, FileOpen)
        #print('maskImageName: {}'.format(self.maskImageName))
        self.maskImageName.setText(self.maskImageDir)
        
        # maskOverlay에 마스크를 입력
        self.selectImage(self.maskOverlayArea, FileOpen)
        
        if self.maskImageDir != 'None':
            self.maskImage = cv2.imread(self.maskImageDir)
        
    def overlayCheck(self):
        if not self.maskRetriev and self.maskOverlayCheckbox.isChecked():
            self.maskOverlayCheckbox.toggle()
        
    def overlayToggle(self, state):
        if state == Qt.Checked:
            self.maskOverlayArea.show()
            self.maskOpacitySlider.show()
        else:
            self.maskOverlayArea.hide()
            self.maskOpacitySlider.hide()
            
    def maskImageToggle(self):
        if self.maskImageCheckbox.isChecked():
            if self.detectionModelCheckbox.isChecked():
                self.detectionModelCheckbox.toggle()
        else:
            if not self.detectionModelCheckbox.isChecked():
                #if not self.actionMasking.isChecked():
                    self.detectionModelCheckbox.toggle()
            
    def detectionModelToggle(self):
        if self.detectionModelCheckbox.isChecked():
            if self.maskImageCheckbox.isChecked():
                self.maskImageCheckbox.toggle()
        else:
            if not self.maskImageCheckbox.isChecked():
                #if not self.actionMasking.isChecked():
                    self.maskImageCheckbox.toggle()
            
    def actionMaskingToggle(self):
        #if not self.actionMasking.isChecked():
            if not self.maskImageCheckbox.isChecked() and not self.detectionModelCheckbox.isChecked():
                self.maskImageCheckbox.toggle()
    
    def overlayOpacity(self):
        opacity_effect = QGraphicsOpacityEffect(self.maskOverlayArea)
        opacity_effect.setOpacity(self.maskOpacitySlider.value()/10)
        self.maskOverlayArea.setGraphicsEffect(opacity_effect)
        
    def setPixmapFunc(self, result, area):
        # 비율이 달라 crop되는 현상 방지 
        if result.height()/result.width() <= 0.75:
            ImageResize = result.scaledToWidth(area.width()-10)
            area.setPixmap(ImageResize)
        else:
            ImageResize = result.scaledToHeight(area.height()-10)
            area.setPixmap(ImageResize)
        area.setAlignment(Qt.AlignCenter)
        
    def MaskRetrievalFunc(self):
        if not self.maskOverlayCheckbox.isChecked():
            self.maskOverlayCheckbox.toggle()
            self.maskOverlayArea.show()
            self.maskOpacitySlider.show()
       
        # 처음부터 마스크를 그릴경우
        if not self.maskImageCheckbox.isChecked() and not self.detectionModelCheckbox.isChecked(): # self.actionMasking.isChecked() and
            pass
        
        # detectionModel을 통한 mask 생성 시
        elif self.detectionModelCheckbox.isChecked():
            shadowImage = base64.b64encode(cv2.imencode('.png', self.shadowImage)[1]).decode()
            
            upload = {'mode': 'Interactive', 'removal': False, 'detection': self.detectionModelCheckbox.isChecked(), 'perf': self.GTImageCheckbox.isChecked(),
                      'removalModel': self.selectRemovalModel.currentText(), 'detectionModel': self.selectDetectionModel.currentText(), 
                      'shadowImage': shadowImage, 'maskImage': 'None', 'GTImage': 'None'}
            
            res = requests.post(self.ip + 'predict', data=upload)
            result = json.loads(res.json())['maskImage']
            result = cv2.imdecode(np.frombuffer(base64.b64decode(result), dtype=np.uint8), flags=1)
            self.maskImage = result
            
            h, w, c = result.shape
            result = QPixmap.fromImage(QImage(result.data, w, h, w*c, QImage.Format_RGB888))
            
            # 비율이 달라 crop되는 현상 방지 
            if result.height()/result.width() <= 0.75:
                ImageResize = result.scaledToWidth(self.maskOverlayArea.width()-10)
                self.maskOverlayArea.setPixmap(ImageResize)
                self.maskImageArea.setPixmap(ImageResize)

            else:
                ImageResize = result.scaledToHeight(self.maskOverlayArea.height()-10)
                self.maskOverlayArea.setPixmap(ImageResize)
                self.maskImageArea.setPixmap(ImageResize)
                
            self.maskOverlayArea.setAlignment(Qt.AlignCenter)
            self.maskImageArea.setAlignment(Qt.AlignCenter)

            self.maskImageDir = 'detection'
        
        # 마스크가 있는 경우   
        else:
            pass
        
        self.maskRetriev = True
        
    def runFunc(self):
        if self.shadowImageDir != 'None' and self.maskImageDir != 'None' and self.maskRetriev == True:
            if self.maskOverlayCheckbox.isChecked():
                self.maskOverlayCheckbox.toggle()
                self.maskOverlayArea.hide()
                self.maskOpacitySlider.hide()
            
            self.runed=True
            
            self.predictImage, self.inpaintImage, self.maskImage = self.shadowRemovalUpload(self.shadowImage, self.maskImage)
            self.shadowImage = self.inpaintImage
            img_predicted = cv2.cvtColor(self.predictImage, cv2.COLOR_BGR2RGB)
            img_inpainted = cv2.cvtColor(self.inpaintImage, cv2.COLOR_BGR2RGB)
            h, w, c = img_inpainted.shape
            img_predicted = QPixmap.fromImage(QImage(img_predicted.data, w, h, w*c, QImage.Format_RGB888))
            img_inpainted = QPixmap.fromImage(QImage(img_inpainted.data, w, h, w*c, QImage.Format_RGB888))
            
            if self.popupCheckbox:
                # Popup Enabled
                Area1 = self.ShadowRemovalPopup.shadowRemovalArea
                Area2 = self.InpaintingPopup.inpaintingArea
            else:
                # Popup Disabled
                Area1 = self.shadowRemovalArea
                Area2 = self.inpaintingArea
            
            if self.shadowRemovalCheckbox:
                # Shadow Removal Only
                self.setPixmapFunc(img_inpainted, Area1)

            elif self.removalAndInpaintingCheckbox:
                # Shadow Removal & Image Inpainting
                self.setPixmapFunc(img_predicted, Area1)
                self.setPixmapFunc(img_inpainted, Area2)
                Area2.show()
            elif self.edgeDetectionCheckbox:
                # Area In-Place & Edge Detection
                self.maskImage = cv2.cvtColor(self.maskImage, cv2.COLOR_BGR2GRAY)
                maskEdge = self.extractMaskEdge(self.maskImage)
                self.predictImage, self.inpaintImage, self.maskImage = self.shadowRemovalUpload(self.shadowImage, maskEdge)
                self.shadowImage = self.inpaintImage
                img_predicted = cv2.cvtColor(self.predictImage, cv2.COLOR_BGR2RGB)
                img_predicted = QPixmap.fromImage(QImage(img_predicted.data, w, h, w*c, QImage.Format_RGB888))
                
                self.setPixmapFunc(img_predicted, Area1)
                self.setPixmapFunc(img_inpainted, Area2)
                Area2.show()
            
            
            self.maskImageArea.show()
            
            if self.popupCheckbox:
                self.ShadowRemovalPopup.show()
                if not self.shadowRemovalCheckbox:
                    self.InpaintingPopup.show()
   
    def change_ui(self):
        from Batch import Batch
        self.batch = Batch()
        self.batch.show()
        self.close()

    def maskSetting_ui(self):
        self.maskSetting.exec()
        
    def visualization_ui(self):
        if self.popupCheckbox:
            self.visualization.popupCheckbox.toggle()
        if self.shadowRemovalCheckbox:
            self.visualization.shadowRemovalCheckbox.toggle()
        if self.removalAndInpaintingCheckbox:
            self.visualization.removalAndInpaintingCheckbox.toggle()
        if self.edgeDetectionCheckbox:
            self.visualization.edgeDetectionCheckbox.toggle()
        
        self.visualization.exec()
        self.popupCheckbox = self.visualization.popupCheckbox.isChecked()
        self.shadowRemovalCheckbox = self.visualization.shadowRemovalCheckbox.isChecked()
        self.removalAndInpaintingCheckbox = self.visualization.removalAndInpaintingCheckbox.isChecked()
        self.edgeDetectionCheckbox = self.visualization.edgeDetectionCheckbox.isChecked()
        
    def extractMaskEdge(self, mask):    # mask [:,:, 1]
        #mask_double =np.asarry(mask, dtype=np.double)
        gx, gy = np.gradient(mask)
        
        temp_edge = gy * gy + gx * gx
        
        dilateValue = self.maskSetting.dilateValue.value() * 3          
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(temp_edge, kernel, dilateValue)
        print(f"Edge Dilate: {dilateValue}")
 
        temp_edge[temp_edge != 0.0] = 255.0
        temp_edge = np.asarray(temp_edge, dtype=np.uint8)
        
        return temp_edge
        
    def shadowRemovalUpload(self, shadow, mask):
            print('removalModel: {}\nshadowImage:{}\nmaskImage:{}'.format(self.selectRemovalModel.currentText(), self.shadowImageDir, self.maskImageDir))
            shadowImage = base64.b64encode(cv2.imencode('.png', shadow)[1]).decode()
            maskImage = base64.b64encode(cv2.imencode('.png', mask)[1]).decode()
            upload = {'mode': 'Interactive', 'removal': True, 'detection': False, 'perf': self.GTImageCheckbox.isChecked(),
                      'removalModel': self.selectRemovalModel.currentText(), 'detectionModel': self.selectDetectionModel.currentText(), 
                      'shadowImage': shadowImage, 'maskImage': maskImage, 'GTImage': 'None'}
            
            res = requests.post(self.ip + 'predict', data=upload)
            img_predicted = json.loads(res.json())['predictImage']
            img_inpainted = json.loads(res.json())['inpaintingImage']
            img_mask = json.loads(res.json())['maskImage']
            
            img_predicted = cv2.imdecode(np.frombuffer(base64.b64decode(img_predicted), dtype=np.uint8), flags=1)
            img_inpainted = cv2.imdecode(np.frombuffer(base64.b64decode(img_inpainted), dtype=np.uint8), flags=1)
            img_mask = cv2.imdecode(np.frombuffer(base64.b64decode(img_mask), dtype=np.uint8), flags=1)
            
            return img_predicted, img_inpainted, img_mask
                
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = Interact()
    myWindow.show()
    sys.exit(app.exec())