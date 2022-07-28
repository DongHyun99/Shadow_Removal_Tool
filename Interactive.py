import os
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from cv2 import inpaint
import requests
import json
import base64
import cv2
import numpy as np
from utils import Masking, Visualization, Server
from Popup import ShadowRemovalPopup, PredictPopup


import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_interactions import zoom_factory

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.filters import sobel
from skimage.color import rgb2gray

class Interact(QMainWindow) :
    def __init__(self) :
        super().__init__()
        self.ui = uic.loadUi('UI/Interactive.ui', self)
        self.setWindowTitle('ShadowRemoval (Interactive mode) Demo')
        
        self.maskRetriev = False # MaskRetrieval 이후 run이 활성화 됨
        self.runed = False # run 이후 overlay enabled가 활성화 됨
        
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
        self.num = 1
        self.originalImage = {}
        # self.ip = 'http://203.249.22.163:50000/'
        self.ip = 'http://129.254.81.18:50000/'
        self.maskColor = (255, 255, 180)
        self.segmentColor = (255, 180, 180)
        self.brushColor = '#2ca02c'
        self.ax = None
        
        # Visualization
        self.popupCheckbox = False
        self.shadowRemovalCheckbox = False
        self.predictAndInpaintingCheckbox = False
        self.edgeDetectionCheckbox = True
        self.segmentation = False
        self.segmentationLogic = 'slic'
        self.maskingMode = False
        
        self.loadModels()
        self.menuBar()
        self.selectDirectory()
        self.checkbox()
        self.slider()
        self.hide()
        self.selectButton()
        self.radioButton()
        
        self.maskImageCheckbox.toggle()
        
        self.maskSetting = Masking()
        self.visualization = Visualization()
        self.ShadowRemovalPopup = ShadowRemovalPopup()
        self.predictPopup = PredictPopup()
        self.server = Server()
        
        # matplotlib Setting
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.fig.tight_layout()
        self.fig.subplots_adjust(wspace=0.05, hspace = 0)
        self.layout.addWidget(self.canvas)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.canvas.flush_events()
        
        self.mouseCapture = False
        self.ctrl_key = False
        self.shift_key = False
        self.alt_key = False
        self.edges = None
        
        #drawing Image
        self.xdata = []
        self.ydata = []
        self.val = 10
        self.line = None
        self.pen = 'brush'
        
    def menuBar(self):
        self.actionBatch.triggered.connect(self.change_ui)
        self.actionMasking.triggered.connect(self.maskSetting_ui)
        self.actionVisualization.triggered.connect(self.visualization_ui)
        self.actionServers.triggered.connect(self.server_ui)
        
    def selectDirectory(self):
        self.shadowImageButton.clicked.connect(self.selectShadow)
        self.maskImageButton.clicked.connect(self.selectMask)
        self.GTImageButton.clicked.connect(self.selectGT)
        
    def checkbox(self):
        self.maskOverlayCheckbox.stateChanged.connect(self.overlayToggle)
        self.maskImageCheckbox.clicked.connect(self.maskImageToggle)
        self.detectionModelCheckbox.clicked.connect(self.detectionModelToggle)
    
    def slider(self):
        self.maskOpacitySlider.setValue(17)
        self.maskOpacitySlider.valueChanged.connect(self.overlayOpacity)
        self.brushSizeSlider.valueChanged.connect(self.brushSizeChanged)
    
    def hide(self):
        # Ovelay hide
        self.maskOpacitySlider.hide()
        self.maskOverlayCheckbox.setEnabled(False)
        
        # Dilate hide
        self.dilateErodeIterationsSlider.hide()
        self.dilateIterationLabel.hide()
        self.Dilate.hide()
        self.Erode.hide()
        self.defaultDilateCheckbox.hide()
        
        # Masking Mode hide
        self.brushColorBtn.hide()
        self.brushSizeSlider.hide()
        self.maskingLabel.hide()
        self.brush.hide()
        self.eraser.hide()
              
    def selectButton(self):
        self.run.clicked.connect(self.runFunc)
        self.maskRetrieval.clicked.connect(self.MaskRetrievalFunc)
        self.Dilate.clicked.connect(self.dilateFunc)
        self.Erode.clicked.connect(self.erodeFunc)
        self.brushColorBtn.clicked.connect(self.brushColorFunc)
        self.Save.clicked.connect(self.saveFunc)
        self.Clear.clicked.connect(self.clearFunc)
 
    def radioButton(self):
        self.brush.clicked.connect(self.brushOrEraser)
        self.eraser.clicked.connect(self.brushOrEraser)
        
    def loadModels(self):
        # Recall model list
        res = requests.get(self.ip + 'modelList')
        models = json.loads(res.json())
        
        for model in models['detection_model']:
            self.selectDetectionModel.addItem(model)
            
        for model in models['removal_model']:
            self.selectRemovalModel.addItem(model)
            
    def eventConnect(self):
        self.fig.canvas.mpl_connect('motion_notify_event', self.onMouseMove)
        self.fig.canvas.mpl_connect('button_press_event', self.onMousePress)
        self.fig.canvas.mpl_connect('button_release_event', self.onMouseRelease)
        self.fig.canvas.mpl_connect('scroll_event', self.zoomEvent)
        #self.fig.canvas.mpl_connect('figure_enter_event', self.onEnterFigure)
        #self.fig.canvas.mpl_connect('figure_leave_event', self.onLeaveFigure)
        #self.fig.canvas.mpl_connect('axes_enter_event', self.onEnterAxes)
        #self.fig.canvas.mpl_connect('axes_leave_event', self.onLeaveAxes)
        
        
    # ===================================================================

    def selectShadow(self):
        # shadow 이미지 선택시
        FileOpen = QFileDialog.getOpenFileName(self, 'Open shadowImage file', './')
        self.shadowImageDir = FileOpen[0]
        
        if self.shadowImageDir != 'None' and self.shadowImageDir != '':
            self.shadowImage = cv2.imread(self.shadowImageDir)

            self.ax = self.setImageFunc(fig = self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=1, inpainted=self.shadowImage)
            self.shadowImageName.setText(self.shadowImageDir)
        
    def selectMask(self):
        # mask 이미지 선택시
        FileOpen = QFileDialog.getOpenFileName(self, 'Open maskImage file', './')
        self.maskImageDir = FileOpen[0]
        
        if self.maskImageDir != 'None' and self.maskImageDir != '':
            self.maskImage = cv2.imread(self.maskImageDir, cv2.IMREAD_GRAYSCALE)
            self.maskImageName.setText(self.maskImageDir)
    
    def selectGT(self):
        # GT 이미지 선택시
        FileOpen = QFileDialog.getOpenFileName(self, 'Open GTImage file', './')
        self.GTDir = FileOpen[0]
        
        if self.GTDir != 'None' and self.GTDir != '':
            self.GTImage = cv2.imread(self.GTDir)
            self.GTImageName.setText(self.GTDir)
    
    def overlayToggle(self, state):
        # overlay checkbox를 toggle 시 작동, maskRetriev 동작전에는 실행할 수 없음
        mask = None
        predict = None
        inpainted = self.shadowImage
        if self.runed:
            mask = self.maskImage
            predict = self.predictImage
            
        if state == Qt.Checked:
            self.maskOpacitySlider.show()

        else:
            self.maskOpacitySlider.hide()
        
        self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=inpainted, prediction=predict, mask = mask)
            
    def maskImageToggle(self):
        # mask를 이미 가지고 있는 이미지를 사용할지, Detection model을 사용할 지 선택
        # Masking mode 시 둘다 선택하지 않아도 사용 가능
        if self.maskImageCheckbox.isChecked():
            if self.detectionModelCheckbox.isChecked():
                self.detectionModelCheckbox.toggle()
        else:
            if not self.detectionModelCheckbox.isChecked():
                if not self.maskingMode:
                    self.detectionModelCheckbox.toggle()
            
    def detectionModelToggle(self):
        # mask를 이미 가지고 있는 이미지를 사용할지, Detection model을 사용할 지 선택
        # Masking mode 시 둘다 선택하지 않아도 사용 가능
        if self.detectionModelCheckbox.isChecked():
            if self.maskImageCheckbox.isChecked():
                self.maskImageCheckbox.toggle()
        else:
            if not self.maskImageCheckbox.isChecked():
                if not self.maskingMode:
                    self.maskImageCheckbox.toggle()
    
    def overlayOpacity(self):
        # overlay의 투명도를 변경 할 때 동작
        mask = None
        predict = None
        inpainted = self.shadowImage
        if self.runed:
            mask = self.maskImage
            predict = self.predictImage
            
        self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=inpainted, prediction=predict, mask = mask)
 
    def brushSizeChanged(self):
        # brush 크기가 달라지면 동작
        self.line, = self.ax.plot(self.xdata, self.ydata, linewidth=self.brushSizeSlider.value(), color=self.brushColor)  
        
    def brushOrEraser(self):
        # Eraser와 Brush 선택
        if self.brush.isChecked():
            self.pen = 'brush'
        if self.eraser.isChecked():
            self.pen = 'eraser'
        print(self.pen, 'mode')
        
    # matplotlib를 통한 이미지 plot 알고리즘    
    def setImageFunc(self, fig, canvas, checkable, num = 0, alpha= 17, inpainted = None, prediction = None, mask = None):
        # 크기 설정
        val = 1
        
        if self.ax != None:
            size = self.ax.axis()
            
        else: 
            size = (0, inpainted.shape[1], inpainted.shape[0], 0)
        
        fig.clear()
        ax = fig.add_subplot(1, num, val)
        ax.axis('off')
        print('set Image')
        
        #brush & eraser 기능인 line 생성
        if self.maskRetriev:
            self.line, = ax.plot(self.xdata, self.ydata, linewidth=self.brushSizeSlider.value(), color=self.brushColor)
            print('set lines')
        
        # maskRetriev 이후에는 첫번째 sublot에 그림자를 띄움
        if mask is not None:
            # mask
            val += 1
            ax.imshow(mask, cmap='gray')
            disconnect_zoom = zoom_factory(ax)
            ax.title.set_text('Mask')
            ax = fig.add_subplot(1, num, val)
            ax.axis('off')
        
        # Overlay를 하지 않는경우
        if not checkable:
            image = inpainted.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            disconnect_zoom = zoom_factory(ax)
            ax.axis(size)
            
        # Overlay 시
        else:
            
            image = inpainted.copy()
            # Mask의 값이 모두 0일 경우 255로 나눌때 ValueError가 생김
            try:
                maskimg = self.maskImage / 255
            except ValueError:
                maskimg = self.maskImage
                
            maskimg = maskimg.astype(np.uint8)            
            
            # alpha 값을 추가하여 opacity를 조정할 수 있도록 mask 조정
            shadow_area = np.array([maskimg, maskimg, maskimg, maskimg]).transpose(1, 2, 0)
            outputImage = np.where(shadow_area == [1, 1, 1, 1], self.maskColor+(alpha*15,), shadow_area)
            
            if self.edges is not None and self.segmentation:
                image[self.y1 : self.y2, self.x1 : self.x2] = self.edges
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            ax.imshow(image)
            ax.imshow(outputImage)
            disconnect_zoom = zoom_factory(ax)
            ax.axis(size)
                
        ax.title.set_text('Inpainting')
        
        # mask, predicted, inpainted 3가지를 plot 하는 경우
        if  num > 1 and prediction is not None:
            # prediction
            val += 1
            ax = fig.add_subplot(1, num, val)
            ax.axis('off')
            
            prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
            
            ax.imshow(prediction)
            disconnect_zoom = zoom_factory(ax)
            ax.title.set_text('Predict')
            
        fig.tight_layout()
        canvas.draw()
        
        return ax
    
    def dilateFunc(self):
        kernel = np.ones((3,3), np.uint8)
        self.maskImage = cv2.dilate(self.maskImage.astype('uint8'), kernel, self.dilateErodeIterationsSlider.value())
        print(f"Dilate ...")
        
        mask = None
        predict = None
        inpainted = self.shadowImage
        if self.runed:
            mask = self.maskImage
            predict = self.predictImage
            
        self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=inpainted, prediction=predict, mask = mask)
        
    def erodeFunc(self):
        kernel = np.ones((3,3), np.uint8)
        self.maskImage = cv2.erode(self.maskImage, kernel, self.dilateErodeIterationsSlider.value())
        print(f"Erode ...")
        mask = None
        predict = None
        inpainted = self.shadowImage
        if self.runed:
            mask = self.maskImage
            predict = self.predictImage
            
        self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=inpainted, prediction=predict, mask = mask)
            
    def MaskRetrievalFunc(self):
        # 처음부터 마스크를 그릴경우
        if not self.maskImageCheckbox.isChecked() and not self.detectionModelCheckbox.isChecked() and self.maskingMode: # self.actionMasking.isChecked() and
            self.maskImage = np.zeros((self.shadowImage.shape[0], self.shadowImage.shape[1]))
            self.maskImage[0][0] = 255
            self.maskImageDir = 'Masking'
        
        # detectionModel을 통한 mask 생성 시
        elif self.detectionModelCheckbox.isChecked():
            shadowImage = base64.b64encode(cv2.imencode('.png', self.shadowImage)[1]).decode()
            
            upload = {'mode': 'Interactive', 'removal': False, 'detection': self.detectionModelCheckbox.isChecked(), 'perf': self.GTImageCheckbox.isChecked(),
                      'removalModel': self.selectRemovalModel.currentText(), 'detectionModel': self.selectDetectionModel.currentText(), 
                      'shadowImage': shadowImage, 'maskImage': 'None', 'GTImage': 'None', 'dilateValue': 'None'}
            
            res = requests.post(self.ip + 'predict', data=upload)
            result = json.loads(res.json())['maskImage']
            result = cv2.imdecode(np.frombuffer(base64.b64decode(result), dtype=np.uint8), flags=1)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            self.maskImage = result

            self.maskImageDir = 'detection'
        
        # 마스크가 있는 경우   
        else:
            pass
        
        self.maskRetriev = True
        
        self.maskOverlayCheckbox.setEnabled(True)
        if not self.maskOverlayCheckbox.isChecked():
            self.maskOverlayCheckbox.toggle()
            self.maskOpacitySlider.show()
        
        self.fig2, self.ax2 = plt.subplots(figsize=(self.maskImage.shape[1]/100, self.maskImage.shape[0]/100)) # figsize = value * dpi(default 100)이므로
        self.originalImage = {'image': [self.shadowImage], 'imName': os.path.basename(self.shadowImageDir)}
        
        self.eventConnect()
        
    def runFunc(self):
        if self.shadowImageDir != 'None' and self.maskImageDir != 'None' and self.maskRetriev == True:
            if self.maskOverlayCheckbox.isChecked():
                self.maskOverlayCheckbox.toggle()
            
            self.runed=True
            self.ax = None
            self.predictImage, self.inpaintImage, self.maskImage = self.shadowRemovalUpload(self.shadowImage, self.maskImage, self.maskSetting.dilateMaskCheckbox.isChecked())
            self.maskImage = cv2.cvtColor(self.maskImage, cv2.COLOR_RGB2GRAY)
            
            self.originalImage['image'].append(self.maskImage.copy())
            self.originalImage['image'].append(self.inpaintImage.copy())
            self.originalImage['image'].append(self.predictImage.copy())
            
            if self.popupCheckbox:
                # Popup Enabled
                self.shadowImage = self.inpaintImage
                
                shadowRemovalFig = self.ShadowRemovalPopup.fig
                shadowRemovalCanvas = self.ShadowRemovalPopup.canvas
                predictFig = self.predictPopup.fig
                predictCanvas = self.predictPopup.canvas
                
                if self.shadowRemovalCheckbox:
                    # Shadow Removal Only
                    self.num = 2
                    self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=self.shadowImage, prediction=None, mask = self.maskImage)
                    self.ax = self.setImageFunc(fig = shadowRemovalFig, canvas=shadowRemovalCanvas, checkable=False, num=1, inpainted=self.inpaintImage, prediction=None, mask = None)
                    self.predictImage =None
                    
                elif self.predictAndInpaintingCheckbox:
                    # Predict & Image Inpainting
                    self.num = 2
                    self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=self.shadowImage, prediction=None, mask = self.maskImage)
                    self.ax = self.setImageFunc(fig = shadowRemovalFig, canvas=shadowRemovalCanvas, checkable=False, num=1, inpainted=self.inpaintImage, prediction=None, mask = None)
                    self.ax = self.setImageFunc(fig = predictFig, canvas = predictCanvas, checkable=False, num=1, inpainted = self.predictImage, prediction=None, mask=None)
                    self.predictImage =None
            
                elif self.edgeDetectionCheckbox:
                    # Area In-Place & Edge Detection
                    self.num=2
                    maskEdge = self.extractMaskEdge(self.maskImage)
                    
                    self.predictImage, self.inpaintImage, mask = self.shadowRemovalUpload(self.shadowImage, maskEdge, removalModel='Inpainting')
                    self.originalImage['image'].append(mask.copy())
                    self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=self.shadowImage, prediction=None, mask = self.maskImage)
                    self.ax = self.setImageFunc(fig = shadowRemovalFig, canvas=shadowRemovalCanvas, checkable=False, num=1, inpainted=self.inpaintImage, prediction=None, mask = None)
                    self.ax = self.setImageFunc(fig = predictFig, canvas = predictCanvas, checkable=False, num=1, inpainted = self.predictImage, prediction=None, mask=None)
                    
                    self.originalImage['image'].append(self.inpaintImage.copy())
                    self.originalImage['image'].append(self.predictImage.copy())
                    
                    self.predictImage =None

            else:
                # Popup Disabled
                self.shadowImage = self.inpaintImage
                if self.shadowRemovalCheckbox:
                    # Shadow Removal Only
                    self.num = 2
                    self.predictImage=None
                    self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=self.shadowImage, prediction=self.predictImage, mask = self.maskImage)
                    
                elif self.predictAndInpaintingCheckbox:
                    # Shadow Removal & Image Inpainting
                    self.num = 3
                    self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=self.shadowImage, prediction=self.predictImage, mask = self.maskImage)

                elif self.edgeDetectionCheckbox:
                    # Area In-Place & Edge Detection
                    self.num = 3
                    maskEdge = self.extractMaskEdge(self.maskImage)
                    
                    self.predictImage, self.inpaintImage, mask = self.shadowRemovalUpload(self.shadowImage, maskEdge, removalModel='Inpainting')
                    self.originalImage['image'].append(mask.copy())
                    self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=self.shadowImage, prediction=self.predictImage, mask = self.maskImage)
                    
                    self.originalImage['image'].append(self.inpaintImage.copy())
                    self.originalImage['image'].append(self.predictImage.copy())
            
            if self.popupCheckbox:
                self.ShadowRemovalPopup.show()
                if not self.shadowRemovalCheckbox:
                    self.predictPopup.show()
                    
    def brushColorFunc(self):
        self.brushColor = QColorDialog.getColor().name()
        # self.brushColor = ImageColor.getcolor(self.brushColor, 'RGB')
        self.brushColorBtn.setStyleSheet(
            'background-color:{}; border:none; border-radius: 5px;'.format(self.brushColor)
        )
        
    def saveFunc(self):
        
        DirOpen = QFileDialog.getExistingDirectory(self, 'Select save Directory', './')
        self.saveDir = DirOpen
         
        if self.saveDir != 'None' and self.saveDir != '':
            print('save dir: '+self.saveDir)
            if self.originalImage !={}:
                length = len(self.originalImage['image'])
                if length >= 2:
                    cv2.imwrite(os.path.join(self.saveDir, self.originalImage['imName'][:-4]+'_original.png'), self.originalImage['image'][0])
                    cv2.imwrite(os.path.join(self.saveDir, self.originalImage['imName'][:-4]+'_mask.png'), self.originalImage['image'][1])
                if length >= 4:
                    cv2.imwrite(os.path.join(self.saveDir, self.originalImage['imName'][:-4]+'_inapinted.png'), self.originalImage['image'][2])
                    cv2.imwrite(os.path.join(self.saveDir, self.originalImage['imName'][:-4]+'_predicted.png'), self.originalImage['image'][3])
                if length >= 7:
                    cv2.imwrite(os.path.join(self.saveDir, self.originalImage['imName'][:-4]+'_maskEdge.png'), self.originalImage['image'][4])
                    cv2.imwrite(os.path.join(self.saveDir, self.originalImage['imName'][:-4]+'_edgeInapinted.png'), self.originalImage['image'][5])
                    cv2.imwrite(os.path.join(self.saveDir, self.originalImage['imName'][:-4]+'_edgePredicted.png'), self.originalImage['image'][6])
    
    def clearFunc(self):
        self.Clear = Interact()
        self.Clear.show()
        self.close()
   
    def change_ui(self):
        from Batch import Batch
        self.batch = Batch()
        self.batch.show()
        self.close()

    def maskSetting_ui(self):
        
        self.maskSetting.exec_()
        self.segmentation = self.maskSetting.segmentationCheckbox.isChecked()
        self.maskColor = self.maskSetting.maskCol
        self.segmentColor = self.maskSetting.segmentCol
        self.maskingMode = self.maskSetting.maskingModeCheckbox.isChecked()
        
        if self.maskSetting.slic.isChecked(): self.segmentationLogic = 'slic'
        elif self.maskSetting.felzenswalb.isChecked(): self.segmentationLogic = 'felzenswalb'
        elif self.maskSetting.quickShift.isChecked(): self.segmentationLogic = 'quickShift'
        else: self.segmentationLogic='waterShed'
        
        if self.segmentation:
            self.dilateErodeIterationsSlider.show()
            self.dilateIterationLabel.show()
            self.Dilate.show()
            self.Erode.show()
            self.defaultDilateCheckbox.show()
            
        else:
            self.dilateErodeIterationsSlider.hide()
            self.dilateIterationLabel.hide()
            self.Dilate.hide()
            self.Erode.hide()
            self.defaultDilateCheckbox.hide()
            
        if self.maskingMode:
            self.brushColorBtn.show()
            self.maskingLabel.show()
            self.brushSizeSlider.show()
            self.brush.show()
            self.eraser.show()
            
        else:
            self.brushColorBtn.hide()
            self.maskingLabel.hide()
            self.brushSizeSlider.hide()
            self.brush.hide()
            self.eraser.hide()
            

        mask = None
        predict = None
        inpainted = self.shadowImage

        if self.runed:
            mask = self.maskImage
            predict = self.predictImage
        
        if self.shadowImageDir != 'None':
            self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=inpainted, prediction=predict, mask = mask)
        
    def visualization_ui(self):
        
        self.visualization.exec_()
        self.popupCheckbox = self.visualization.popupCheckbox.isChecked()
        self.shadowRemovalCheckbox = self.visualization.shadowRemovalCheckbox.isChecked()
        self.predictAndInpaintingCheckbox = self.visualization.predictAndInpaintingCheckbox.isChecked()
        self.edgeDetectionCheckbox = self.visualization.edgeDetectionCheckbox.isChecked()
        
    def server_ui(self):
        self.server.serverEdit.setText(self.ip[7:-1])
        self.server.exec_()
        self.ip = 'http://' + self.server.serverEdit.text() + '/'
        
        print(self.ip)
                

    def extractMaskEdge(self, mask):    # mask [:,:, 1]
        #mask_double =np.asarry(mask, dtype=np.double)
        gx, gy = np.gradient(mask)
        
        temp_edge = gy * gy + gx * gx
        
        dilateValue = self.maskSetting.dilateValueSlider.value()         
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(temp_edge, kernel, dilateValue)
        print(f"Edge Dilate: {dilateValue}")
 
        mask[mask != 0.0] = 255.0
        mask = np.asarray(mask, dtype=np.uint8)
        
        return mask
    
    def fig2Array(self, fig):
        fig.canvas.draw()
        return np.array(fig.canvas.renderer._renderer)

        
    def shadowRemovalUpload(self, shadow, mask, dilate = False, removalModel = None):
        
        if removalModel == None:
            removalModel = self.selectRemovalModel.currentText()
        
        print('removalModel: {}\nshadowImage:{}\nmaskImage:{}'.format(removalModel, self.shadowImageDir, self.maskImageDir))
        # 이미지 base64 인코딩 
        shadowImage = base64.b64encode(cv2.imencode('.png', shadow)[1]).decode()
        maskImage = base64.b64encode(cv2.imencode('.png', mask)[1]).decode()
        GTImage = 'None'
        dilateValue = 'None'
        
        if dilate:
            dilateValue = self.maskSetting.dilateValueSlider.value()
            print('Dilate Mask')
            
        if self.GTImageCheckbox.isChecked():
            GTImage = base64.b64encode(cv2.imencode('.png', self.GTImage)[1]).decode()
        
        
        upload = {'mode': 'Interactive', 'removal': True, 'detection': False, 'perf': self.GTImageCheckbox.isChecked(),
                  'removalModel': removalModel, 'detectionModel': self.selectDetectionModel.currentText(), 
                  'shadowImage': shadowImage, 'maskImage': maskImage, 'GTImage': GTImage, 'dilateValue': dilateValue}
        
        # predict requests
        res = requests.post(self.ip + 'predict', data=upload)
        res = json.loads(res.json())
        img_predicted = res['predictImage']
        img_inpainted = res['inpaintingImage']
        img_mask = res['maskImage']
        perf = res['perf']
        
        # 예측 값 디코딩
        img_predicted = cv2.imdecode(np.frombuffer(base64.b64decode(img_predicted), dtype=np.uint8), flags=1)
        img_inpainted = cv2.imdecode(np.frombuffer(base64.b64decode(img_inpainted), dtype=np.uint8), flags=1)
        img_mask = cv2.imdecode(np.frombuffer(base64.b64decode(img_mask), dtype=np.uint8), flags=1)
        
        # RMSE update
        if perf != 'None' and perf != None:
            print(perf)
            if self.shadowRemovalCheckbox:
                value = 'RMSE: {}, RMSE-Shadow: {}, RMSE-nonShadow: {}'.format(round(perf['inpainted']['RMSE'],3), round(perf['inpainted']['RMSE-S'],3), round(perf['inpainted']['RMSE-N'],3))
            else: 
                value = 'inpainted: [RMSE: {}, RMSE-Shadow: {}, RMSE-nonShadow: {}] / predicted: [RMSE: {}, RMSE-Shadow: {}, RMSE-nonShadow: {}]'.format(
                    round(perf['inpainted']['RMSE'],3), round(perf['inpainted']['RMSE-S'],3), round(perf['inpainted']['RMSE-N'],3),
                    round(perf['predicted']['RMSE'],3), round(perf['predicted']['RMSE-S'],3), round(perf['predicted']['RMSE-N'],3)
                )
                
            self.perf.setText(value)
            font = self.perf.font()
            font.setBold(True)
            self.perf.setFont(font)
            self.perf.setAlignment(Qt.AlignCenter)
        
        return img_predicted, img_inpainted, img_mask      
    
    def onMousePress(self, event):
        x1 = event.xdata + 0.5
        y1 = event.ydata + 0.5
        print(f"Mousekey {event.button}, shift {self.shift_key}, ctrl {self.ctrl_key} MouseClick {x1},{y1}")

        if event.button == 1 and self.ctrl_key == True and self.shift_key == True:
            #  Slic 영역 선택       
            self.mouseCapture = True
            self.x1 = int(x1)
            self.y1 = int(y1)
            print(f"MouseCapture : {self.mouseCapture} left, ctrl, no shift --> Select Point {self.x1}, {self.y1}")
    
        if event.button == 1 and self.ctrl_key == False and self.shift_key == True:
            x1 = int(x1)
            y1 = int(y1)
            # 선택된 Mask 영역 삭제
            self.mouseCapture = False
            print(f"MouseCapture : {self.mouseCapture} left, ctrl, shift --> Select Point{self.x1}, {self.y1}")
            cropMask = self.maskImage[self.y1:self.y2, self.x1:self.x2]   
            # segmentation 영역에서 선택한 픽셀의 Label ID 가져오기
            label = self.mask_segments[y1-self.y1, x1-self.x1]
            if label >= 1:
                mask_target = np.where(self.mask_segments == label, 0, 1)
                
                #cropMask = np.logical_or(mask_target, cropMask)
                logical_mask = np.logical_and(self.maskImage[self.y1:self.y2, self.x1:self.x2], mask_target)
                self.maskImage[self.y1:self.y2, self.x1:self.x2] = np.where(logical_mask ==0, 0, 255)
                
                mask = None
                predict = None
                inpainted = self.shadowImage
                if self.runed:
                    mask = self.maskImage
                    predict = self.predictImage
                if self.shadowImageDir != 'None':                
                    self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=inpainted, prediction=predict, mask = mask)
            
        if event.button == 1 and self.ctrl_key == True and self.shift_key == False:
            x1 = int(x1)
            y1 = int(y1)
            # Mask 영역 선택 추가
            if x1 >= self.x1 and x1 <= self.x2 and y1 >= self.y1 and y1 <= self.y2:
                cropMask = self.maskImage[self.y1:self.y2, self.x1:self.x2]   
                
                # segmentation 영역에서 선택한 픽셀의 Label ID 가져오기
                label = self.mask_segments[y1-self.y1, x1-self.x1]
                if label >= 1:
                    mask_target = np.where(self.mask_segments == label, 1, 0)
                    
                    # default dilate 적용 
                    if self.defaultDilateCheckbox.isChecked():
                        kernel = np.ones((5,5), np.uint8)
                        mask_target = mask_target.astype(np.uint8)
                        mask_target = cv2.dilate(mask_target, kernel, self.dilateErodeIterationsSlider.value())

                    
                    #cropMask = np.logical_or(mask_target, cropMask)
                    logical_mask = np.logical_or(self.maskImage[self.y1:self.y2, self.x1:self.x2], mask_target)
                    self.maskImage[self.y1:self.y2, self.x1:self.x2] = np.where(logical_mask ==0, 0, 255)
                    
                    mask = None
                    predict = None
                    inpainted = self.shadowImage
                    if self.runed:
                        mask = self.maskImage
                        predict = self.predictImage
                    
                    if self.shadowImageDir != 'None':
                        self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=inpainted, prediction=predict, mask = mask)
                        
            if event.button == 3 and self.maskingMode and self.maskRetriev and not self.runed and self.ctrl_key == False and self.shift_key == False and self.alt_key == False:
                
                self.xdata.append(x1)
                self.ydata.append(y1)
                self.line.set_data(self.xdata, self.ydata)
                self.canvas.draw()
                
    def onMouseRelease(self, event):
        # 마우스 클릭이 끝날 때 동작
        x2 = event.xdata
        y2 = event.ydata
        
        print(f"MouseCapture : {self.mouseCapture}, onMouseRelease {x2},{y2}")
        if event.button == 1 and self.ctrl_key == True and self.shift_key == True and self.alt_key == False and self.segmentation:
            self.x2 = int(x2)
            self.y2 = int(y2)
            print("-------------<<<<>>>>")
            
            if self.x1 > self.x2:
                self.x1, self.x2 = self.x2, self.x1
            if self.y1 > self.y2:
                self.y1, self.y2 = self.y2, self.y1
            
            tempImage = self.shadowImage.copy()
            cropImage = tempImage[self.y1:self.y2, self.x1:self.x2]
            
            if self.segmentationLogic == 'slic':
                self.mask_segments = slic(cropImage, n_segments=250, compactness=10, sigma=1,  start_label=1)
            elif self.segmentationLogic == 'felzenswalb':
                self.mask_segments = felzenszwalb(cropImage, scale=100, sigma=0.5, min_size=50)
            elif self.segmentationLogic == 'quickShift':
                self.mask_segments = quickshift(cropImage, kernel_size=3, max_dist=6, ratio=0.5)
            else:
                gradient = sobel(rgb2gray(cropImage))
                self.mask_segments = watershed(gradient, markers=250, compactness=0.001)
            
            self.edges = mark_boundaries(cropImage, self.mask_segments, color=self.segmentColor) #, mode='inner')
            self.edges = np.where(self.edges == self.segmentColor, self.segmentColor, cropImage)
            
            #target = self.masking_params["baseimage"]
            tempImage[self.y1:self.y2, self.x1:self.x2] = self.edges
            
            mask = None
            predict = None
            inpainted = self.shadowImage
            if self.runed:
                mask = self.maskImage
                predict = self.predictImage
                
            if self.shadowImageDir != 'None':
                self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=inpainted, prediction=predict, mask = mask)
                
        if event.button == 3 and self.maskingMode and self.maskRetriev and not self.runed and self.ctrl_key == False and self.shift_key == False and self.alt_key == False:
            
            self.xdata.append(x2)
            self.ydata.append(y2)
            
            self.line.set_data(self.xdata, self.ydata)
            self.canvas.draw()
            
            xd = self.xdata.copy()
            yd = self.ydata.copy()
            
            plt.cla()
            self.ax2.axis((0,self.shadowImage.shape[1], self.shadowImage.shape[0], 0)) # 크기를 mask 영상 크기로 고정
            self.ax2.set_axis_off() # axis 표시 제거
            
            plt.tight_layout(pad=0) # padding 제거
            
            xy = np.asarray([xd, yd]).T
            xy=xy[(xy[:,0] != None) & (xy[:,1] != None)]
            xy=xy.T
            
            if self.ax.axis() == (0.0, 1.0, 0.0, 1.0):
                self.val = self.brushSizeSlider.value()
            else: 
                self.val = self.brushSizeSlider.value() * (self.ax.axis()[1] - self.ax.axis()[0]) / self.shadowImage.shape[1]
            
            # 실제 사이즈와 브러시 로 표시할때 나타나는 픽셀간의 차이에 대한 오류 수정
            width = self.val * (self.maskImage.shape[0]/1000)
            line2, = self.ax2.plot(xy[0], xy[1], linewidth = width, color='black')
                
            draw_mask = self.fig2Array(self.fig2)
            draw_mask = cv2.cvtColor(draw_mask, cv2.COLOR_BGR2GRAY)
            draw_mask = 255 - draw_mask

            draw_mask = np.array(np.where(draw_mask > 0, 255, 0), np.uint8)
            
            if self.pen == 'brush':
                self.maskImage = np.logical_or(self.maskImage, draw_mask)
                
            elif self.pen == 'eraser':
                draw_mask = np.logical_and(self.maskImage,draw_mask)
                draw_mask = np.where(draw_mask == False, 255, 0)
                self.maskImage = np.logical_and(self.maskImage, draw_mask)
                
            self.maskImage = np.where(self.maskImage == False, 0 ,255)
            
            self.xdata.clear()
            self.ydata.clear()
            
            mask = None
            predict = None
            inpainted = self.shadowImage
            if self.runed:
                mask = self.maskImage
                predict = self.predictImage
            
            self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=inpainted, prediction=predict, mask = mask)
            
    def onMouseMove(self, event):
        x2 = event.xdata
        y2 = event.ydata
        
        if event.button == 3 and self.maskingMode and self.maskRetriev and not self.runed and self.ctrl_key == False and self.shift_key == False and self.alt_key == False:
            self.xdata.append(x2)
            self.ydata.append(y2)
            self.line.set_data(self.xdata, self.ydata)
            self.canvas.draw()
                
    def zoomEvent(self, event):
        
        self.line, = self.ax.plot(self.xdata, self.ydata, linewidth=self.brushSizeSlider.value(), color=self.brushColor)  

    def keyReleaseEvent(self, event):
        
        if event.key() == Qt.Key_Control:
            self.ctrl_key = False
            print(f"Release Control Key : {self.ctrl_key}")
        if event.key() == Qt.Key_Shift:
            self.shift_key = False
            print(f"Release -- Shift Key {self.shift_key}")
        if event.key() == Qt.Key_Alt:
            self.alt_key = False
            print(f"Release Alt_Key")
                
    def keyPressEvent(self, e):
        print(e.key())
        if e.key() == Qt.Key_Control:
            self.ctrl_key = True
            print("Press -- CTRL Key")
        if e.key() == Qt.Key_Shift:
            self.shift_key = True
            print("Press -- Shift Key")
        
        if e.key() == Qt.Key_Alt:
            self.alt_key = True
            print(f"Press Alt_Key")
            
        if e.key() == 67:       # 'C' : clear segment 
            self.edges = None
            mask = None
            predict = None
            inpainted = self.shadowImage
            if self.runed:
                mask = self.maskImage
                predict = self.predictImage
        
            if self.shadowImageDir != 'None':
                self.ax = self.setImageFunc(fig=self.fig, canvas=self.canvas, checkable=self.maskOverlayCheckbox.isChecked(), num=self.num, alpha = self.maskOpacitySlider.value(), inpainted=inpainted, prediction=predict, mask = mask)
                  
            
        print(f"Key: {e.key()}")
            
            
    
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = Interact()
    myWindow.show()
    sys.exit(app.exec())