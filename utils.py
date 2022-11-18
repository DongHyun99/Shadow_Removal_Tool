# -*- coding: utf-8 -*-

from PIL import ImageColor
from PyQt5.QtWidgets import *
from PyQt5 import uic


class Masking(QDialog):
    """Masking setting UI class.
    """
    
    def __init__(self):
        """masking setting 기본 설정 ui
        """
        
        super().__init__()
        self.ui = uic.loadUi('UI/Masking.ui', self)
        self.setWindowTitle('Masking Setting')
        
        # 기본 색상
        self.maskCol = (255, 255, 180)
        self.segmentCol = (255, 180, 180)
        
        # 버튼 클릭 이벤트
        self.maskColorButton.clicked.connect(self.maskColorFunc)
        self.boundaryColorButton.clicked.connect(self.segmentColorFunc)
        self.segmentationCheckbox.clicked.connect(self.segmentationChecked)
        self.slic.clicked.connect(self.slicToggle)
        self.felzenswalb.clicked.connect(self.felzenswalbToggle)
        self.quickShift.clicked.connect(self.quickShiftToggle)
        self.waterShed.clicked.connect(self.waterShedToggle)
        
        # slic을 기본으로 사용
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
        """QColorDialog를 통해 색상을 가져와 적용
        """
        
        self.maskCol = QColorDialog.getColor().name()
        self.maskCol = ImageColor.getcolor(self.maskCol, 'RGB')
        self.maskColorButton.setStyleSheet(
            'background-color:rgb{};'.format(self.maskCol)
        )
        
    def segmentColorFunc(self):
        """QColorDialog를 통해 색상을 가져와 적용
        """
        self.segmentCol = QColorDialog.getColor().name()
        self.segmentCol = ImageColor.getcolor(self.segmentCol, 'RGB')
        self.boundaryColorButton.setStyleSheet(
            'background-color:rgb{};'.format(self.segmentCol)
        )
        
    def segmentationChecked(self):
        """segmentation checkbox가 check되어있을 경우 segmentation 기능 활성화
        """
        
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
        """slic 체크시 다른 알고리즘 체크 해제
        """
        
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
        """felzenswalb 체크시 다른 알고리즘 체크 해제
        """
        
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
        """quickShift 체크시 다른 알고리즘 체크 해제
        """
        
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
        """waterShed 체크시 다른 알고리즘 체크 해제
        """        
        
        if self.waterShed.isChecked():
            if self.felzenswalb.isChecked():
                self.felzenswalb.toggle()
            if self.quickShift.isChecked():
                self.quickShift.toggle()
            if self.slic.isChecked():
                self.slic.toggle()
                
        elif not self.felzenswalb.isChecked() and not self.quickShift.isChecked() and not self.slic.isChecked():
            self.slic.toggle()


class Visualization(QDialog):
    """ Visualization Setting UI Class.
    """
    
    def __init__(self):
        """visualization setting 기본 설정 ui
        """
        
        super().__init__()
        self.ui = uic.loadUi('UI/Visualization.ui', self)
        self.setWindowTitle('Visualization Setting')
        
        self.shadowRemovalCheckbox.clicked.connect(self.shadowRemovalToggle)
        self.predictAndInpaintingCheckbox.clicked.connect(self.removalAndInpaintingToggle)
        self.edgeDetectionCheckbox.clicked.connect(self.edgeDetectionToggle)
        
        
    def shadowRemovalToggle(self):
        """그림자 제거, 그림자 제거 및 non-shadow 부분 덮어쓰기, edge 부분 그림자 제거 모드중 선택
        """
        
        if self.shadowRemovalCheckbox.isChecked():
            if self.predictAndInpaintingCheckbox.isChecked():
                self.predictAndInpaintingCheckbox.toggle()
            if self.edgeDetectionCheckbox.isChecked():
                self.edgeDetectionCheckbox.toggle()
                
        elif not self.predictAndInpaintingCheckbox.isChecked() and not self.edgeDetectionCheckbox.isChecked():
            self.shadowRemovalCheckbox.toggle()
                         
    def removalAndInpaintingToggle(self):
        """그림자 제거, 그림자 제거 및 non-shadow 부분 덮어쓰기, edge 부분 그림자 제거 모드중 선택
        """
        
        if self.predictAndInpaintingCheckbox.isChecked():
            if self.shadowRemovalCheckbox.isChecked():
                self.shadowRemovalCheckbox.toggle()
            if self.edgeDetectionCheckbox.isChecked():
                self.edgeDetectionCheckbox.toggle()
                
        elif not self.shadowRemovalCheckbox.isChecked() and not self.edgeDetectionCheckbox.isChecked():
            self.shadowRemovalCheckbox.toggle()
                
    def edgeDetectionToggle(self):
        """그림자 제거, 그림자 제거 및 non-shadow 부분 덮어쓰기, edge 부분 그림자 제거 모드중 선택
        """
        
        if self.edgeDetectionCheckbox.isChecked():
            if self.shadowRemovalCheckbox.isChecked():
                self.shadowRemovalCheckbox.toggle()
            if self.predictAndInpaintingCheckbox.isChecked():
                self.predictAndInpaintingCheckbox.toggle()
        
        elif not self.shadowRemovalCheckbox.isChecked() and not self.predictAndInpaintingCheckbox.isChecked():
            self.shadowRemovalCheckbox.toggle()
            
class Server(QDialog) :
    """ 서버 setting UI Class.
    """
    
    def __init__(self) :
        """server setting 기본 설정 ui
        """
        
        super().__init__()
        self.ui = uic.loadUi('UI/Server.ui', self)
        self.setWindowTitle('Server Setting')