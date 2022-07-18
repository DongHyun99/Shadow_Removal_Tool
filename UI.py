from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
import glob
import cv2
import numpy as np
import os
from mypredict import predict, predictWithModel
import torch
# from argparse import ArgumentParser

# pushButton_4 : Sekect Mask 
# pushButton : Select Shadow Image

# parser = ArgumentParser()
# parser.add_argument('--src', type=str, help='source directory')
# parser.add_argument('--dst', type=str, help='destination directory')
def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


class Popup(QDialog):
    def __init__(self, main, title):
        super(Popup, self).__init__()
        self.ui = uic.loadUi('Popup.ui', self)
        self.main = main
        self.setWindowTitle(title)
        self.title = title
        #a.append(self.ui.imageLabel)
    def cancel_clicked(self):
        #self.main.loadImage(None, self.main.raw_img)
        self.close()
    def confirm_clicked(self):
        self.main.loadImage(None, self.main.res_img[self.title])
        self.close()


class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.ui = uic.loadUi('untitled.ui', self)
        self.setFixedSize(self.size())
        
        self.masking_mode = self.ui.checkMaskingMode.checkState()
        self.ui.checkMaskingMode.stateChanged.connect(self.checkMaskingMode_changed)

        #self.dilate_mask = self.ui.checkDilateMask.checkState()
        #self.ui.checkDilateMask.stateChanged.connect(self.checkDilateMask_changed)
        print(f"Mouse Masking Mode : {self.masking_mode}")
##        self.base_path = src
##        if not os.path.exists(dst):
##            os.mkdir(dst)
##        self.mask_path = dst
##        self.img_list = os.listdir(src)
##        self.pos = 0
##        self.total = len(self.img_list)

        self.ptsize = 30
        self.t = False
        self.printer = QPrinter()
        
        prjdir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.modeldir = os.path.join(os.path.join(os.getcwd(), 'sdr_models'))
        print(os.listdir(self.modeldir))
        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(os.listdir(self.modeldir))
        self.modelIndex = -1

        self.dialog = Popup(self, 'Inpainted')
        self.dialog2 = Popup(self, 'Predicted')

        self.res_img = dict()

    def checkMaskingMode_changed(self):
        self.masking_mode = self.ui.checkMaskingMode.isChecked()
        print(f"Mouse Masking Mode : {self.masking_mode}")
        
    def popupImage(self, directimg, dialog):
        self.maxw = self.ui.imageLabel.width()
        self.maxh = self.ui.imageLabel.height()
        print('max', self.maxw, self.maxh)
        img = directimg
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] > img.shape[1]:
            img = img.transpose([1,0,2])
        resw = img.shape[1]
        resh = img.shape[0]
        print('resolution:', resw, resh)
        if self.maxw < resw:
            resh = int((self.maxw/resw) * resh)
            resw = self.maxw
            #print('res:', resw, resh)
        if self.maxh < resh:
            resw = int((self.maxh/resh) * resw)
            resh = self.maxh
            #print('res:', resw, resh)
        print('adjusted resolution:', resw, resh)
        img = cv2.resize(img, (resw, resh))
##        self.mask =  np.zeros_like(self.image)
##        self.change = False
##        print('dialog open: ', dialog)
##        print(img)
        self.openImage(image=self.toQImage(img), label=dialog.ui.imageLabel)
        raw_img_BGR = cv2.cvtColor(self.raw_img, cv2.COLOR_RGB2BGR)
        self.loadImage(None, raw_img_BGR)
        dialog.show()
        
##        self.resw = resw
##        self.resh = resh
##        self.maskcox = self.ui.imageLabel.x() + self.ui.imageLabel.width()//2 - resw//2 +10
##        self.maskcoy = self.ui.imageLabel.y() + self.ui.imageLabel.height()//2 - resh//2 +10

        
    def loadImage(self, path, directimg):
        self.maxw = self.ui.imageLabel.width()
        self.maxh = self.ui.imageLabel.height()
        print('max', self.maxw, self.maxh)
        if path == None:
            img = directimg
        else:
            img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        self.raw_img = img
        self.imsize = img.shape[1::-1]          
        self.t = False
        if img.shape[0] > img.shape[1]:
            img = img.transpose([1,0,2])
            self.imsize = img.shape[1::-1]
            self.t = True
        resw = img.shape[1]
        resh = img.shape[0]
        print('[loadimage] adjusted resolution :', resw, resh)
        if self.maxw < resw:
            resh = int((self.maxw/resw) * resh)
            resw = self.maxw
        #    print('res:', resw, resh)
        if self.maxh < resh:
            resw = int((self.maxh/resh) * resw)
            resh = self.maxh
        #    print('res:', resw, resh)
        print('[loadimage] adjusted resolution :', resw, resh)
        self.image = cv2.resize(img, (resw, resh))
        #self.image = cv2.cvtColor(self.image, cv2.COLOR_HLS2RGB)
        self.mask =  np.zeros_like(self.image)
        self.change = False
        self.openImage(image=self.toQImage(self.image), label=self.ui.imageLabel)
        self.clearBuff()
        self.resw = resw
        self.resh = resh
        self.maskcox = self.ui.imageLabel.x() + self.ui.imageLabel.width()//2 - resw//2 +10
        self.maskcoy = self.ui.imageLabel.y() + self.ui.imageLabel.height()//2 - resh//2 +10

    def normalSize(self):
        self.imageLabel.adjustSize()

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()
        self.updateActions()

    def createActions(self):
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S",
                enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False,
                checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)
    def updateActions(self):
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())
       
    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))
        
    def toQImage(self, im, copy=False):
        if im is None:
            return QImage()
        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(self.gray_color_table)
                return qim.copy() if copy else qim
            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
                    return qim.copy() if copy else qim

    def openImage(self, label, image=None, fileName=None):
        if image == None:
            image = QImage(fileName)
        if image.isNull():
            QMessageBox.information(self, "Image Viewer",
                                    "Cannot load %s." % fileName)
            return
        label.setPixmap(QPixmap.fromImage(image))
        #self.fitToWindowAct.setEnabled(True)
        #self.updateActions()
        #if not self.fitToWindowAct.isChecked():
        #    self.imageLabel.adjustSize()

    def runFunc(self):
        if ((self.ui.label_filename.text() != 'Image Name') & self.change) | (self.ui.label_maskname != 'None'):
            
            print(self.ui.radioInplace.isChecked())
            img_inpainted, img_predicted = self.Inpaint()
            cv2.imwrite("temp/inpaint.jpg", img_inpainted)
            cv2.imwrite("temp/predict.jpg", img_predicted)
            if self.ui.radioInplace.isChecked():
                self.raw_img = img_inpainted
                self.loadImage(None, img_inpainted)
            elif self.ui.radioPopup.isChecked():
                self.res_img['Inpainted'] = img_inpainted
                self.popupImage(img_inpainted, self.dialog)
            elif self.ui.radioPopup2.isChecked():
                self.res_img['Inpainted'] = img_inpainted
                self.res_img['Predicted'] = img_predicted
                self.popupImage(img_inpainted, self.dialog)
                self.popupImage(img_predicted, self.dialog2)

            if self.ui.radioAreaRemoveandEdgePrediction.isChecked() == True:
                #--- dilated mask edge  --> convert to float 
                mask = self.maskforedge.astype(np.float32)/255.0
                edge_mask = self.extractMaskEdge(mask)
                edge_mask = edge_mask.astype('float32') / 255
                
                next_data = dict()
                #input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image = img_inpainted.transpose([2, 0, 1])
                input_image = input_image.astype('float32') / 255.0
                next_data['image'] = input_image
                next_data['mask'] = np.expand_dims(edge_mask, 0)
                next_data['shape'] = np.array(img_inpainted.transpose([2, 0, 1]).shape[1:])
                img_inpainted, img_predicted = predictWithModel(next_data, self.model)
                cv2.imwrite("temp/edgeinpained.jpg", img_inpainted)
                cv2.imwrite("temp/edgepredict.jpg", img_predicted)


                        
    def Inpaint(self):
        img = self.raw_img
        if 'hsv' in self.ui.comboBox.currentText():
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = img.transpose([2,0,1])
        img = img.astype('float32') / 255
        img = pad_img_to_modulo(img, 8)
        maskname = self.ui.label_maskname.text()
        if maskname == 'None':
            mask = self.saveMask()
        else:
            mask = cv2.imread(maskname)
            
        mask = mask.transpose([2,0,1])[0]
        
        self.maskforedge = mask 
        
        if self.ui.checkDilateMask.isChecked() == True:    
            dilateValue = self.ui.dilateValue.value()          
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.dilate(mask, kernel, dilateValue)
            print(f"Apply Dilated Mask : 3 * {dilateValue}")

        mask = np.expand_dims(mask, 0)
        mask = mask.astype('float32') / 255
        mask = pad_img_to_modulo(mask, 8)
        
        print('img saved:', img.shape)
        print('masksaved: ', mask.shape)
        data = dict()
        data['image'] = img
        data['mask'] = mask
        data['shape'] = np.array(img.shape[1:])
        idx = self.ui.comboBox.currentIndex()
        if idx != self.modelIndex:
            path = os.path.join(self.modeldir, self.ui.comboBox.currentText())
            print(f"Predict Model Path: {path}")
            img_inpainted, img_predicted, self.model = predict(data, path)
            self.modelIndex = idx
        else:
            print('same model called')
            img_inpainted, img_predicted = predictWithModel(data, self.model)
        if 'hsv' in self.ui.comboBox.currentText():
            img_inpainted = cv2.cvtColor(img_inpainted, cv2.COLOR_HSV2BGR)
            img_predicted = cv2.cvtColor(img_predicted, cv2.COLOR_HSV2BGR)
        else:
            img_inpainted = cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR)
            img_predicted = cv2.cvtColor(img_predicted, cv2.COLOR_RGB2BGR)
        #res = cv2.cvtColor(res, cv2.COLOR_HLS2BGR)
        #res2 = cv2.cvtColor(res2, cv2.COLOR_HLS2BGR)
        return img_inpainted, img_predicted


    def extractMaskEdge(self, mask):    # mask [:,:, 1]
        #mask_double =np.asarry(mask, dtype=np.double)
        gx, gy = np.gradient(mask)
        
        temp_edge = gy * gy + gx * gx
        
        dilateValue = self.ui.dilateValue.value() * 3          
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(temp_edge, kernel, dilateValue)
        print(f"Edge Dilate: {dilateValue}")
 
        temp_edge[temp_edge != 0.0] = 255.0
        temp_edge = np.asarray(temp_edge, dtype=np.uint8)
        
        cv2.imwrite("./temp/mask_edge.png", temp_edge)
        return temp_edge
        

    def saveMask(self):
        if self.change:
            #img_name = self.img_list[pos]
            mask = cv2.resize(self.mask, self.imsize, cv2.INTER_NEAREST)
            if self.t:
                mask = mask.transpose([1,0,2])
            fname = QFileDialog.getSaveFileName(self, 'Save Mask file', './', "Image Files (*.png)")
            print("Save Mask : ", type(fname) , fname)
            if fname[0]:
                if os.path.splitext(fname[0])[1] == '':
                    maskfilename = f"{fname[0]}.jpg"
                    cv2.imwrite(maskfilename, mask)    
                else:
                    cv2.imwrite(fname[0], mask)
            return mask
        else:
            print('no change')
            exit()
##            cv2.imwrite(f'{self.mask_path}/{img_name[:-4]}_mask.png', mask)
##            img = cv2.imread(f'{self.base_path}/{img_name}')
##            img[mask>0] = 255
##            cv2.imwrite(f'{self.mask_path}/{img_name[:-4]}_masked.png', img)

##    def keyPressEvent(self, e):
##        if e.key() == 65:
##            if not self.pos == 0:
##                self.saveMask(self.pos)
##                self.pos -= 1
##                self.loadImage()
##                print('\r' + self.img_list[self.pos], end="")
##                                                
##        elif e.key() == 68:
##            self.pos += 1
##            if self.total == self.pos:
##                self.pos -= 1
##            else:
##                self.saveMask(self.pos-1)
##                self.loadImage()
##                print('\r' + self.img_list[self.pos], end="")
    def saveFile(self):
        if self.ui.label_filename.text() != 'Image Name':
            img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2RGB)
            fname = QFileDialog.getSaveFileName(self, 'Open file', './', "Image Files (*.png *.jpg)")
            print(type(fname) , fname)
            if fname[0]:
                cv2.imwrite(fname[0], img)

    def mouseButtonKind(self, buttons):
        if buttons & Qt.LeftButton:
            print('LEFT')
        if buttons & Qt.MidButton:
            print('MIDDLE')
        if buttons & Qt.RightButton:
            print('RIGHT')

    def clearBuff(self):
        self.buffer = self.image.copy()
        self.m_buffer = self.mask.copy()
        self.c_buffer = self.change
        print(self.change)
        
    def restoreBuff(self):
        self.image = self.buffer
        self.mask = self.m_buffer
        self.change = self.c_buffer
        print(self.change)
        self.openImage(image=self.toQImage(self.image), label=self.ui.imageLabel)

    def mousePressEvent(self, e):  # e ; QMouseEvent
        if self.masking_mode == False:
            return
        print('BUTTON PRESS')
        self.mouseButtonKind(e.buttons())
        if self.ui.label_filename.text() != 'Image Name':
            if e.buttons() & Qt.LeftButton :
                self.clearBuff()
            if e.buttons() & Qt.RightButton :
                self.restoreBuff()

    def mouseReleaseEvent(self, e):  # e ; QMouseEvent
        print('BUTTON RELEASE')
        self.mouseButtonKind(e.buttons())

    def wheelEvent(self, e):  # e ; QWheelEvent
        print('wheel')
        if self.ui.label_filename.text() != 'Image Name':
            if e.angleDelta().y() < 0 and self.ptsize > 5:
                self.ptsize -= 4
            elif e.angleDelta().y() > 0 and self.ptsize < 50:
                self.ptsize += 4
            print(self.ptsize)

    def mouseMoveEvent(self, e):  # e ; QMouseEvent
        if self.masking_mode == False:
            return
        if e.buttons() & Qt.LeftButton & (self.ui.label_filename.text() != 'Image Name'):
            x, y = int(e.x()-self.maskcox-self.ptsize/2), int(e.y()-self.maskcoy-self.ptsize/2)
            x = 0 if x < 0 else x
            y = 0 if y < 0 else y
            x = self.resw - self.ptsize if (x+self.ptsize) > self.resw else x
            y = self.resh - self.ptsize if (y+self.ptsize) > self.resh else y
            self.image[y:y+self.ptsize, x:x+self.ptsize] = 255
            self.mask[y:y+self.ptsize, x:x+self.ptsize] = 255
            print(self.mask.shape)
            print(x, y)
            self.change = True
            self.openImage(image=self.toQImage(self.image), label=self.ui.imageLabel)

    def slot_fileopen(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './images', 'Image files (*.jpg *.gif *.png)')
        print(type(fname) , fname)
        if fname[0]:
            self.ui.label_filename.setText(fname[0])
            self.loadImage(fname[0], None)

    def slot_maskopen(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './images', 'Image files (*.jpg *.gif *.png)')
        print(type(fname) , fname)
        if fname[0]:
            self.ui.label_maskname.setText(fname[0])
            self.change = True
            #self.loadImage(fname[0], None)
        else:
            self.ui.label_maskname.setText('None')
            self.change = False
        
        

if __name__ == '__main__':
    # args = parser.parse_args()
    import sys
##    src = 'C:/lama/test_data/input'
##    dst = 'C:/lama/test_data/careful_mask'
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())
