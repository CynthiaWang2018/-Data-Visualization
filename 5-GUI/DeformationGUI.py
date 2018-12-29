#python2.7
#windows7
from PySide.QtGui import (QWidget, QToolTip,QPushButton, QApplication,QLabel,QHBoxLayout,QVBoxLayout,QComboBox,QSlider)
from PySide import QtGui
from PySide.QtGui import QIcon,QFont,QPixmap,QImage
from PySide.QtCore import QCoreApplication,Qt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
import numpy as np
from PIL import Image
import cv2
#from transformation import local_affine_transformation,affine_points_MLS,preprocess
from Local_Affine import local_affine,preprocess
from Moving_Least_Square import MLS
from morphing import morphingAction
from util import *
    #rgb change to bgr  
def rgb2bgr(img):
    b,g,r,a = cv2.split(img)
    return cv2.merge([r,g,b,a])#Deformation GUI
class Deformation_GUI(QWidget):
    #Initialize the variable
    def __init__(self,oriPath,protoPath,oriPointsPath,protoPointsPath,regionsPath):
        super(Deformation_GUI,self).__init__() 
        localParas,points = preprocess(oriPath,protoPath,oriPointsPath,protoPointsPath,regionsPath,'L2')
        self.oriImg,self.protoImg,self.regionsPoints,self.is_in_regions_fun,self.distance_funs,self.affine_funs = localParas
        self.oriPlotDict,self.protoPlotDict,self.oriPoints ,self.protoPoints = points
        self.oriPoints = np.array(self.oriPoints)
        self.protoPoints = np.array(self.protoPoints)
        self.e = 2
        self.alpha = 1
        self.oriPath = oriPath
        self.protoPath = protoPath
        self.oriPointsPath = ''
        self.protoPointsPath=''
        self.regionsPath = regionsPath
        self.transform = 'Morphing'
        self.newImg = None
        self.initUI()
    #Initialize the GUI
    def initUI(self):
        #Set Window
        QToolTip.setFont(QFont('Serif', 10))
        self.setGeometry(280, 210, 800, 450)
        self.setWindowTitle('Image Registration Based on Control Points')
        #Set Algorithm
        self.comboAffineLabel = QLabel(self)
        self.comboAffineLabel.setText('Algorithm:')
        self.comboAffineLabel.setGeometry(60,270,230,30)     
        self.comboAffine = QComboBox(self)
        self.comboAffine.addItem("Morphing")
        self.comboAffine.addItem("Local Affine Transformation")
        self.comboAffine.addItem("Moving Least Squares")
        self.comboAffine.setGeometry(22,290,225,30)
        self.comboAffine.activated[str].connect(self.affineChoiceChange)   
        #Choose Human Face Image
        self.oriBtn = QPushButton('Human Face Image', self)
        self.oriBtn.setToolTip('Human Face Image')
        self.oriBtn.setGeometry(20,330,230,30)
        self.oriBtn.clicked.connect(self.showOriDialog)
        #Choose Ape or another Human image
        self.protoBtn = QPushButton('Ape or Human image', self)
        self.protoBtn.setToolTip('Ape or Human image')
        self.protoBtn.setGeometry(310,330,230,30)
        self.protoBtn.clicked.connect(self.showProtoDialog)
        #parameter e
        self.eLabel = QLabel(self)
        self.eLabel.setText('E Value:0.00')
        self.eLabel.setGeometry(550,300,200,30)
        self.eSld = QSlider(Qt.Horizontal, self )
        self.eSld.setRange(0,10**5)
        self.eSld.setFocusPolicy(Qt.NoFocus)
        self.eSld.setGeometry(550,330,120,30)
        self.eSld.valueChanged[int].connect(self.changeEValue)
        #parameter alpha
        self.aLabel = QLabel(self)
        self.aLabel.setText('Alpha Value:0.00')
        self.aLabel.setGeometry(680,300,200,30)
        self.aSld = QSlider(Qt.Horizontal, self)
        self.aSld.setRange(0,10**5)
        self.aSld.setFocusPolicy(Qt.NoFocus)
        self.aSld.setGeometry(680,330,100, 30)
        self.aSld.valueChanged[int].connect(self.changeAlphaValue)
        # The Image
        self.oriTextLabel = QLabel(self)
        self.protoTextLabel = QLabel(self)
        self.transTextLabel = QLabel(self)
        self.oriTextLabel.setText('The Human Image')
        self.protoTextLabel.setText('The Ape or another Human Image')
        self.transTextLabel.setText('Deformation Image')
        self.oriTextLabel.move(70,5)
        self.protoTextLabel.move(350,5)
        self.transTextLabel.move(580,5)
        self.oriLabel = QLabel(self)
        self.protoLabel = QLabel(self)
        self.transLabel = QLabel(self)
        pixmap = QPixmap(self.oriPath)
        pixmap2 = QPixmap(self.protoPath)
        self.oriLabel.setPixmap(pixmap)
        self.protoLabel.setPixmap(pixmap2)
        self.transLabel.setPixmap(pixmap)
        #Set Position
        self.oriLabel.setGeometry(20,20,230,230)
        self.protoLabel.setGeometry(290,20,230,230)
        self.transLabel.setGeometry(560,20,230,230)
        self.oriLabel.setScaledContents(True)
        self.protoLabel.setScaledContents(True)
        self.transLabel.setScaledContents(True)
        #import points
        self.loadOriBtn = QPushButton('Deformed Points', self)
        self.loadOriBtn.setToolTip('Load Control Points From Txt File')
        self.loadOriBtn.setGeometry(20,365,230,30)
        self.loadOriBtn.clicked.connect(self.showLoadOriDialog)
        self.loadProtoBtn = QPushButton('Control Points', self)
        self.loadProtoBtn.setToolTip('Load Control Points From Txt File')
        self.loadProtoBtn.setGeometry(310,365,230,30)
        self.loadProtoBtn.clicked.connect(self.showLoadProtoDialog)
        #Deformed
        self.confirmBtn = QPushButton('Deformed', self)
        self.confirmBtn.setToolTip('Deformed')
        self.confirmBtn.setGeometry(580, 365, 150,30)
        self.confirmBtn.clicked.connect(self.transformAction)
        self.show()
    #Deformed Generate
    def transformAction(self):
        
        try:
            #three algorithm 
            if self.transform == 'Morphing':
                self.oriImg = np.array(Image.open(self.oriPath))
                self.protoImg = np.array(Image.open(self.protoPath)) 
                newImg = morphingAction((self.oriPath).encode('utf-8'),self.protoPath.encode('utf-8'),self.oriPointsPath.encode('utf-8'),self.protoPointsPath.encode('utf-8'),self.alpha)
            else:
                localParas,points = preprocess(self.oriPath,self.protoPath,self.oriPointsPath,self.protoPointsPath,self.regionsPath,'L2')
                if points == None:
                    QtGui.QMessageBox.information(self, "Error", localParas)
                    return 
                self.oriImg,self.protoImg,self.regionsPoints,self.is_in_regions_fun,self.distance_funs,self.affine_funs = localParas
                self.oriPlotDict,self.protoPlotDict,self.oriPoints ,self.protoPoints = points
                if self.oriImg.shape[len(self.oriImg.shape)-1] != self.protoImg.shape[len(self.protoImg.shape)-1]:
                    QtGui.QMessageBox.information(self, "Error", "The type of the figures is not the same, please choose another figure")
                    return
                if self.transform == 'Local Affine Transformation':
                    newImg = local_affine(self.oriImg,self.protoImg,self.e,self.regionsPoints,self.is_in_regions_fun,self.distance_funs,self.affine_funs)
                elif self.transform == "Moving Least Squares":
                    newImg = MLS(self.oriImg,self.protoImg,self.oriPoints,self.protoPoints,self.alpha)
        except BaseException :   
            QtGui.QMessageBox.information(self, "Error", "There are error in the point choice or other things.")
            newImg = morphingAction((self.oriPath).encode('utf-8'),self.protoPath.encode('utf-8'),self.oriPointsPath.encode('utf-8'),self.protoPointsPath.encode('utf-8'),self.alpha)

        self.newImg = np.uint8(newImg)
        newImg=rgb2bgr(np.uint8(newImg))
        qimage = QImage(newImg,newImg.shape[1],newImg.shape[0],QImage.Format_ARGB32)
        pixmap_array = QPixmap.fromImage(qimage)
        self.transLabel.setPixmap(pixmap_array)
    
    def showProtoDialog(self):
        fname, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file',
                    '/home',"Image Files (*.png *.jpg *.bmp)")
        if  fname != None and fname != '':
            self.protoPath = fname
            self.protoLabel.setPixmap(QPixmap(self.protoPath))
            
    def showLoadOriDialog(self):
        fname, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file',
                    '/home',"Text files (*.txt)")
        if  fname != None and fname != '':
            self.oriPointsPath = fname
    
    def showLoadProtoDialog(self):
        fname, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file',
                    '/home',"Text files (*.txt)")
        if  fname != None and fname != '':
            self.protoPointsPath = fname
        else:
            self.protoPointsPath = ''
        
    def showOriDialog(self):
        fname, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file',
                    '/home',"Image Files (*.png *.jpg *.bmp)")
        if  fname != None and fname != '':
            self.oriPath = fname
            print(self.oriPath)
            self.oriLabel.setPixmap(QPixmap(self.oriPath))
        
    #e value in Local Affine Algorithm    
    def changeEValue(self,x):
        self.e = 4.0*x/10**5
        self.eLabel.setText('E Value:'+str( '%.2f' % self.e))

    #alpha value in MLS and Morphing Algorithm
    #the alpha range is  different
    def changeAlphaValue(self,x):
        if self.transform == 'Moving Least Squares':
            self.alpha = 2.0*x/10**5
        elif self.transform == 'Morphing':
            self.alpha = 1.0*x/10**5
        self.aLabel.setText('Alpha Value:'+str('%.2f' % self.alpha))
    # 3 Algorithm
    def affineChoiceChange(self,item):   
        if self.transform in ['Moving Least Squares','Morphing'] and item in ['Moving Least Squares','Morphing'] and item != self.transform:
            self.alpha = 0.0
            self.aSld.setValue(self.alpha)
        self.transform = item
#Deformed
if __name__ == '__main__':
    app = QApplication(sys.argv)
    #oriPath = 'zxh.png'
    #protoPath = 'ape.png'
    #ex = Deformation_GUI(oriPath,protoPath,'zxh.txt','ape.txt','regions.txt') 
    #oriPath = 'professorZH.png'
    #protoPath = 'ape.png'
    #ex = Deformation_GUI(oriPath,protoPath,'q_teacherZH.txt','p_points.txt','Value.txt') 
    oriPath = 'zhuzhu_c.png'
    protoPath = 'baibai_c.png'
    ex = Deformation_GUI(oriPath,protoPath,'q_teacherZH.txt','p_points.txt','Value.txt')  
    sys.exit(app.exec_())
    
