from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUiType

import sys
import os

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
import cv2 as cv

from math import pow, log


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=100):
        fig = Figure(dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

MainUi,_ = loadUiType('GUI.ui')
class Main(QMainWindow, MainUi):
    def __init__(self,parent=None):
        super(Main,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        
        self.revertButton.setEnabled(False)
        self.revertButton2.setEnabled(False)
        self.revertButton3.setEnabled(False)
        
        self.Handle_Buttons()
        self.Handle_Themes()
        
        

    image = None
    imagePath = None
    outputImage = None
    accomulatedEffects = []
    
    firstTime = True
    firstTime2 = True
    
    def Handle_Buttons(self):
        self.actionBrowse.triggered.connect(self.browse_image)
        self.actionSave_Image.triggered.connect(self.saveImage)
        
        self.toGray.clicked.connect(self.convert2Gray)
        self.Threshold.clicked.connect(self.threshold)
        self.resampleUp.clicked.connect(self.resample_Up)
        self.subSample.clicked.connect(self.sub_Sample)
        self.changeGrayLevel.clicked.connect(self.change_Gray_Level)
        self.negativeTransform.clicked.connect(self.negative_Transform)
        self.logTransform.clicked.connect(self.log_Transform)
        self.powerLaw.clicked.connect(self.power_Law)
        self.Contrast.clicked.connect(self.contrast)
        self.sliceGrayLevel.clicked.connect(self.slice_Gray_Level)
        self.addConstant.clicked.connect(self.add_Constant)
        self.subtractConstant.clicked.connect(self.subtract_Constant)
        self.subtractImage.clicked.connect(self.subtract_Image)
        self.logicalOperation.clicked.connect(self.logical_Operation)
        self.bitPlaneSlice.clicked.connect(self.bit_Plane_Slice)
        self.specificBitPlaneSlice.clicked.connect(self.specific_Bit_Plane_Slice)
        self.equalizeImage.clicked.connect(self.equalize_Image)
        self.applyAverageFilter.clicked.connect(self.apply_Average_Filter)
        self.applyMinFilter.clicked.connect(self.apply_Min_Filter)
        self.applyMaxFilter.clicked.connect(self.apply_Max_Filter)
        self.applyMedianFilter.clicked.connect(self.apply_Median_Filter)
        self.applyweightedAverageFilter.clicked.connect(self.apply_weightedAverage_Filter)
        self.applysharpening1stDerivativeFilter.clicked.connect(self.apply_sharpening1stDerivative_Filter)
        self.applysharpening2ndDerivativeCompositeLaplacianFilter.clicked.connect(self.apply_sharpening2ndDerivativeCompositeLaplacian_Filter)
        self.applySobelOperatorsFilter.clicked.connect(self.apply_SobelOperators_Filter)
        self.applyRobertsOperatorsFilter.clicked.connect(self.apply_RobertsOperators_Filter)
        self.revertButton.clicked.connect(self.revert)
        self.revertButton2.clicked.connect(self.revert)
        self.revertButton3.clicked.connect(self.revert)
        self.applyIdealLowPassFilter.clicked.connect(self.apply_Ideal_LowPass_Filter)
        self.applyIdealHighPassFilter.clicked.connect(self.apply_Ideal_HighPass_Filter)
        self.applyButterworthLowPassFilter.clicked.connect(self.apply_Butterworth_LowPass_Filter)
        self.applyButterworthHighPassFilter.clicked.connect(self.apply_Butterworth_HighPass_Filter)
        self.applyGaussianLowPassFilter.clicked.connect(self.apply_Gaussian_LowPass_Filter)
        self.applyGaussianHighPassFilter.clicked.connect(self.apply_Gaussian_HighPass_Filter)
    
    def browse_image(self):
        self.imagePath = None
        self.imagePath = QFileDialog.getOpenFileName(self, 'Open Image', './', 'Image Files (*.png *.jpg *.jpeg)')[0]
        if self.imagePath != None and self.imagePath != '':
            self.image = self.RGB2GRAY(imagePATH = self.imagePath)
            self.accomulatedEffects = []
            self.accomulatedEffects.append(self.image)
            
            ##### reset output image #####
            self.output_Image.setVisible(False)
            if not self.firstTime :
                self.groupBox_4.layout().itemAt(0).widget().setVisible(False)
            #####
            
            self.plotOriginalImage()
            self.showOriginalImage()
    
    def saveImage(self):
        text, okPressed = QInputDialog.getText(self, "Save Image", "<html style='font-size:10pt; color:red;'>Enter Image Name:</html>", QLineEdit.Normal, "")
        if okPressed:
            if not os.path.exists('Output'):
                os.makedirs('Output')
            if os.path.exists('output/output_image.png'):
                os.remove('output/output_image.png')
            cv.imwrite(f'output/{text}.png', np.array(self.outputImage))
    
    def revert(self):
        if len(self.accomulatedEffects) == 1:
            self.outputImage = self.accomulatedEffects[-1]
            self.image = self.outputImage
            self.plotOutputImage(False)
            self.showOutputImage()
            
        elif len(self.accomulatedEffects) > 1:
            self.accomulatedEffects = self.accomulatedEffects[:-1] 
            self.outputImage = self.accomulatedEffects[-1]
            self.image = self.outputImage
            self.plotOutputImage(False)
            self.showOutputImage()
            if len(self.accomulatedEffects) == 1:
                self.revertButton.setEnabled(False)
                self.revertButton2.setEnabled(False)
                self.revertButton3.setEnabled(False)
    
    def RGB2GRAY(self, image = None,imagePATH = None):
        if imagePATH != None:
            image = cv.imread(imagePATH)
    
        newImage = []
        for row in image:
            tmpRow = []
            for pixel in row:
                tmpRow.append(int(sum(pixel)/3))
            newImage.append(tmpRow)
        return newImage
    
    def getImageMap(self, image):
        numberList = list(range(0,256))
        valMap = {}
        for row in image:
            for pixel in row:
                if pixel < 0 :
                    pixel = 0
                elif pixel > 255:
                    pixel = 255
                else:
                    pixel = int(pixel)
                if pixel in valMap.keys():
                    valMap[pixel] += 1
                else:
                    valMap[pixel] = 1
                    numberList.remove(pixel)
                    
        for n in numberList:
            valMap[n] = 0
            
        return valMap
    
    def plotOriginalImage(self):
        valMap = self.getImageMap(self.image)
        
        sc = MplCanvas(dpi=100)
        sc.axes.stem(list(valMap.keys()), list(valMap.values()))
        sc.axes.set_xlabel('Pixel Value')
        sc.axes.set_ylabel('Frequency')
        
        if self.firstTime:
            self.layoutVert1 = QVBoxLayout()
            self.layoutVert1.addWidget(sc)
            self.groupBox.setLayout(self.layoutVert1)
            self.firstTime = False
        else:
            self.layoutVert1.replaceWidget(self.groupBox.layout().itemAt(0).widget(), sc)
    
    def showOriginalImage(self):
        #self.original_Image.setScaledContents(True)
        self.original_Image.setPixmap(QPixmap(self.imagePath))
        
    def plotOutputImage(self, addOutput = True):
        if addOutput:
            self.accomulatedEffects.append(self.outputImage)
            self.image = self.outputImage
            if len(self.accomulatedEffects) > 1:
                self.revertButton.setEnabled(True)
                self.revertButton2.setEnabled(True)
                self.revertButton3.setEnabled(True)
        
        valMap = self.getImageMap(self.outputImage)
    
        sc2 = MplCanvas(dpi=100)
        sc2.axes.stem(list(valMap.keys()), list(valMap.values()))
        sc2.axes.set_xlabel('Pixel Value')
        sc2.axes.set_ylabel('Frequency')
        
        if self.firstTime2:
            self.layoutVert2 = QVBoxLayout()
            self.layoutVert2.addWidget(sc2)
            self.groupBox_4.setLayout(self.layoutVert2)
            self.firstTime2 = False
        else:
            self.layoutVert2.replaceWidget(self.groupBox_4.layout().itemAt(0).widget(), sc2)
            self.groupBox_4.layout().itemAt(0).widget().setVisible(True)
    
    def showOutputImage(self):
        #self.output_Image.setScaledContents(True)
        if not os.path.exists('temp'):
            os.makedirs('temp')
        if os.path.exists('temp/tmp.png'):
            os.remove('temp/tmp.png')
            
        cv.imwrite('temp/tmp.png', np.array(self.outputImage))
        self.output_Image.setPixmap(QPixmap('temp/tmp.png'))  
        self.output_Image.setVisible(True)
    
    def convert2Gray(self):
        self.outputImage = self.image
        self.plotOutputImage()
        self.showOutputImage()
    
    def threshold(self):
        myMap = self.getImageMap(self.image)
        width, heigth = np.array(self.image).shape
        pixels = width * heigth
        comulativeSum = 0
        threshold = None
        
        for i in range(256):
            if comulativeSum < (pixels/2):
                comulativeSum += myMap[i]
            else:
                threshold = i
                break
                
        newImage = []
        for row in self.image:
            tmpRow = []
            for pixel in row:
                tmpRow.append(255 if pixel > threshold else 0)
            newImage.append(tmpRow)
            
        self.outputImage = newImage
        self.plotOutputImage()
        self.showOutputImage()
    
    def resample_Up(self):
        scale, okPressed = QInputDialog.getInt(self, "Resample Up", "<html style='font-size:10pt; color:red;'>Enter Scale Factor :</html>", QLineEdit.Normal)
        if okPressed:
            newImage = []
            for row in self.image:
                tmp = []
                for pixel in row:
                    tmp.append(pixel)
                    # duplicate column
                    for _ in range(scale-1):
                        tmp.append(pixel)
                        
                newImage.append(tmp)
                # duplicate row
                for _ in range(scale-1):
                    newImage.append(tmp)
                    
            self.outputImage = newImage
            self.plotOutputImage()
            self.showOutputImage()
        
    def sub_Sample(self):
        n_subSamples, okPressed = QInputDialog.getInt(self, "subSample", "<html style='font-size:10pt; color:red;'>Enter number of subSampling times :</html>", QLineEdit.Normal)
        
        if okPressed:
            
            newImage = None
            self.outputImage = self.image
            rowCounter = 0
            columnCounter = 0
            nIgnoredSamples = 1
            
            for _ in range(n_subSamples):
                newImage = []
                for row in self.outputImage:
                    tmp = []
                    for column in row:
                        if columnCounter == 0:
                            columnCounter = nIgnoredSamples
                            tmp.append(column)
                        else:
                            columnCounter -= 1
                        
                    if rowCounter == 0:
                        rowCounter = nIgnoredSamples
                        newImage.append(tmp)
                    else:
                        rowCounter -= 1
                        
                self.outputImage = newImage
            self.plotOutputImage()
            self.showOutputImage()
            
    def change_Gray_Level(self):
        grayLevelBit, okPressed = QInputDialog.getInt(self, "Change Gray Level", "<html style='font-size:10pt; color:red;'>Enter Gray Level to change to :</html>", QLineEdit.Normal)
    
        if okPressed:
            newImage = []
            TARGETED_GRAY_LEVEL = pow(2, grayLevelBit)
            TARGET_COMPR_FACTOR = 256/TARGETED_GRAY_LEVEL


            for row in self.image:
                tmpImage = []
                for pixel in row:
                    tmpImage.append(np.floor( (pixel/256) * TARGETED_GRAY_LEVEL) * TARGET_COMPR_FACTOR)
                newImage.append(tmpImage)
                
            self.outputImage = newImage
            self.plotOutputImage()
            self.showOutputImage()
            
    def negative_Transform(self):
        newImage = []
        for row in self.image:
            tmpImage = []
            for pixel in row:
                tmpImage.append(255 - pixel)
            newImage.append(tmpImage)
            
        self.outputImage = newImage
        self.plotOutputImage()
        self.showOutputImage()

    def log_Transform(self):
        constant, okPressed = QInputDialog.getInt(self, "Adding Constant", "<html style='font-size:10pt; color:red;'>Enter Constant integer :</html>", QLineEdit.Normal)
        if okPressed:
            newImage = []
            for row in self.image:
                tmpImage = []
                for pixel in row:
                    tmpImage.append(round(constant * log(1 + pixel)))
                newImage.append(tmpImage)
                
            self.outputImage = newImage
            self.plotOutputImage()
            self.showOutputImage()

    def power_Law(self):
            """gamma = 1 ==> no change"""
            """gamma > 1 ==> brighter"""
            """gamma < 1 ==> darker"""
            gamma, okPressed = QInputDialog.getDouble(self, "Adding Gamma", "<html style='font-size:10pt; color:red;'>Enter Gamma :</html>", QLineEdit.Normal)
            if okPressed:
                constant, okPressed = QInputDialog.getInt(self, "Adding Constant", "<html style='font-size:10pt; color:red;'>Enter Constant integer :</html>", QLineEdit.Normal)
                if okPressed:
                    newImage = []
                    for row in self.image:
                        tmpImage = []
                        for pixel in row:
                            val = pixel ** gamma
                            val = int(val * constant)
                            tmpImage.append(val if val < 255 else 255)
                        newImage.append(tmpImage)
                    
                self.outputImage = newImage
                self.plotOutputImage()
                self.showOutputImage()
        
    def contrast(self):
        newImage = []
        minValue = np.min(self.image)
        maxValue = np.max(self.image)
        for row in self.image:
            tmpImage = []
            for pixel in row:
                result = int( ( (255 - 0) / (maxValue - minValue) ) * (pixel - minValue) + 0 )
                tmpImage.append(result)
            newImage.append(tmpImage)
            
        self.outputImage = newImage
        self.plotOutputImage()
        self.showOutputImage()
        
    def slice_Gray_Level(self):
        approach, okPressed = QInputDialog.getInt(self, "Select Approach", "<html style='font-size:10pt; color:red;'>Enter Approach to use :</html>", QLineEdit.Normal)
        if okPressed:
            startRange, okPressed = QInputDialog.getInt(self, "Select Start Range", "<html style='font-size:10pt; color:red;'>Enter Start Range :</html>", QLineEdit.Normal)
            if okPressed:
                endRange, okPressed = QInputDialog.getInt(self, "Select End Range", "<html style='font-size:10pt; color:red;'>Enter End Range :</html>", QLineEdit.Normal)
                if okPressed:
                    newImage = []
                    for row in self.image:
                        tmpImage = []
                        for pixel in row:
                            if pixel >= startRange and pixel <= endRange:
                                tmpImage.append(255)
                            else:
                                if approach == 1:
                                    tmpImage.append(0)
                                elif approach == 2:
                                    tmpImage.append(pixel)
                        newImage.append(tmpImage)
                    
                    self.outputImage = newImage
                    self.plotOutputImage()
                    self.showOutputImage()

    def add_Constant(self):
        constant, okPressed = QInputDialog.getInt(self, "Adding Constant", "<html style='font-size:10pt; color:red;'>Enter Constant integer :</html>", QLineEdit.Normal)
        if okPressed:
            newImage = []
            for row in self.image:
                tmpRow = []
                for pixel in row:
                    tmpRow.append(pixel + constant)
                newImage.append(tmpRow)
                
            self.outputImage = newImage
            self.plotOutputImage()
            self.showOutputImage()

    def subtract_Constant(self):
        constant, okPressed = QInputDialog.getInt(self, "Adding Constant", "<html style='font-size:10pt; color:red;'>Enter Constant integer :</html>", QLineEdit.Normal)
        if okPressed:
            newImage = []
            for row in self.image:
                tmpRow = []
                for pixel in row:
                    tmpRow.append(pixel - constant)
                newImage.append(tmpRow)
                
            self.outputImage = newImage
            self.plotOutputImage()
            self.showOutputImage()
    
    def subtract_Image(self):
        secondFilePath = QFileDialog.getOpenFileName(self, 'Open Image', './', 'Image Files (*.png *.jpg)')[0]
        if secondFilePath != '':
            image2 = self.RGB2GRAY(imagePATH = secondFilePath)
            newImage = []
            for row1,row2 in zip(self.image, image2):
                tmpRow = []
                for pixel1, pixel2 in zip(row1, row2):
                    tmpRow.append(np.abs(pixel1 - pixel2))
                newImage.append(tmpRow)
            
            self.outputImage = newImage
            self.plotOutputImage()
            self.showOutputImage()
    
    def logical_Operation(self):
        secondFilePath = QFileDialog.getOpenFileName(self, 'Open Image', './', 'Image Files (*.png *.jpg *.jpeg)')[0]
        if secondFilePath != '':
            operator, okPressed = QInputDialog.getText(self, "Choosing Operator", "<html style='font-size:10pt; color:red;'>Enter Operator (and, or, xor) :</html>", QLineEdit.Normal)
            if okPressed :
                image2 = self.RGB2GRAY(imagePATH = secondFilePath)
                newImage = []
                
                image1 = np.array(self.image)
                image2 = np.array(image2)
                if operator == "and":
                    newImage = cv.bitwise_and(image1, image2)
                elif operator == "or":
                    newImage = cv.bitwise_or(image1, image2)
                elif operator == "xor":
                    newImage = cv.bitwise_xor(image1, image2)
                    
                self.outputImage = newImage
                self.plotOutputImage()
                self.showOutputImage()    
        
    def decimalTo_8bit_Binary(self, n):
        binaryNum =  bin(n).replace("0b", "")
        for _ in range(len(binaryNum), 8):
            binaryNum = "0" + binaryNum
        return binaryNum

    def getPixelValue(self, binaryNum, n_slices):
        value = 0
        for i in range(n_slices):
            if binaryNum[i] == "1":
                value += 2**int(7-i)
        return value

    def bit_Plane_Slice(self):
        n_slices, okPressed = QInputDialog.getInt(self, "number of slices", "<html style='font-size:10pt; color:red;'>Enter Number of slices to extract :</html>", QLineEdit.Normal)
        if okPressed:
            newImage = []
            for row in self.image:
                tmpImage = []
                for pixel in row:
                    tmpImage.append(self.getPixelValue(self.decimalTo_8bit_Binary(pixel), n_slices))
                newImage.append(tmpImage)
            
            self.outputImage = newImage
            self.plotOutputImage()
            self.showOutputImage()

    def getSpecificBitPixelValue(self, binaryNum, specificBit):
        return 2 ** (7 - specificBit) if binaryNum[specificBit] == "1" else 0
    
    def specific_Bit_Plane_Slice(self):
        specificBit, okPressed = QInputDialog.getInt(self, "Select Specific Bit", "<html style='font-size:10pt; color:red;'>Enter Specific Bit to extract :</html>", QLineEdit.Normal)
        if okPressed:
            newImage = []
            for row in self.image:
                tmpImage = []
                for pixel in row:
                    tmpImage.append(self.getSpecificBitPixelValue(self.decimalTo_8bit_Binary(pixel), specificBit))
                newImage.append(tmpImage)
            
            self.outputImage = newImage
            self.plotOutputImage()
            self.showOutputImage()
            
    def equalize_Image(self):
        imageMap = self.getImageMap(self.image)
        comulativeMap = {}
        comulativeMap[0] = imageMap[0]
        for i in range(1, 256):
            comulativeMap[i] = imageMap[i] + comulativeMap[i-1]
        
        newImage = []
        for row in self.image:
            tmpImage = []
            for pixel in row:
                result = round( (255/comulativeMap[255]) * comulativeMap[pixel] )
                tmpImage.append(result)
            newImage.append(tmpImage)
        
        self.outputImage = newImage
        self.plotOutputImage()
        self.showOutputImage()
        
    def getPixelMask(self, image, index, index2, filterSize):
        maskStartUpRange = index - int(filterSize/2)
        maskEndBottomRange = index + int(filterSize/2)
        maskStartLeftRange = index2 - int(filterSize/2)
        maskEndRightRange = index2 + int(filterSize/2)
        numberOfRows = len(image)
        numberOfColumns = len(image[0])
        mask = []
        
        if filterSize == 2:
            pixel1 = image[index][index2]
            pixel2 = image[index][index2+1] if index2+1 < numberOfColumns else 0
            pixel3 = image[index+1][index2] if index+1 < numberOfRows else 0
            pixel4 = image[index+1][index2+1] if index2+1 < numberOfColumns and index+1 < numberOfRows else 0
            
            mask.append([[pixel1, pixel2],[pixel3, pixel4]])
        else:
            for i in  range(maskStartUpRange, maskEndBottomRange + 1):
                tmpRow = []
                for j in range(maskStartLeftRange, maskEndRightRange + 1):
                    if i < 0 or j < 0 or i >= numberOfRows or j >= numberOfColumns:
                        tmpRow.append(0)
                    else:
                        tmpRow.append(image[i][j])
                mask.append(tmpRow)
        
        mask = np.array(mask).reshape(filterSize, filterSize)
        return mask

    def getMasks(self, filterSize=3):
        masks = []
        for index in range(len(self.image)):
            for index2 in range(len(self.image[0])):
                masks.append(self.getPixelMask(self.image, index, index2, filterSize))
        return masks
    
    def applySpecificFilter(self, masks, filterType, shape):
        newImage = []
        if filterType == "average":
            for i in masks:
                newImage.append(round(np.average(i)))
                
        elif filterType == "weightedAverage":
            for i in masks:
                weightedFilter = np.array([[1,2,1],[2,4,2],[1,2,1]])
                value = np.sum(np.multiply(i, weightedFilter))/16
                newImage.append(round(value))
                
        elif filterType == "median":
            for i in masks:
                newImage.append(np.median(i))
                
        elif filterType == "max":
            for i in masks:
                newImage.append(np.max(i))
                
        elif filterType == "min":
            for i in masks:
                newImage.append(np.min(i))
                
        elif filterType == "sharpening1stDerivative":
            for i in range(len(self.image)):
                for j in range(len(self.image[0])):
                    currentPixel = self.image[i][j]
                    nextPixel = self.image[i][j+1] if j+1 < len(self.image[0]) else 0
                    
                    value = currentPixel + (nextPixel - currentPixel)
                    newImage.append(value)
                    
        elif filterType == "sharpening2ndDerivativeCompositeLaplacian":
            for i in masks:
                compositeFilter = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                
                value = np.sum(np.multiply(i, compositeFilter))
                if value < 0:
                    value = 0
                elif value > 255:
                    value = 255
                newImage.append(value)
        
        elif filterType == "SobelOperators":
            for i in masks:
                filter1 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
                filter2 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
                
                value = np.sum(np.multiply(i, filter1)) + np.sum(np.multiply(i, filter2))
                if value < 0:
                    value = 0
                elif value > 255:
                    value = 255
                newImage.append(value if value <= 255 else 0)
        
        elif filterType == "RobertsOperators":
            for i in masks:
                filter1 = np.array([[-1,0],[0,1]])
                filter2 = np.array([[0,-1],[1,0]])
                
                value = np.sum(np.multiply(i, filter1)) + np.sum(np.multiply(i, filter2))
                if value < 0:
                    value = 0
                elif value > 255:
                    value = 255
                newImage.append(value if value <= 255 else 0)
        
        newImage = np.array(newImage).reshape(shape)
        return newImage              

    def applyFilter(self, filterSize = None, filterType = None):
        """filterType: average, median, max, min, weightedAverage, sharpening1stDerivative, sharpening2ndDerivativeCompositeLaplacian, SobelOperator, RobertsOperator"""
        """filterSize: odd number only (3, 5, 7, 9, 11, 13, 15, ...) Except for RobertsOperator filterSize = 2 and for Robert and weigtedAverage filterSize = 3"""
        
        masks = self.getMasks(filterSize)
        newImage = self.applySpecificFilter(masks, filterType, (len(self.image), len(self.image[0])))
        
        self.outputImage = newImage
        self.plotOutputImage()
        self.showOutputImage()
        
    def apply_Average_Filter(self):
        filterSize, okPressed = QInputDialog.getInt(self, "Filter Size", "<html style='font-size:10pt; color:red;'>Enter Filter Size (3, 5, 7, ...) :</html>", QLineEdit.Normal)
        if okPressed:
            self.applyFilter(filterSize, "average")
    def apply_Min_Filter(self):
        filterSize, okPressed = QInputDialog.getInt(self, "Filter Size", "<html style='font-size:10pt; color:red;'>Enter Filter Size (3, 5, 7, ...) :</html>", QLineEdit.Normal)
        if okPressed:
            self.applyFilter(filterSize, "min")
    def apply_Max_Filter(self):
        filterSize, okPressed = QInputDialog.getInt(self, "Filter Size", "<html style='font-size:10pt; color:red;'>Enter Filter Size (3, 5, 7, ...) :</html>", QLineEdit.Normal)
        if okPressed:
            self.applyFilter(filterSize, "max")
    def apply_Median_Filter(self):
        filterSize, okPressed = QInputDialog.getInt(self, "Filter Size", "<html style='font-size:10pt; color:red;'>Enter Filter Size (3, 5, 7, ...) :</html>", QLineEdit.Normal)
        if okPressed:
            self.applyFilter(filterSize, "median")
    def apply_weightedAverage_Filter(self):
        self.applyFilter(3, "weightedAverage")
    def apply_sharpening1stDerivative_Filter(self):
        self.applyFilter(3, "sharpening1stDerivative")
    def apply_sharpening2ndDerivativeCompositeLaplacian_Filter(self):
        self.applyFilter(3, "sharpening2ndDerivativeCompositeLaplacian")
    def apply_SobelOperators_Filter(self):
        self.applyFilter(3, "SobelOperators")
    def apply_RobertsOperators_Filter(self):
        self.applyFilter(2, "RobertsOperators")
        
    
    def apply_Ideal_LowPass_Filter(self):
        D0, okPressed = QInputDialog.getInt(self, "Choosing D0", "<html style='font-size:10pt; color:red;'>Enter Cutoff Value (D0) :</html>", QLineEdit.Normal)
        if okPressed:
            M = len(self.image)
            N = len(self.image[0])

            FT_img = np.fft.fft2(self.image)

            u = np.arange(0, M)
            idx = np.argwhere(u>M/2)
            u[idx] = u[idx]-M

            v = np.arange(0, N)
            idy = np.argwhere(v>N/2)
            v[idy] = v[idy]-N

            V, U = np.meshgrid(v, u)

            D = np.sqrt(U**2 + V**2)

            H = (D <= D0)

            self.outputImage = np.real(np.fft.ifft2(FT_img * H))
            self.plotOutputImage()
            self.showOutputImage()
    def apply_Ideal_HighPass_Filter(self):
        D0, okPressed = QInputDialog.getInt(self, "Choosing D0", "<html style='font-size:10pt; color:red;'>Enter Cutoff Value (D0) :</html>", QLineEdit.Normal)
        if okPressed:
            M = len(self.image)
            N = len(self.image[0])

            FT_img = np.fft.fft2(self.image)

            u = np.arange(0, M)
            idx = np.argwhere(u>M/2)
            u[idx] = u[idx]-M

            v = np.arange(0, N)
            idy = np.argwhere(v>N/2)
            v[idy] = v[idy]-N

            V, U = np.meshgrid(v, u)

            D = np.sqrt(U**2 + V**2)

            H = (D > D0)

            self.outputImage = np.real(np.fft.ifft2(FT_img * H))
            self.plotOutputImage()
            self.showOutputImage()
    def apply_Butterworth_LowPass_Filter(self):
        D0, okPressed = QInputDialog.getInt(self, "Choosing D0", "<html style='font-size:10pt; color:red;'>Enter Cutoff Value (D0) :</html>", QLineEdit.Normal)
        if okPressed:
            n, okPressed = QInputDialog.getInt(self, "Choosing n", "<html style='font-size:10pt; color:red;'>Enter Value of n :</html>", QLineEdit.Normal)
            if okPressed:
                M = len(self.image)
                N = len(self.image[0])

                FT_img = np.fft.fft2(self.image)

                u = np.arange(0, M)
                idx = np.argwhere(u>M/2)
                u[idx] = u[idx]-M

                v = np.arange(0, N)
                idy = np.argwhere(v>N/2)
                v[idy] = v[idy]-N

                V, U = np.meshgrid(v, u)

                D = np.sqrt(U**2 + V**2)

                H = 1/(1+(D/D0)**n)
                
                self.outputImage = np.real(np.fft.ifft2(FT_img * H))
                self.plotOutputImage()
                self.showOutputImage()   
    def apply_Butterworth_HighPass_Filter(self):
        D0, okPressed = QInputDialog.getInt(self, "Choosing D0", "<html style='font-size:10pt; color:red;'>Enter Cutoff Value (D0) :</html>", QLineEdit.Normal)
        if okPressed:
            n, okPressed = QInputDialog.getInt(self, "Choosing n", "<html style='font-size:10pt; color:red;'>Enter Value of n :</html>", QLineEdit.Normal)
            if okPressed:
                M = len(self.image)
                N = len(self.image[0])

                FT_img = np.fft.fft2(self.image)
                
                u = np.arange(0, M)
                idx = np.argwhere(u>M/2)
                u[idx] = u[idx]-M

                v = np.arange(0, N)
                idy = np.argwhere(v>N/2)
                v[idy] = v[idy]-N

                V, U = np.meshgrid(v, u)

                D = np.sqrt(U**2 + V**2)

                H = 1/(1+(D0/D)**n)
                
                self.outputImage = np.real(np.fft.ifft2(FT_img * H))
                self.plotOutputImage()
                self.showOutputImage()  
    def apply_Gaussian_LowPass_Filter(self):
        D0, okPressed = QInputDialog.getInt(self, "Choosing D0", "<html style='font-size:10pt; color:red;'>Enter Cutoff Value (D0) :</html>", QLineEdit.Normal)
        if okPressed:
            M = len(self.image)
            N = len(self.image[0])

            FT_img = np.fft.fft2(self.image)
            
            D0 = (D0**2)*2
            u = np.arange(0, M)
            idx = np.argwhere(u>M/2)
            u[idx] = u[idx]-M

            v = np.arange(0, N)
            idy = np.argwhere(v>N/2)
            v[idy] = v[idy]-N

            V, U = np.meshgrid(v, u)

            D = np.sqrt(U**2 + V**2)
            D = -D**2

            H = np.exp(D/D0)
            
            self.outputImage = np.real(np.fft.ifft2(FT_img * H))
            self.plotOutputImage()
            self.showOutputImage()
    def apply_Gaussian_HighPass_Filter(self):
        D0, okPressed = QInputDialog.getInt(self, "Choosing D0", "<html style='font-size:10pt; color:red;'>Enter Cutoff Value (D0) :</html>", QLineEdit.Normal)
        if okPressed:
            n, okPressed = QInputDialog.getInt(self, "Choosing n", "<html style='font-size:10pt; color:red;'>Enter Value of n :</html>", QLineEdit.Normal)
            if okPressed:
                M = len(self.image)
                N = len(self.image[0])

                FT_img = np.fft.fft2(self.image)
                
                D0 = (D0**2)*2
                u = np.arange(0, M)
                idx = np.argwhere(u>M/2)
                u[idx] = u[idx]-M

                v = np.arange(0, N)
                idy = np.argwhere(v>N/2)
                v[idy] = v[idy]-N

                V, U = np.meshgrid(v, u)

                D = np.sqrt(U**2 + V**2)
                D = -D**2

                H = 1-np.exp(D/D0)
                
                self.outputImage = np.real(np.fft.ifft2(FT_img * H))
                self.plotOutputImage()
                self.showOutputImage()


    def Handle_Themes(self):
        self.actionAMOLED.triggered.connect(lambda: self.Apply_AMOLED_Style())
        self.actionAqua.triggered.connect(lambda: self.Apply_Aqua_Style())
        self.actionClassic.triggered.connect(lambda: self.Apply_Classic_Style())
        self.actionConsole.triggered.connect(lambda: self.Apply_Console_Style())
        self.actionDarkBlue.triggered.connect(lambda: self.Apply_DarkBlue_Style())
        self.actionDarkGray.triggered.connect(lambda: self.Apply_DarkGray_Style())
        self.actionDarkOrange.triggered.connect(lambda: self.Apply_DarkOrange_Style())
        self.actionElegantDark.triggered.connect(lambda: self.Apply_ElegantDark_Style())
        self.actionMacOS.triggered.connect(lambda: self.Apply_MacOS_Style())
        self.actionManjaroMix.triggered.connect(lambda: self.Apply_ManjaroMix_Style())
        self.actionMaterialDark.triggered.connect(lambda: self.Apply_MaterialDark_Style())
        self.actionNeonButtons.triggered.connect(lambda: self.Apply_NeonButtons_Style())
        self.actionUbuntu.triggered.connect(lambda: self.Apply_Ubuntu_Style())
        self.actionOriginal.triggered.connect(lambda: self.Apply_Original_Style())

    def Apply_AMOLED_Style(self):
        style = open('./themes/AMOLED.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_Aqua_Style(self):
        style = open('./themes/Aqua.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_Classic_Style(self):
        style = open('./themes/Classic.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_Console_Style(self):
        style = open('./themes/Console.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_DarkBlue_Style(self):
        style = open('./themes/DarkBlue.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_DarkGray_Style(self):
        style = open('./themes/DarkGray.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_DarkOrange_Style(self):
        style = open('./themes/DarkOrange.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_ElegantDark_Style(self):
        style = open('./themes/ElegantDark.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_MacOS_Style(self):
        style = open('./themes/MacOS.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_ManjaroMix_Style(self):
        style = open('./themes/ManjaroMix.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_MaterialDark_Style(self):
        style = open('./themes/MaterialDark.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_NeonButtons_Style(self):
        style = open('./themes/NeonButtons.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_Ubuntu_Style(self):
        style = open('./themes/Ubuntu.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)
    def Apply_Original_Style(self):
        style = open('./themes/Original.qss', 'r')
        style = style.read()
        self.setStyleSheet(style)

def main():
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
