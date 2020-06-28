import sys
import cv2
import numpy as np
# import math
# import convolve as conv
import xlsxwriter
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QDialog,QApplication,QMainWindow,QFileDialog
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
# import PIL

class showImage (QMainWindow):
    def __init__(self):
        super(showImage,self).__init__()
        loadUi('main.ui',self)
        self.image=None
        self.actionOpen.triggered.connect(self.openClicked)
        self.actionSave.triggered.connect(self.saveClicked)
        self.actionGrayscale.triggered.connect(self.grayClicked)
        self.actionReduksiNoise.triggered.connect(self.reduksiClicked)
        self.actionSharpening.triggered.connect(self.sharpeningClicked)
        self.actionXls.triggered.connect(self.xlsClicked)
        self.actionDeteksiTepi.triggered.connect(self.tepiClicked)
        # self.actionPlotRGB.triggered.connect(self.rgbClicked)

    @pyqtSlot()
    def openClicked(self):
        flname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\User\\', "Image Files(*.jpg)")
        if flname:
            self.loadImage(flname)
        else:
            print('Invalid Image')

    # def rgbClicked(self):
    #     img = self.image
    #     color = ('b', 'g', 'r')
    #     for i, col in enumerate(color):
    #         histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    #         plt.plot(histr, color=col)
    #         plt.xlim([0, 256])
    #     plt.show()

    def xlsClicked(self):
        workbook = xlsxwriter.Workbook('arrays.xlsx')
        worksheet = workbook.add_worksheet()
        array = self.image
        row = 0
        for col, data in enumerate(array):
            worksheet.write_column(row, col, data)
        workbook.close()

    def saveClicked(self):
        flname, filter = QFileDialog.getSaveFileName(self, 'save file', 'D:\\', "Images Files(*.jpg)")
        if flname:
            cv2.imwrite(flname, self.image)
        else:
            print('Saved')

    def reduksiClicked(self):
        img = self.image
        red = (1.0 / 9) * np.array(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]])
        img_out = cv2.filter2D(img, -1, red)
        print(img_out[40:43, 50:53])
        self.image = img_out
        self.displayImage(2)

    def sharpeningClicked(self):
        img = self.image
        laplace = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        img_out = cv2.filter2D(img, -1, laplace)
        # print(img_out)
        self.image = img_out
        self.displayImage(2)

    def grayClicked(self):
        # cv2.imshow('Original',self.image)
        # img = self.image
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.image[i, j, 0] + 0.587 * self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0, 255)
        self.image = gray
        self.displayImage(2)

    def Konvolusi(self, X, F):
        X_height = X.shape[0]
        X_width = X.shape[1]

        F_height = F.shape[0]
        F_width = F.shape[1]

        H = (F_height) // 2
        W = (F_width) // 2

        out = np.zeros((X_height, X_width))

        for i in np.arange(H + 1, X_height - H):
            for j in np.arange(W + 1, X_width - W):
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum
        return out

    def tepiClicked(self):
        img = self.image

        # 1 REDUKSI NOISE - GAUSSIAN
        gauss = (1.0 / 16) * np.array(
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]])

        imgker = self.Konvolusi(img, gauss)
        imgker = imgker.astype("uint8")

        # 2 FINDING GRADIEN - SOBEL
        x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sx = cv2.filter2D(imgker, cv2.CV_64F, x)
        sy = cv2.filter2D(imgker, cv2.CV_64F, y)

        sobel = cv2.sqrt((sx * sx) + (sy * sy))
        H, W = sobel.shape[:2]
        for i in np.arange(H):
            for j in np.arange(W):
                a = sobel.item(i, j)
                if a > 255:
                    a = 255
                elif a < 0:
                    a = 0
                else:
                    a = a

        theta = np.arctan2(sy, sx)

        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        Z = np.zeros((H, W))

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255

                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = sobel[i, j + 1]
                        r = sobel[i, j - 1]
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = sobel[i + 1, j - 1]
                        r = sobel[i - 1, j + 1]
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = sobel[i + 1, j]
                        r = sobel[i - 1, j]
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = sobel[i - 1, j - 1]
                        r = sobel[i + 1, j + 1]
                    if (sobel[i, j] >= q) and (sobel[i, j] >= r):
                        Z[i, j] = sobel[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        img_N = Z.astype("uint8")
        # 3 NON-MAXIMUM SUPPRESSION
        weak = 127
        strong = 200

        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (a > weak):
                    b = weak
                    if (a > strong):
                        b = 255

                else:
                    b = 0

                img_N.itemset((i, j), b)

        img_H1 = img_N.astype("uint8")

        # 4 HYSTERESIS THRESHOLDING
        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if (img_H1[i, j] == weak):
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or (img_H1[i + 1, j] == strong) or
                                (img_H1[i + 1, j + 1] == strong) or (img_H1[i, j - 1] == strong) or
                                (img_H1[i, j + 1] == strong) or (img_H1[i - 1, j - 1] == strong) or
                                (img_H1[i - 1, j] == strong) or (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass

        img_H2 = img_H1.astype("uint8")

        # cv2.imwrite('10000canny.jpg', img_H2)
        self.image = img_H2
        self.displayImage(2)
        plt.hist(img.ravel(), 255, [0, 255])
        plt.show()

    def loadImage(self,flname):
        self.image=cv2.imread(flname)
        self.displayImage()

    def displayImage(self, windows=1):
        qformat=QImage.Format_Indexed8

        if len(self.image.shape)==3:  # row[0],col[1],channel[2]
            if (self.image.shape[2])==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888


        img=QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        img=img.rgbSwapped()


        if windows==1:
            self.imgIn.setPixmap(QPixmap.fromImage(img))
            self.imgIn.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.imgIn.setScaledContents(True)

        if windows==2:
            self.imgOut.setPixmap(QPixmap.fromImage(img))
            self.imgOut.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.imgOut.setScaledContents(True)

app=QtWidgets.QApplication(sys.argv)
window=showImage()
window.setWindowTitle('Show Image GUI')
window.show()
sys.exit(app.exec_())