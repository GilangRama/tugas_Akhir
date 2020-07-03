import sys
import cv2
import numpy as np
import xlsxwriter
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QMainWindow,QFileDialog
from PyQt5.uic import loadUi

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
        self.actionThreshold.triggered.connect(self.thresholdClicked)

    @pyqtSlot()
    def openClicked(self):
        flname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\User\\', "Image Files(*.jpg)")
        if flname:
            self.loadImage(flname)
        else:
            print('Invalid Image')

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
        self.image = img_out
        self.displayImage(2)

    def sharpeningClicked(self):
        img = self.image
        laplace = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        img_out = cv2.filter2D(img, -1, laplace)
        self.image = img_out
        self.displayImage(2)

    def grayClicked(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.image[i, j, 0] + 0.587 * self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0, 255)
        self.image = gray
        self.displayImage(2)

    def thresholdClicked(self):
        img = self.image
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("Number of Contours = " + str(contours))

        print(contours[1])

        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

        if len(contours) >= 170:
            print("Anthracnose Disease Detected")
            cv2.putText(img, "Anthracnose", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA )
        else:
            print("Healthy")
            cv2.putText(img, "Healthy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("image", img)
        cv2.waitKey(0)

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