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
# from matplotlib import pyplot as plt

    def grayClicked(self):
        img = cv2.imread('file name')
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i,j] = np.clip(0.299 * self.image[i,j,0] + 0.587 * self.image[i,j,1] + 0.114 * self.image[i,j,2])
        # self.image = gray
        # self.displayimage
        cv2.imshow('hasil', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sharpClicked(self):
        img = cv2.imread('file name')
        mean = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_out = conv.convolve(img, mean)
        cv2.imshow('hasil', img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()