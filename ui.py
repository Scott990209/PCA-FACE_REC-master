from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
import cv2
import Face_Rec
from qtui import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import face_pca
import numpy as np

trainimg_path = ""

test_img = []
# 维度，初始化为50维度
n_d = 50


class Show_UI(Ui_MainWindow, QMainWindow):

    def __init__(self, parent=None):
        super(Show_UI, self).__init__(parent)
        # 维度
        self.yuce_path = None
        self.test_img = None
        self.n_d = None
        self.V_r = None
        self.data_mean = None
        self.final_data = None
        self.testimg_path = None
        self.setupUi(self)


    # 图片矢量化
    def img2vector(image):
        # img= cv2.imread("D:/BaiduNetdiskWorkspace/PCA-python-master/FaceDB_orl/001/01.png")
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # 读取图片

        rows = img.shape[0]
        cols = img.shape[1]
        imgVector = np.zeros((1, rows * cols))
        imgVector = np.reshape(img, (1, rows * cols))
        return imgVector

    def pca(self):
        train_face, train_label = face_pca.load_orl(10)
        self.n_d = self.textEdit_2.toPlainText()
        self.final_data, self.data_mean, self.V_r = face_pca.PCA(train_face, n_d)
        self.textEdit_4.setText("降维成功")

    def test(self):
        print(self)
        temp_face = face_pca.img2vector(self.testimg_path)
        temp_face = temp_face - np.tile(self.data_mean, (temp_face.shape[0], 1))
        data_test_new = temp_face * self.V_r  # 得到测试脸在特征向量下的数据
        data_test_new = np.array(data_test_new)  # mat change to array
        data_train_new = np.array(self.final_data)
        diffMat = data_train_new - data_test_new  # 训练数据与测试脸之间距离
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)  # 按行求和
        j = 0
        min = sqDistances[0]
        for i in range(400):
            if (min > sqDistances[i]):
                j = i
                min = sqDistances[i]

        yuce_people = (j + 11) / 10
        photo_num = (j + 11) % 10
        yuce_path = self.data_path
        if (yuce_people <= 9):
            if (photo_num <= 9):
                self.yuce_path = yuce_path + '/00' + str(int(yuce_people)) + '/0' + str(photo_num) + '.png'
            else:
                self.yuce_path = yuce_path + '/00' + str(int(yuce_people)) + '/' + str(photo_num) + '.png'
        else:
            if (photo_num <= 9):
                self.yuce_path = yuce_path + '/0' + str(int(yuce_people)) + '/0' + str(photo_num) + '.png'
            else:
                self.yuce_path = yuce_path + '/0' + str(int(yuce_people)) + '/' + str(photo_num) + '.png'
        result = "第" + str(int(yuce_people)) + "个人"
        self.textEdit_5.setText(result)
        print(self.yuce_path)
        text_png = QtGui.QPixmap(self.yuce_path).scaled(self.label_4.width(), self.label_4.height())
        self.label_4.setPixmap(text_png)

    def setupUi(self, MainWindow):
        super(Show_UI, self).setupUi(MainWindow)
        # Ui_MainWindow.setupUi(self, MainWindow)
        # u = Ui_MainWindow()
        # u.setupUi()
        self.pushButton.clicked.connect(self.test)
        self.pushButton_2.clicked.connect(self.opendir)
        self.pushButton_4.clicked.connect(self.pca)
        self.pushButton_3.clicked.connect(self.openfile)

    def opendir(self):
        fname = QFileDialog.getExistingDirectory(None, "打开文件夹", "./")
        self.data_path = fname
        self.textEdit.setText(fname)

    def openfile(self):
        fname = QFileDialog.getOpenFileName(None, "打开文件", "./", "*.png;;All Files(*)")
        self.testimg_path = fname[0]
        self.textEdit_3.setText(fname[0])
        # # 实际测试图片显示
        png = QtGui.QPixmap(self.testimg_path).scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(png)

        # img = cv2.imread(self.testimg_path, 0)  # 灰度模式读取
        # img = img.reshape(1, -1)
        # img = self.get_pca.transform(img)
        # jpg = QPixmap(self.testimg_path)
        # test_image = QPixmap(jpg)

        # self.label_3.setPixmap(self,test_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Show_UI()
    ui.show()
    sys.exit(app.exec_())
