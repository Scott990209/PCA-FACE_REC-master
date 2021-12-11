import os
from numpy import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import mpl


# mpl.rcParams['font.sans-serif'] = ['SimHei']


# 图片矢量化
def img2vector(image):
    # img= cv2.imread("D:/BaiduNetdiskWorkspace/PCA-python-master/FaceDB_orl/001/01.png")
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # 读取图片

    rows = img.shape[0]
    cols = img.shape[1]
    imgVector = np.zeros((1, rows * cols))
    imgVector = np.reshape(img, (1, rows * cols))
    return imgVector


orlpath = "D:/BaiduNetdiskWorkspace/PCA-python-master/FaceDB_orl"


# 读入人脸库,每个人随机选择k张作为训练集,其余构成测试集
def load_orl(k):
    train_face = np.zeros((400, 112 * 92))
    train_label = np.zeros(400)  # [0,0,.....0](共40*k个0)
    for i in range(40):  # 共有40个人
        people_num = i + 1
        for j in range(10):  # 每个人都有10张照片
            if (i <= 8):
                if (j <= 8):
                    image = orlpath + '/00' + str(people_num) + '/0' + str(j + 1) + '.png'
                else:
                    image = orlpath + '/00' + str(people_num) + '/' + str(j + 1) + '.png'
            else:
                if (j <= 8):
                    image = orlpath + '/0' + str(people_num) + '/0' + str(j + 1) + '.png'
                else:
                    image = orlpath + '/0' + str(people_num) + '/' + str(j + 1) + '.png'
            # 读取图片并进行矢量化
            img = img2vector(image)
            # 构成训练集
            train_face[i * 10 + j, :] = img
            train_label[i * 10 + j] = people_num
    return train_face, train_label


# 定义PCA算法
def PCA(data, r):
    data = np.float32(np.mat(data))
    rows, cols = np.shape(data)
    data_mean = np.mean(data, 0)  # 对列求平均值
    A = data - np.tile(data_mean, (rows, 1))  # 将所有样例减去对应均值得到A
    C = A * A.T  # 得到协方差矩阵
    D, V = np.linalg.eig(C)  # 求协方差矩阵的特征值和特征向量
    V_r = V[:, 0:r]  # 按列取前r个特征向量
    V_r = A.T * V_r  # 小矩阵特征向量向大矩阵特征向量过渡
    for i in range(r):
        V_r[:, i] = V_r[:, i] / np.linalg.norm(V_r[:, i])  # 特征向量归一化

    final_data = A * V_r
    return final_data, data_mean, V_r


##人脸识别
def face_rec(img, r):
    train_face, train_label = load_orl(10)  # 得到数据集
    # r = 100
    x_value = []
    y_value = []
    # 利用PCA算法进行训练
    data_train_new, data_mean, V_r = PCA(train_face, r)
    temp_face = img2vector(img)
    temp_face = temp_face - np.tile(data_mean, (temp_face.shape[0], 1))
    data_test_new = temp_face * V_r  # 得到测试脸在特征向量下的数据
    data_test_new = np.array(data_test_new)  # mat change to array
    data_train_new = np.array(data_train_new)

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
    print("第%d个人,第%d张照片" % (yuce_people, photo_num))
    return

if __name__ == '__main__':
    image_path = orlpath + '/029' + '/0' + str(9) + '.png'
    face_rec(image_path, r=50)
