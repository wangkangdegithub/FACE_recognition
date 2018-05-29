# created at 2018-05-11
# updated at 2018-05-14
# By TimeStamp
# cnblogs: http://www.cnblogs.com/AdaminXie

#   return_128d_features()          获取某张图像的128d特征
#   write_into_csv()                将某个文件夹中的图像读取特征兵写入csv
#   compute_the_mean()              从csv中读取128d特征，并计算特征均值

import cv2
import os
import dlib
from skimage import io
import csv
import numpy as np
import pandas as pd

path_pics = "C:/pythonProject/Dlib_face_recognition_from_camera-master/data/get_from_camera/"
path_csv = "C:/pythonProject/Dlib_face_recognition_from_camera-master/data/csvs/"

# detector to find the faces
detector = dlib.get_frontal_face_detector()

# shape predictor to find the face landmarks
predictor = dlib.shape_predictor("C:/Users/axnb029/Anaconda3/Lib/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat")

# face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1("C:/Users/axnb029/Anaconda3/Lib/site-packages/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat")


# 返回单张图像的128D特征
def return_128d_features(path_img):
    img = io.imread(path_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img_gray, 1)

    print("检测的人脸图像：", path_img, "\n")


    if len(dets) != 0:
        shape = predictor(img_gray, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)  # 计算人脸图像的128维特征点向量
    else:
        face_descriptor = 0
        print("no face")

    # print(face_descriptor)
    return face_descriptor



def write_into_csv(path_pics, path_csv):
    dir_pics = os.listdir(path_pics)

    with open(path_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(dir_pics)):
            # 调用return_128d_features()得到128d特征
            print("正在读的人脸图像：", path_pics + dir_pics[i])
            features_128d = return_128d_features(path_pics + dir_pics[i])
            #  print(features_128d)
            # 遇到没有检测出人脸的图片跳过
            if features_128d == 0:
                i += 1
            else:
                writer.writerow(features_128d)


write_into_csv(path_pics, path_csv + "default_person.csv")

path_csv_rd = "C:/pythonProject/Dlib_face_recognition_from_camera-master/data/csvs/wangkang.csv"


# 从csv中读取数据，计算128d特征的均值
def compute_the_mean(path_csv_rd):
    column_names = []

    # 128列特征
    for i in range(128):
        column_names.append("features_" + str(i + 1))

    # 利用pandas读取csv
    rd = pd.read_csv(path_csv_rd, names=column_names)

    # 存放128维特征的均值
    feature_mean = []

    for i in range(128):
        tmp_arr = rd["features_" + str(i + 1)]
        tmp_arr = np.array(tmp_arr)

        # 计算某一个特征的均值
        tmp_mean = np.mean(tmp_arr)

        feature_mean.append(tmp_mean)

    print(feature_mean)
    return feature_mean

# compute_the_mean(path_csv_rd)
