# created at 2018-05-11
# updated at 2018-05-14
# By TimeStamp
# cnblogs: http://www.cnblogs.com/AdaminXie

import dlib  # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2  # 图像处理的库OpenCv

# face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1(
    "C:/Users/axnb029/Anaconda3/Lib/site-packages/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat")


# 计算两个向量间的欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print(dist)

    if dist > 0.4:
        return "diff"
    else:
        return "same"


features_mean_default_person = [-0.069742083, 0.069641717, 0.007250059, 0.007856009, -0.023368419, -0.086927901,
                                -0.049494835, -0.171500595, 0.13304741, -0.112849606, 0.21683076, -0.04884465,
                                -0.169016312, -0.115150563, -0.018590111, 0.161953911, -0.162161766, -0.099222879,
                                -0.076867051, -0.032289989, 0.092634934, -0.011479428, 0.041992811, 0.01040706,
                                -0.121379138, -0.334998402, -0.100977303, -0.107880417, -0.013899203, -0.02141675,
                                -0.010237776, 0.029943274, -0.171180955, -0.075855811, 0.010253188, 0.042088367,
                                -0.017167717, -0.027471955, 0.223302267, 0.015053514, -0.16807275, 0.06161015,
                                0.025295158, 0.276133484, 0.18536664, 0.091184989, 0.039625088, -0.15451491,
                                0.142591171, -0.188058658, 0.049275831, 0.106618686, 0.064034981, 0.082064004,
                                0.010331722, -0.101241502, 0.075761763, 0.06341897, -0.168805932, 0.032351608,
                                0.11376028, -0.063292461, -0.055659348, -0.04518367, 0.293036186, 0.071684681,
                                -0.13616534, -0.078974134, 0.088623021, -0.125477107, -0.08936159, -0.002104238,
                                -0.159810477, -0.17756911, -0.342589686, 0.035649034, 0.440023979, 0.095499569,
                                -0.228993442, 0.011300093, -0.065893406, 0.020892012, 0.199648259, 0.14588378,
                                -0.002415635, -0.042107073, -0.064753455, 0.000844009, 0.223764946, -0.066844232,
                                -0.076482166, 0.210918158, 0.01573298, 0.0791432, -0.006100287, 0.023222945,
                                -0.065575469, 0.031521409, -0.099043659, 0.007068649, 0.092037124, 0.010388323,
                                0.010762968, 0.110102179, -0.157556958, 0.180023236, 0.008877103, 0.036078898,
                                0.074287135, 0.001731667, -0.10875979, -0.065427134, 0.098422027, -0.209416073,
                                0.217935888, 0.178958694, 0.046826462, 0.125045842, 0.071425682, 0.103636526,
                                -0.038726452, 0.067209723, -0.239664237, -0.034611995, 0.09584148, 0.031715665,
                                0.09829342, 0.012134465
                                ]

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    'C:/Users/axnb029/Anaconda3/Lib/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat')

# 创建cv2摄像头对象
cap = cv2.VideoCapture('http://172.16.11.245:8080/?action=stream')  # 获取树莓派的视频流
cap.set(3, 480)


# 返回单张图像的128D特征
def get_128d_features(img_gray):
    dets = detector(img_gray, 1)
    if len(dets) != 0:
        shape = predictor(img_gray, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
    return face_descriptor


while cap.isOpened():
    flag, im_rd = cap.read()
    kk = cv2.waitKey(1)  # 每帧数据延时1ms，延时为0读取的是静态帧
    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)  # 取灰度图
    dets = detector(img_gray, 0)  # 人脸数dets

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im_rd, "q: quit", (20, 400), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

    if len(dets) != 0:
        features_rd = get_128d_features(im_rd)
        compare = return_euclidean_distance(features_rd, features_mean_default_person)  # 将捕获到的人脸提取特征和内置特征进行比对

        # 让人名跟随在矩形框的下方
        # 确定人名的位置坐标
        pos_text_1 = tuple([dets[0].left(), int(dets[0].bottom() + (dets[0].bottom() - dets[0].top()) / 4)])
        im_rd = cv2.putText(im_rd, compare.replace("same", "Wangkang"), pos_text_1, font, 0.8, (0, 255, 0), 1,
                            cv2.LINE_AA)
        for k, d in enumerate(dets):
            # 绘制矩形框
            im_rd = cv2.rectangle(im_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 0), 2)
        cv2.putText(im_rd, "faces: " + str(len(dets)), (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    else:
        cv2.putText(im_rd, "no face", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)  # 没有检测到人脸

    if kk == ord('q'):  # 按下q键退出
        break

    # 窗口显示
    cv2.imshow("camera", im_rd)

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()
