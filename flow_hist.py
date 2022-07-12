#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import rospy
from sensor_msgs.msg import Image
import cv2
import cv_bridge
import numpy as np


class GetTrainData:
    def __init__(self):
        # 光流相关参数
        self.color = (0, 255, 0)
        self.feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=5, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.bridge = cv_bridge.CvBridge()
        self.initialize_flag = False
        self.image_sub = rospy.Subscriber('/image_changed', Image, self.image_callback)

    def hist(self, num, l):
        if num >= -180 and num < -150:
            l[0] += 1
        elif num >= -150 and num < -120:
            l[1] += 1
        elif num >= -120 and num < -90:
            l[2] += 1
        elif num >= -90 and num < -60:
            l[3] += 1
        elif num >= -60 and num < -30:
            l[4] += 1
        elif num >= -30 and num < 0:
            l[5] += 1
        elif num >= 0 and num < 30:
            l[6] += 1
        elif num >= 30 and num < 60:
            l[7] += 1
        elif num >= 60 and num < 90:
            l[8] += 1
        elif num >= 90 and num < 120:
            l[9] += 1
        elif num >= 120 and num < 150:
            l[10] += 1
        else:
            l[11] += 1
        return l

    def encoder(self, zone1, zone2, zone3, zone4, z1, z2, z3, z4):
        a = [i / z1 for i in zone1] + [i / z2 for i in zone2] + [i / z3 for i in zone3] + [i / z4 for i in zone4]
        return [0 if np.isnan(a[n]) == True else a[n] for n in range(len(a))]

    def angular(self, p):
        stander = [10, 0]
        r = np.dot(p, stander) / (np.linalg.norm(p) * (np.linalg.norm(stander)))
        theta = np.rad2deg(np.arccos(r))
        if p[1] < stander[1]:
            return -int(theta)
        return int(theta)

    def image_callback(self, msg):
        new_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')[24:424, 248:648]

        if self.initialize_flag == False:  # 跳过第一帧
            self.prev_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            self.prev = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            self.mask = np.zeros_like(new_frame)
            self.initialize_flag = True
        else:  # 从第二帧开始光流追踪
            gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            FeaturePointNum, _, _ = self.prev.shape

            if FeaturePointNum <= 10:  # 如果少于10个特征点则进行补充
                self.p = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
                self.prev = np.vstack((self.prev, self.p))
            nex, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev, None, **self.lk_params)
            good_old = self.prev[status == 1]
            good_new = nex[status == 1]

            # 开始编码
            z1, z2, z3, z4 = 0, 0, 0, 0
            zone1, zone2, zone3, zone4 = np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv2.line(self.mask, (a, b), (c, d), self.color, 2)
                frame = cv2.circle(new_frame, (a, b), 3, self.color, -1)

                x_vol = b - d  # 图片竖直方向坐标差
                y_vol = a - c  # 图片水平方向坐标差
                val = [y_vol, x_vol]
                angle = self.angular(val)

                # b:竖直方向坐标
                if b <= 200:
                    if a <= 200:
                        zone1 = self.hist(angle, zone1)
                        z1 += 1
                    else:
                        zone2 = self.hist(angle, zone2)
                        z2 += 1
                else:
                    if a <= 200:
                        zone3 = self.hist(angle, zone3)
                        z3 += 1
                    else:
                        zone4 = self.hist(angle, zone4)
                        z4 += 1
            res = self.encoder(zone1, zone2, zone3, zone4, z1, z2, z3, z4)

            # 保存训练数据
            num = '\n' + str(res)
            f = open('./Left.txt', 'a')
            f.write(num)
            f.close()

            output = cv2.add(frame, self.mask)
            self.mask = np.zeros_like(frame)
            self.prev_gray = gray.copy()
            self.prev = good_new.reshape(-1, 1, 2)

            # 显示图像
            cv2.imshow('Optical Flow', output)
            cv2.waitKey(3)


if __name__ == '__main__':
    rospy.init_node('TrainData')
    GTD = GetTrainData()
    rospy.spin()
