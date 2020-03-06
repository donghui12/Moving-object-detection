#!/usr/bin/env python
# encoding: utf-8

import cv2
import numpy as np
import os

from PIL import Image, ImageStat, ImageGrab
import math
import time
import copy

import sys

"""
    轨迹分类
"""

# from numpy import *
import numpy as np


def circle_point(circle_path):
    with open(circle_path, 'r') as f:
        datasets = np.array([[float(data.replace("\n", '').split(', ')[0]), float(data.replace("\n", '').split(', ')[0])] for data in f.readlines()])
    labels = []
    for i in range(len(datasets)):
        if i <= 1668:
            labels.append("circle")
        else:
            labels.append("foot")
    # labels = ["circle" for i in range(len(datasets))]
    return datasets, labels


def load_test_file(test_path):
    with open(test_path, 'r') as f:
        datasets = np.array([[float(data.replace("\n", '').split(', ')[0]), float(data.replace("\n", '').split(', ')[0])] for data in f.readlines()])
    return datasets

# createDataSet()
def kNN_Classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 关于tile函数的用法
    # >>> b=[1,3,5]
    # >>> tile(b,[2,3])
    # array([[1, 3, 5, 1, 3, 5, 1, 3, 5],
    #       [1, 3, 5, 1, 3, 5, 1, 3, 5]])
    sqDiffMat = diffMat ** 2
    sqDistances = np.sum(sqDiffMat, axis=1)
    distances = sqDistances ** 0.5  # 算距离
    sortedDistIndicies = np.argsort(distances)
    # 关于argsort函数的用法
    # argsort函数返回的是数组值从小到大的索引值
    # >>> x = np.array([3, 1, 2])
    # >>> np.argsort(x)
    # array([1, 2, 0])
    classCount = {}  # 定义一个字典
    #   选择k个最近邻
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        #                                                     计算k个最近邻中各类别出现的次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    #                                                         返回出现次数最多的类别标签
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex

def predicte(point):
    point_array = np.array(point)
    datasets, labels = circle_point("local_3.txt")
    outputLabel = kNN_Classify(point_array, datasets, labels, 5)
    return outputLabel


class Vision:
    def __init__(self):
        self.master = []
        self.MAX_COUNT = 3
        self.min_counter_size = 500
        self.max_counter_size = 3000
        self.MAX_SUM = 10000  # sum area size
        self.MAX_DIS = 200   # distance between two center
        self.last_x = -1
        self.last_y = -1
        self.dis = 0
        self.dis_x = 0  # fabs(last_x - mx)
        self.dis_y = 0  # fabs(last_y - my)
        self.last_time = 0
        self.averange_point = (-1, -1)  # The average coordinate value of ten points
        self.ten_point = []
        self.counter = 0
        self.min_occupy = 0.6
        self.len_count = 20
        self.object_count = 0
        self.last_time_ = 0
        cam_width = 360
        cam_length = 240
        cam_no = 1
        self.cam = cv2.VideoCapture(cam_no)
        self.cam.set(3, cam_width)  # 1280
        self.cam.set(4, cam_length)  # 720

    def get_frame(self):
        try:
            flag = time.time() - Vision.get_frame.last
        except AttributeError as e:
            flag = time.time()
        if flag > 0.2:
            Vision.get_frame.last = time.time()
            Vision.get_frame.img = cv2.flip(self.cam.read()[1], 1)
        return Vision.get_frame.img

    def get_moving_distance_x_y(self, input_c1, input_c2):
        x_1 = input_c1[0]
        y_1 = input_c1[1]
        x_2 = input_c2[0]
        y_2 = input_c2[1]
        return abs(x_2-x_1), abs(y_2-y_1)

    def get_averange_point(self):
        if len(self.ten_point) < 1:
            return (-1,-1)

        sum_x = 0
        sum_y = 0
        for point in self.ten_point:
            sum_x += point[0]
            sum_y += point[1]
        return (sum_x/len(self.ten_point), sum_y/len(self.ten_point))

    # Moving object: return moving object coords
    def get_moving_object(self):
        frame0 = cv2.flip(self.cam.read()[1], 1)
        frame1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.GaussianBlur(frame1, (25, 25), 0)   # 15 15
        if len(self.master) is 0:
            self.master = copy.deepcopy(frame2)
        frame3 = cv2.absdiff(self.master, frame2)
        frame4 = cv2.threshold(frame3, 15, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((2, 2), np.uint8)
        frame5 = cv2.erode(frame4, kernel, iterations=4)
        frame5 = cv2.dilate(frame5, kernel, iterations=8)
        # find contours on thresholded image
        nada, contours, nada = cv2.findContours(
            frame5.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # make coutour frame
        frame7 = frame0.copy()

        # target contours
        targets = []
        len_c = len(contours)
        if len_c < self.MAX_COUNT:
            # loop over the contours
            for c in contours:
                # if the contour is too small, ignore it
                if self.max_counter_size > cv2.contourArea(c) > self.min_counter_size:
                    # contour data
                    M = cv2.moments(c)  # ;print( M )
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    x, y, w, h = cv2.boundingRect(c)
                    rx = x + int(w / 2)
                    ry = y + int(h / 2)
                    ca = cv2.contourArea(c)

                    targets.append((cx, cy, ca))
            # make target
            mx = 0
            my = 0
            if targets:
                area = 0
                area_sum = 0
                for x, y, a in targets:
                    if a > area:
                        mx = x
                        my = y
                        now_time = time.time()
                        area = a
                        area_sum += a
            # print(mx, my)  # 这个是运动的坐标点
            self.averange_point = self.get_averange_point()
            if self.averange_point[0] != -1:
                self.dis_x, self.dis_y = self.get_moving_distance_x_y(
                    [mx, my], self.averange_point)

            if len(targets):
                if self.last_time == 0:
                    self.last_time = now_time
                    self.last_x = mx
                    self.last_y = my
                    self.ten_point.append((mx, my))
                elif area_sum < self.MAX_SUM:
                    if 0 < self.dis_y < 40 and 0 < self.dis_x < 40 and 50 < mx < 280 and 40 < my < 180\
                            and (float(str(now_time))-float(str(self.last_time))) < 0.2 and predicte((float(mx), float(my)))=="circle":
                        cv2.circle(frame7, (mx, my), 30, (0, 255, 0, 0), 2)
                        cv2.circle(frame7, (mx, my), 2, (0, 255, 0), 2)

                        self.last_x = mx
                        self.last_y = my
                        self.last_time = time.time()

                        self.ten_point.append((mx, my))
                        if len(self.ten_point) > 10:
                            self.ten_point = self.ten_point[-10:]

                        self.object_count += 1
                else:
                    self.object_count = 0
                    self.counter = 0

                if abs(self.last_time - now_time) > 0.5:
                    self.last_time = 0
                    self.ten_point = []
                    self.dis_x = -1
                    self.dis_y = -1

        # update master
        self.master = frame2

        # Verify
        self.counter += 1
        if self.counter > self.len_count:
            obj_occu = self.object_count / self.len_count

            self.object_count = 0
            self.counter = 0

            if obj_occu > self.min_occupy:
                _x = copy.deepcopy(self.last_x)
                _y = copy.deepcopy(self.last_y)
                obj_coords = [float(_x), float(_y)]

        cv2.imshow("img5", frame5)
        cv2.imshow("img7", frame7)

        # return None


if __name__ == "__main__":
    a = Vision()
    while True:
        cv2.imshow("img", a.get_frame())
        result = a.get_moving_object()
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
