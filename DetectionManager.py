# encoding=utf-8
import hiai
from hiai.nn_tensor_lib import DataType
import cv2 as cv
import threading
import yolov3manager
import graph
import dvpp_process.dvpp_process as dp
import numpy as np
import os
from presenter_types import *


class DetectionManager(object):
    def __init__(self, model_width=640, model_height=352):
        self.model_width = model_width
        self.model_height = model_height
        self.graph_detection = graph.Graph('./model/detection.om')
        self.yolov3 = yolov3manager.yolov3manager()

    def inference_from_file_src(self, filename):
        img = cv.imread(filename)
        src_img = img
        img = cv.resize(img, (self.model_width, self.model_height))
        result = {}
        result = self.graph_detection.Inference(img)
        if result is None:
            print('Inference Result is No:{}'.format(filename))
            return None
        detection_list = self.yolov3.Yolov3_post_process_customize(result, img.shape[1], img.shape[0], self.model_width,
                                                                   self.model_height)
        detection_list = self.set_fact_boxinfo(detection_list, src_img.shape[1], src_img.shape[0])
        return src_img, detection_list

    def inference_from_file_resize(self, filename):
        img = cv.imread(filename)
        img = cv.resize(img, (self.model_width, self.model_height))
        result = {}
        result = self.graph_detection.Inference(img)
        if result is None:
            print('Inference Result is No:{}'.format(filename))
            return None
        detection_list = self.yolov3.Yolov3_post_process_customize(result, img.shape[1], img.shape[0], self.model_width,
                                                                   self.model_height)
        return img, detection_list
    # 从原图里面挖取头像
    def set_fact_boxinfo(self, detection_list,image_width, image_height):
        for idx,obj in enumerate(detection_list):
            witdh_radio = image_width/self.model_width
            height_radio = image_height / self.model_height
            detection_list[idx].lt.x = int(witdh_radio*obj.lt.x)
            detection_list[idx].lt.y = int(height_radio * obj.lt.y)
            detection_list[idx].rb.x = int(witdh_radio * obj.rb.x)
            detection_list[idx].rb.y = int(height_radio * obj.rb.y)
        return detection_list


    def get_detectioninfo(self, src_img, detection_list, confidence=0):
        detection_info = DetectionInfo()
        detection_info.src_image = src_img
        detection_info.detection_list = detection_list
        for detection in detection_list:
            if detection.confidence >= confidence:
                crop_image = src_img[detection.lt.y:detection.rb.y, detection.lt.x:detection.rb.x]
                detection_info.image_list.append(crop_image)
            else:
                detection_info.image_list.append(None)
        return detection_info
