import hiai
from hiai.nn_tensor_lib import DataType
import cv2
import threading
import graph


class ClassifyManager(object):
    def __init__(self, model_width=224, model_height=224):
        self.model_width=model_width
        self.model_height=model_height
        self.graph_classify =  graph.Graph('./model/classify.om')
        self.class_name_dic={0:"head",1:"nohead",2:"cloth",3:"nocloth",4:"legs",5:"unkonwn"}
    def inference_from_file(self,filename):
        img = cv2.imread(filename)
        if img is None:
            return None
        # img = cv2.cvtColor(img, cv.COLOR_YUV2RGB_I420)
        img = cv2.resize(img, (self.model_width, self.model_height))
        result =  self.graph_classify.Inference(img)
        # print('result is :',result)
        return  result
    def inference_from_image(self,img):
        # img = cv2.cvtColor(img, cv.COLOR_YUV2RGB_I420)
        if img is None:
            print('image is None')
            return None
        img = cv2.resize(img, (self.model_width, self.model_height))
        result =  self.graph_classify.Inference(img)
        print('inference result is =',result)
        return result
    def get_class_name(self,result_list):
        if result_list is None or len(result_list)==0:
            return self.class_name_dic[5]
        max_index = self.find_max_index(result_list)
        return self.class_name_dic[max_index]

    def find_max_index(self,result_list):
        max = result_list[0][0][0][0][0]
        idx=0
        for index in range(5):
            #print(result_list[0][0][0][0][index])
            if max < result_list[0][0][0][0][index]:
                max = result_list[0][0][0][0][index]
                idx = index
        return idx







