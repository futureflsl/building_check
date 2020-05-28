#enconding=utf-8
import numpy as np
import copy
import time
from presenter_types import *

class yolov3manager(object):
    def __init__(self):
        self.class_names = ["person", "face", "mask"]
        self.class_num = 3
        self.stride_list = [8, 16, 32]
        self.anchors_1 = np.array([[10, 13], [16, 30], [33, 23]]) / self.stride_list[0]
        self.anchors_2 = np.array([[30, 61], [62, 45], [59, 119]]) / self.stride_list[1]
        self.anchors_3 = np.array([[116, 90], [156, 198], [163, 326]]) / self.stride_list[2]
        self.anchor_list = [self.anchors_1, self.anchors_2, self.anchors_3]

        self.conf_threshold = 0.3
        self.iou_threshold = 0.4
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    
    def sigmoid(self,x):
        s = 1 / (1 + np.exp(-1*x))
        return s

    def getMaxClassScore(self, class_scores):
        class_score = 0
        class_index = 0
        for i in range(len(class_scores)):
            if class_scores[i] > class_score:
                class_index = i+1
                class_score = class_scores[i]
        return class_score,class_index

    def getBBox(self, feat, anchors, image_shape, confidence_threshold):
        box = []
        for i in range(len(anchors)):
            for cx in range(feat.shape[0]):
                for cy in range(feat.shape[1]):
                    tx = feat[cx][cy][0 + 85 * i]
                    ty = feat[cx][cy][1 + 85 * i]
                    tw = feat[cx][cy][2 + 85 * i]
                    th = feat[cx][cy][3 + 85 * i]
                    cf = feat[cx][cy][4 + 85 * i]
                    cp = feat[cx][cy][5 + 85 * i:85 + 85 * i]

                    bx = (sigmoid(tx) + cx)/feat.shape[0]
                    by = (sigmoid(ty) + cy)/feat.shape[1]
                    bw = anchors[i][0]*np.exp(tw)/image_shape[0]
                    bh = anchors[i][1]*np.exp(th)/image_shape[1]

                    b_confidence = self.sigmoid(cf)
                    b_class_prob = self.sigmoid(cp)
                    b_scores = b_confidence*b_class_prob
                    b_class_score,b_class_index = self.getMaxClassScore(b_scores)

                    if b_class_score > confidence_threshold:
                        box.append([bx,by,bw,bh,b_class_score,b_class_index])
        return box

    def donms(self, boxes,nms_threshold):
        b_x = boxes[:, 0]
        b_y = boxes[:, 1]
        b_w = boxes[:, 2]
        b_h = boxes[:, 3]
        scores = boxes[:,4]
        areas = (b_w+1)*(b_h+1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(b_x[i], b_x[order[1:]])
            yy1 = np.maximum(b_y[i], b_y[order[1:]])
            xx2 = np.minimum(b_x[i] + b_w[i], b_x[order[1:]] + b_w[order[1:]])
            yy2 = np.minimum(b_y[i] + b_h[i], b_y[order[1:]] + b_h[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            IoU = inter / union
            inds = np.where(IoU <= nms_threshold)[0]
            order = order[inds + 1]
        final_boxes = [boxes[i] for i in keep]
        return final_boxes

    def getBoxes(self, resultList, anchors, img_shape, confidence_threshold, nms_threshold):
        boxes = []
        for i in range(resultList):
            feature_map = resultList[i][0].transpose((2, 1, 0))
            box = self. getBBox(feature_map, anchors[i], img_shape, confidence_threshold)
            boxes.extend(box)
        Boxes = donms(np.array(boxes),nms_threshold)
        return Boxes
    def overlap(self, x1, x2, x3, x4):
        left = max(x1, x3)
        right = min(x2, x4)
        return right - left

    def cal_iou(self,box, truth):
        w = self.overlap(box[0], box[2], truth[0], truth[2])
        h = self.overlap(box[1], box[3], truth[1], truth[3])
        if w <= 0 or h <= 0:
            return 0
        inter_area = w * h
        union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
        return inter_area * 1.0 / union_area

    def apply_nms(self, all_boxes, thres):
        res = []

        for cls in range(self.class_num):
            cls_bboxes = all_boxes[cls]
            sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]

            p = dict()
            for i in range(len(sorted_boxes)):
                if i in p:
                    continue

                truth = sorted_boxes[i]
                for j in range(i + 1, len(sorted_boxes)):
                    if j in p:
                        continue
                    box = sorted_boxes[j]
                    iou = self.cal_iou(box, truth)
                    if iou >= thres:
                        p[j] = 1

                for i in range(len(sorted_boxes)):
                    if i not in p:
                        res.append(sorted_boxes[i])
        return res

    def decode_bbox(self, conv_output, anchors, img_w, img_h):
        _, h, w = conv_output.shape
        pred = conv_output.transpose((1, 2, 0)).reshape((h * w, 3, 5 + self.class_num))

        pred[..., 4:] = self.sigmoid(pred[..., 4:])
        pred[..., 0] = (self.sigmoid(pred[..., 0]) + np.tile(range(w), (3, h)).transpose((1, 0))) / w
        pred[..., 1] = (self.sigmoid(pred[..., 1]) + np.tile(np.repeat(range(h), w), (3, 1)).transpose((1, 0))) / h
        pred[..., 2] = np.exp(pred[..., 2]) * anchors[:, 0:1].transpose((1, 0)) / w
        pred[..., 3] = np.exp(pred[..., 3]) * anchors[:, 1:2].transpose((1, 0)) / h

        bbox = np.zeros((h * w, 3, 4))
        bbox[..., 0] = np.maximum((pred[..., 0] - pred[..., 2] / 2.0) * img_w, 0)  # x_min
        bbox[..., 1] = np.maximum((pred[..., 1] - pred[..., 3] / 2.0) * img_h, 0)  # y_min
        bbox[..., 2] = np.minimum((pred[..., 0] + pred[..., 2] / 2.0) * img_w, img_w)  # x_max
        bbox[..., 3] = np.minimum((pred[..., 1] + pred[..., 3] / 2.0) * img_h, img_h)  # y_max

        pred[..., :4] = bbox
        pred = pred.reshape((-1, 5 + self.class_num))
        pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)
        pred = pred[pred[:, 4] >= self.conf_threshold]
        pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)

        all_boxes = [[] for ix in range(self.class_num)]
        for ix in range(pred.shape[0]):
            box = [int(pred[ix, iy]) for iy in range(4)]
            box.append(int(pred[ix, 5]))
            box.append(pred[ix, 4])
            all_boxes[box[4] - 1].append(box)

        return all_boxes

    def get_result(self, model_outputs, img_w, img_h, model_w, model_h):
        num_channel = 3 * (self.class_num + 5)
	#print('num_channel is:',num_channel)
        all_boxes = [[] for ix in range(self.class_num)]
        #print("all_boxes",all_boxes)
        for ix in range(3):
            pred = model_outputs[2 - ix].reshape((num_channel, model_h // self.stride_list[ix], model_w // self.stride_list[ix]))
            #print('pred is:',pred)
            anchors = self.anchor_list[ix]
            boxes = self.decode_bbox(pred, anchors, img_w, img_h)
            all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(self.class_num)]
            #print('all boxes is:',all_boxes)

        res = self.apply_nms(all_boxes, self.iou_threshold)

        return res

    def Yolov3_post_process_customize(self, resultList, img_w, img_h, model_w, model_h):
        result_return = dict()
        res = self.get_result(resultList, img_w, img_h, model_w, model_h,)
        if not res:
            print("not res")
            result_return['detection_classes'] = []
            result_return['detection_boxes'] = []
            result_return['detection_scores'] = []
            #return result_return
        else:
            new_res = np.array(res)
            picked_boxes = new_res[:, 0:4]
            picked_boxes = picked_boxes[:, [1, 0, 3, 2]]
            picked_classes = self.convert_labels(new_res[:, 4])
            picked_score = new_res[:, 5]
            result_return['detection_classes'] = picked_classes
            result_return['detection_boxes'] = picked_boxes.tolist()
            result_return['detection_scores'] = picked_score.tolist()
        
        detection_result_list = []
        for i in range(len(result_return['detection_classes'])):
            #item = result[i, 0, 0, ]
            #if item[2] < confidence_threshold:
            #    continue
            box = result_return['detection_boxes'][i]
            detection_item = ObjectDetectionResult()
            detection_item.attr = result_return['detection_classes'][i]
            print("detection_item.attr",detection_item.attr)
            detection_item.confidence = result_return['detection_scores'][i]
            detection_item.lt.x = int(box[1])
            detection_item.lt.y = int(box[0])
            detection_item.rb.x = int(box[3])
            detection_item.rb.y = int(box[2])
            #print("*****---------******:",detection_item.lt.x,detection_item.lt.y,detection_item.rb.x,detection_item.rb.y)
            if self.class_names == []:
                detection_item.result_text = str(detection_item.attr) + " " + str(round(detection_item.confidence*100,2)) + "%"
            else:
                detection_item.result_text = str(self.class_names[detection_item.attr]) + " " + str(round(detection_item.confidence*100,2)) + "%"
            detection_result_list.append(detection_item)
        return detection_result_list
    
    def convert_labels(self, label_list):
        """
            class_names = ['person', 'face']
            :param label_list: [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]
            :return: 
            """
        if isinstance(label_list, np.ndarray):
            label_list = label_list.tolist()
            label_names = [ int(index) for index in label_list]
            #print("label_names",label_names)
        return label_names
