import ClassifyManager
import DetectionManager
import os
import cv2

def main():
    detction = DetectionManager.DetectionManager()
    classify = ClassifyManager.ClassifyManager()
    images_path = '/home/HwHiAiUser/img/'
    count = 1
    for filename in os.listdir(images_path):
        img, detection_list = detction.inference_from_file_resize(images_path + filename)
        print('detection_list len:',len(detection_list))
        detection_infos = detction.get_detectioninfo(img, detection_list)
        for idx,image in enumerate(detection_infos.image_list):
            if image is None:
                print('image is None')
                continue
            else:
                #cv2.imwrite("/home/HwHiAiUser/result/"+str(count)+".jpg", image)
                count += 1
                #classify = ClassifyManager.ClassifyManager()
                result = classify.inference_from_image(image)
                if result is None:
                    print(str(idx)+'result is None')
                if result:
                    class_name = classify.get_class_name(result)
                    print('inference result is:', class_name)
        break


if __name__ == '__main__':
    main()
