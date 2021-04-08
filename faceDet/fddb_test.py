import numpy as np
import cv2,os,sys
import datetime
import argparse
import pdb
from tqdm import tqdm
import xml.etree.ElementTree as ET
sys.path.insert(0, "/home/chenyuyang/software/caffe-quant/python")
import caffe

threshold_value=0.5
detect_thresh = 0.5
#the function of calculate IOU
def IOU (box,boxes):
    box_area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
    area=(boxes[:,2]-boxes[:,0]+1)*(boxes[:,3]-boxes[:,1]+1)
    xx1=np.maximum(box[0],boxes[:,0])
    yy1=np.maximum(box[1],boxes[:,1])
    xx2=np.minimum(box[2],boxes[:,2])
    yy2=np.minimum(box[3],boxes[:,3])
    
    w=np.maximum(0,xx2-xx1+1)
    h=np.maximum(0,yy2-yy1+1)
    inter_area=w*h
    iou=inter_area/(box_area+area-inter_area)
    return iou

def GetAnnotBoxLoc(AnotPath):#AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  #打开文件，解析成一棵树型结构
    root = tree.getroot()#获取树型结构的根
    ObjectSet=root.findall('object')#找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet=[] #以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        BndBox=Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)#-1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)#-1
        x2 = int(BndBox.find('xmax').text)#-1
        y2 = int(BndBox.find('ymax').text)#-1
        ObjBndBoxSet.append([x1,y1,x2,y2])
    return ObjBndBoxSet

def draw_rect(draw_image,pred_bboxes,time=1000):
    for box in pred_bboxes:
        cv2.rectangle(draw_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0),2)
    cv2.imshow("win",draw_image)
    cv2.waitKey(time)

def fddb_test(model_def, model_weights,base_dir,test_path):
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    total_nums = 0
    count = 0
    for line in tqdm(open(test_path,'r')):
        temp_img_path,temp_xml_path = line.strip().split(" ")
        Image_Path = os.path.join(base_dir,temp_img_path)
        Xml_Path = os.path.join(base_dir,temp_xml_path)
        true_bboxes = GetAnnotBoxLoc(Xml_Path)
        total_nums += len(true_bboxes)
        # image = caffe.io.load_image(Image_Path,color=True)
        image = cv2.imread(Image_Path)
        draw_image = image.copy()
        im_scale = float(320) / np.max(draw_image.shape[:2])
        if im_scale != 1:
            image = cv2.resize(image, None, None, fx=im_scale,
                             fy=im_scale, interpolation=cv2.INTER_LINEAR)
        #image = cv2.resize(image,(320,180))
        #im_scalex = float(320) / float(draw_image.shape[1])
        #im_scaley = float(180) / float(draw_image.shape[0])
        data = image.transpose((2, 0, 1))
        data = (data - 127.5) * 0.0078125
        net.blobs["data"].reshape(1, 3, data.shape[1], data.shape[2])
        net.blobs['data'].data[...] = data

        detections = net.forward()['detection_out']
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]
        pred_bboxes = []
        # import pdb;pdb.set_trace()
        for i in range(det_conf.shape[0]):
            if det_conf[i] < detect_thresh:
                continue
            xmin = det_xmin[i] * image.shape[1] / im_scale
            ymin = det_ymin[i] * image.shape[0] / im_scale
            xmax = det_xmax[i] * image.shape[1] / im_scale
            ymax = det_ymax[i] * image.shape[0] / im_scale
            pred_bboxes.append([xmin,ymin,xmax,ymax])
        true_bboxes = np.array(true_bboxes,dtype=np.float32)
        pred_bboxes = np.array(pred_bboxes,dtype=np.float32)
        # draw_rect(draw_image,pred_bboxes,time=1000)
        for pred_box in pred_bboxes:
            # import pdb;pdb.set_trace()
            iou = IOU(pred_box,true_bboxes)
            if np.max(iou) >= threshold_value:
                count += 1
    print(f"acc:{(count/total_nums)*100:.2f}%")  
if __name__=='__main__':
    caffe.set_mode_gpu()
    caffe.set_device(0)
    model_def = "/home/chenyuyang/Code/faceBox_oneClass/train_320/deploy.prototxt"
    model_weights = "/home/chenyuyang/Code/faceBox_oneClass/train_320/models/mask_v11_iter_23000.caffemodel"
    base_dir = "/home/chenyuyang/Code/faceBox_oneClass/data/brainwash-other/"
    test_path = "/home/chenyuyang/Code/faceBox_oneClass/data/brainwash-other/ImageSets/test.txt"
    fddb_test(model_def, model_weights, base_dir,test_path)
