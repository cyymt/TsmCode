#coding=utf-8
"""
Created on 12th, Mar, 2019
Detect the face and crop image
@author: Luyujing
"""
import numpy as np
import os,sys
caffe_root = '../caffe_landmark/python'
sys.path.insert(0, caffe_root)
import caffe
import cv2

if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(0)

    #---------------------------------------- model -------------------------------------------------------
    weights = "models_ir_color/landmark25_mask_v1.1_iter_150000.caffemodel"
    deploy = "deploy.prototxt"

    net = caffe.Net(deploy, weights, caffe.TEST)
    root_dir = "/home/chenyuyang/Code/3DDFA/landmar25_train/train_landmark25/data/trainLandmark25_Graydata96/"
    txt_path = "/home/chenyuyang/Code/3DDFA/landmar25_train/train_landmark25/data/trainLandmark25_Graydata96/train_landmark25_clear.txt"
    for line in open(txt_path,"r"):
        line  = line.strip().split(" ")
        points_src = (np.array(list(map(float,line[1:51])))*96).astype(np.int)
        image_path = os.path.join(root_dir,line[0])
        image = cv2.imread(image_path)
        data = ((image - 127.5) / 128.0).transpose((2,0,1))
        net.blobs['data'].data[0] = data
        out = net.forward()
        points = (np.squeeze(out['fc_para'])*96).astype(np.int)
        for index in range(25):
            cv2.circle(image,(points[index],points[index+25]),3,(0,0,255),-1)
        cv2.imshow("win",image)
        cv2.waitKey(1000)
        #import pdb; pdb.set_trace()
