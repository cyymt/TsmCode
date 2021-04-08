#!/usr/bin/env sh
../../caffe-augmentation/build/tools/caffe train \
    --solver=solver.prototxt \
    --weights=./model_v18/face2d_wudi_iter_200000.caffemodel \
    --gpu=2,3,4,5,6,7 2>&1 | tee ./logs/face2d_20210407.log

# v2:
#--weights=../newdata_resnet18-1st-half-0.25k3conv_nir_visfake_crop2len_224_focal_6th_retrain/models/saved_AFnet_nir_iter_319930.caffemodel \
#--snapshot=./models/AFnet_nir_iter_40365.solverstate \
#--snapshot=./models/AFnet_nir_iter_82225.solverstate \
#--snapshot=./models/AFnet_nir_iter_101660.solverstate \
