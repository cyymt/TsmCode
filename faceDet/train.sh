#!/usr/bin/env sh
../caffe_ssd/build/tools/caffe train \
    --solver=solver.prototxt \
    --weights=./base_model/mask_v13_iter_70000_loss2.867.caffemodel \
    --gpu=0,1,2,3,4,5,6,7 2>&1 | tee ./logs/maskDet_v1.4.log

# v2
#--weights=./base_model/mask_iter_88000.caffemodel \
#--snapshot=./models/mask_v12_iter_21500.solverstate \
