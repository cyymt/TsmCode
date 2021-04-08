../caffe_landmark/build/tools/caffe train \
    --solver=solver_l2.prototxt \
    --weights=./base_model/landmark25_mask_v1.2_iter_96000.caffemodel \
    --gpu=4,5,6,7 2>&1 | tee ./logs/landmarkIrGrayColormaskRote_20210310.log
# v2:
#--weights=../newdata_resnet18-1st-half-0.25k3conv_nir_visfake_crop2len_224_focal_6th_retrain/models/saved_AFnet_nir_iter_319930.caffemodel \
#--snapshot=./models/AFnet_nir_iter_40365.solverstate \
#--snapshot=./models/AFnet_nir_iter_82225.solverstate \
#--snapshot=./models/AFnet_nir_iter_101660.solverstate \
