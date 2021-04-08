#ifdef USE_CUDNN
#include "caffe/filler.hpp"
#include "caffe/layers/binary_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define sign(x) ((x)>=0?1:-1)
//__global__ void binary_sync_conv_groups() { }

template <typename Dtype>
__global__ void BinaryGPU_binarize(const int n, const int num, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype sum = 0;
    for (int coor = 0; coor < num; coor++) {
      sum += std::abs(in[index * num + coor]) / Dtype(num);
    }
    for (int coor = 0; coor < num; coor++) {
      out[index * num + coor] = sign(in[index * num + coor]) * sum;
    }
  }  
}


template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::binarizeGPUTo(const shared_ptr<Blob<Dtype> > weights, const shared_ptr<Blob<Dtype> > wb) {
  const int count = weights->num();
  const int div = weights->count() / count;
  BinaryGPU_binarize<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, div, weights->gpu_data(), wb->mutable_gpu_data() );
}


template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    binarizeGPUTo(this->blobs_[0], W_b);
  copyGPUTo(this->blobs_[0], W_buffer);
  copyGPUTo(W_b, this->blobs_[0]);
  
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
  copyGPUTo(W_buffer, this->blobs_[0]);
}
template void BinaryInnerProductLayer<float>::binarizeGPUTo(const shared_ptr<Blob<float> > weights, const shared_ptr<Blob<float> > wb);
template void BinaryInnerProductLayer<double>::binarizeGPUTo(const shared_ptr<Blob<double> > weights, const shared_ptr<Blob<double> > wb);
INSTANTIATE_LAYER_GPU_FUNCS(BinaryInnerProductLayer);

}  // namespace caffe

#else

#include "caffe/layers/binary_inner_product_layer.hpp"
namespace caffe {

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryInnerProductLayer);
}
#endif