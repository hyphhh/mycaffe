#include <vector>

#include "caffe/layers/euclidean_dist_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //Layer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  diff_sqr_.ReshapeLike(*bottom[0]);
  dist_.Reshape(bottom[0]->num(), 1, 1, 1);
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_sqr(
      count, 
      diff_.cpu_data(), 
      diff_sqr_.mutable_cpu_data());
  for(int i=0; i<num; ++i) {
    for(int j=0; j<channels; ++j) {
      dist_.mutable_cpu_data()[i] += diff_sqr_.cpu_data()[i*channels+j];
    }
    top[0]->mutable_cpu_data()[i] = dist_.cpu_data()[i];
  }
}

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  for(int i=0; i<2; ++i) {
    const Dtype sign = (i==0)?1:-1;
    for(int j=0; j<num; ++j) {
      const Dtype alpha = sign * top[0]->cpu_diff()[j] * 2;
      caffe_cpu_axpby(
        channels,
        alpha,
        diff_.cpu_data() + (j*channels),
        Dtype(0),
        bottom[i]->mutable_cpu_diff() + (j*channels));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanDistLayer);
#endif

INSTANTIATE_CLASS(EuclideanDistLayer);
REGISTER_LAYER_CLASS(EuclideanDist);

}  // namespace caffe