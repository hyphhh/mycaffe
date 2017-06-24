#include <algorithm>
#include <vector>

#include "caffe/layers/discriminative_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DiscriminativeLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sqr_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_.Reshape(bottom[0]->num(), 1, 1, 1);
  z_.Reshape(bottom[0]->num(), 1, 1, 1);
  z_act_.Reshape(bottom[0]->num(), 1, 1, 1);
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void DiscriminativeLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.discriminative_loss_param().margin();
  Dtype tau = this->layer_param_.discriminative_loss_param().tau();
  caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_.mutable_cpu_data());
  caffe_sqr(count, diff_.cpu_data(), diff_sqr_.mutable_cpu_data());
  for(int i=0; i<num; ++i) {
    for(int j=0; j<channels; ++j) {
      dist_.mutable_cpu_data()[i] += diff_sqr_.cpu_data()[i*channels+j];
    }
  }
  caffe_add_scalar(num, tau*Dtype(-1.0), dist_.mutable_cpu_data());
  caffe_mul(num, bottom[2]->cpu_data(), dist_.cpu_data(), z_.mutable_cpu_data());
  caffe_add_scalar(num, margin*Dtype(1.0), z_.mutable_cpu_data());
  caffe_exp(num, z_.cpu_data(), z_.mutable_cpu_data());
  caffe_cpu_axpby(num, Dtype(1.0), z_.cpu_data(), Dtype(0), z_act_.mutable_cpu_diff());
  caffe_add_scalar(num, Dtype(1.0), z_act_.mutable_cpu_diff());
  caffe_log(num, z_act_.cpu_diff(), z_act_.mutable_cpu_data());
  Dtype loss(0.0);
  for(int i=0; i<num; ++i) {
    loss += z_act_.cpu_data()[i];
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DiscriminativeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  for(int i=0; i<2; ++i) {
    const Dtype sign = (i==0)?1:-1;
    caffe_div(num, z_.cpu_data(), z_act_.cpu_diff(), z_.mutable_cpu_diff());
    caffe_mul(num, z_.cpu_diff(), bottom[2]->cpu_data(), dist_.mutable_cpu_diff());
    for(int j=0; j<num; ++j) {
      for(int k=0; k<channels; ++k) {
        bottom[i]->mutable_cpu_diff()[j*channels+k] = Dtype(2.0) * sign* dist_.cpu_diff()[j] * diff_.cpu_data()[j*channels+k];
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DiscriminativeLossLayer);
#endif

INSTANTIATE_CLASS(DiscriminativeLossLayer);
REGISTER_LAYER_CLASS(DiscriminativeLoss);

}  // namespace caffe