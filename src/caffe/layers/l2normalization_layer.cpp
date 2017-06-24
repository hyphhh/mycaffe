#include <vector>

#include "caffe/layers/l2normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2NormalizationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  squared_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
}

template <typename Dtype>
void L2NormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* squared_data = squared_.mutable_cpu_data(); //���������ڴ洢ƽ���ģ��൱��ǰ����x_i^2
  int n = bottom[0]->num();
  int d = bottom[0]->count() / n;
  caffe_sqr<Dtype>(n*d, bottom_data, squared_data);
  for (int i=0; i<n; ++i) {
    Dtype normsqr = caffe_cpu_asum<Dtype>(d, squared_data+i*d);  //�ѵ�i��ͼ�е�ÿ��ƽ��������
    caffe_cpu_scale<Dtype>(d, pow(normsqr, -0.5), bottom_data+i*d, top_data+i*d);  //ע����������ű�����һ�� -0.5 �η���Ҳ����ǰ���ķ�ĸ
  }
}

template <typename Dtype>
void L2NormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int n = top[0]->num();
  int d = top[0]->count() / n;
  for (int i=0; i<n; ++i) {
    Dtype a = caffe_cpu_dot(d, top_data+i*d, top_diff+i*d);  //top_diff��Ӧ������ʽ�е�E��y��ƫ���������൱��ʵ�ֵ��Ƿ��������еڶ������Ͳ���
    caffe_cpu_scale(d, a, top_data+i*d, bottom_diff+i*d);  //�����൱��ʵ���˷��������еĵڶ�����������bottom_diff��
    caffe_sub(d, top_diff+i*d, bottom_diff+i*d, bottom_diff+i*d);  //����ʵ���˷����ķ��Ӳ��֣�������Ǵ�����bottom_diff��
    a = caffe_cpu_dot(d, bottom_data+i*d, bottom_data+i*d);  //�������ڼ����ĸ�ĸ������沿�֣�������ʵҲ������ǰ����ʱ��ʹ�ñ�������������
    caffe_cpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff+i*d, bottom_diff+i*d); //����ʵ��������������ʽ
  }
}

#ifdef CPU_ONLY
STUB_GPU(L2NormalizationLayer);
#endif

INSTANTIATE_CLASS(L2NormalizationLayer);
REGISTER_LAYER_CLASS(L2Normalization);

}  // namespace caffe