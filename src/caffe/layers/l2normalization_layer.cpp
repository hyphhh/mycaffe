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
  Dtype* squared_data = squared_.mutable_cpu_data(); //这里是用于存储平方的，相当于前馈的x_i^2
  int n = bottom[0]->num();
  int d = bottom[0]->count() / n;
  caffe_sqr<Dtype>(n*d, bottom_data, squared_data);
  for (int i=0; i<n; ++i) {
    Dtype normsqr = caffe_cpu_asum<Dtype>(d, squared_data+i*d);  //把第i个图中的每个平方加起来
    caffe_cpu_scale<Dtype>(d, pow(normsqr, -0.5), bottom_data+i*d, top_data+i*d);  //注意这里的缩放比例有一个 -0.5 次方，也就是前馈的分母
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
    Dtype a = caffe_cpu_dot(d, top_data+i*d, top_diff+i*d);  //top_diff对应反馈公式中的E对y的偏导，这行相当于实现的是反馈分子中第二项的求和部分
    caffe_cpu_scale(d, a, top_data+i*d, bottom_diff+i*d);  //这里相当于实现了反馈分子中的第二项，结果存入了bottom_diff中
    caffe_sub(d, top_diff+i*d, bottom_diff+i*d, bottom_diff+i*d);  //这里实现了反馈的分子部分，结果还是存入了bottom_diff中
    a = caffe_cpu_dot(d, bottom_data+i*d, bottom_data+i*d);  //这里是在计算分母的根号里面部分，不过其实也可以在前馈的时候，使用保存下来的数据
    caffe_cpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff+i*d, bottom_diff+i*d); //最终实现了整个反馈公式
  }
}

#ifdef CPU_ONLY
STUB_GPU(L2NormalizationLayer);
#endif

INSTANTIATE_CLASS(L2NormalizationLayer);
REGISTER_LAYER_CLASS(L2Normalization);

}  // namespace caffe