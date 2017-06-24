import sys, os
#sys.path.insert(0,os.environ['CAFFE_ROOT'] + '/python')
import caffe
class SiameseLabels(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 2:
       raise Exception('must have exactly two inputs')
    if len(top) != 1:
       raise Exception('must have exactly one output')
  def reshape(self,bottom,top):
    top[0].reshape( *bottom[0].shape )
  def forward(self,bottom,top):
    top[0].data[...] = (bottom[0].data == bottom[1].data).astype('f4')
  def backward(self,top,propagate_down,bottom):
      # no back prop
      pass
