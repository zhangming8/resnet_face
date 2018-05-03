#coding=utf-8
#加载必要的库
import numpy as np
import json
import sys,os

#设置当前目录
caffe_root = '/data/wule/Exp4/caffe-face/' 
sys.path.insert(0, caffe_root + 'python')
import caffe

net_file=caffe_root + 'face_example/face_deploy.prototxt'
caffe_model=caffe_root + 'face_example/face_snapshot/face_train_test_iter_195000.caffemodel'
mean_file=caffe_root + 'examples/scene/mean.npy'
img_dir = '/data/wule/Exp4/msra_lfw/lfw/'
net = caffe.Net(net_file,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
mu = np.array([127.5, 127.5, 127.5])
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255) 
transformer.set_channel_swap('data', (2,1,0))
imgs_dir = open('/data/wule/Exp4/msra_lfw/lfw/imglist', 'r')
result = []
for img in imgs_dir.readlines():
	im=caffe.io.load_image(img_dir+img.strip())
	net.blobs['data'].data[...] = transformer.preprocess('data',im)
	out = net.forward()
	feature = net.blobs['fc5'].data[0].tolist()
	result_str = str(feature[0])
	for item in feature[1:]:
		result_str += ' ' + str(item)
	result.append(result_str)
imgs_dir.close()
for i in range(len(result)):
	f = open(os.path.join('feature.txt'), 'a')
	f.write(result[i] + '\n')
	f.close()




