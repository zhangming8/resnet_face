# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:55:55 2015

@author: 陈日伟 <riwei.chen@outlook.com>
@brief：在lfw数据库上验证训练好了的网络
"""
import sklearn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage 
import sys
caffe_root = '/data/Experiments/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import caffe
import sklearn.metrics.pairwise as pw


filelist_left='/data/msra_lfw/lfw/left.list'
filelist_right='/data/msra_lfw/lfw/right.list'
filelist_label='/data/msra_lfw/lfw/label.list'

itera_str='120000'
deploy_file='./ResNet_50_deploy.prototxt'
model_file='./snapshot/ResNet50Out192_iter_' + itera_str + '.caffemodel'
lfw_dir='/data/msra_lfw/lfw/'

#use_feature=False
use_feature=True
def read_imagelist(filelist):
    '''
    @brief：从列表文件中，读取图像数据到矩阵文件中
    @param：filelist 图像列表文件
    @return：4D 的矩阵
    '''
    fid=open(filelist)
    lines=fid.readlines()
    test_num=len(lines)
    fid.close()
    X=np.empty((test_num,3,224,224))
    i =0
    for line in lines:
        word=line.split('\n')
        filename=word[0]
        im1=skimage.io.imread(lfw_dir+filename,as_grey=False)
        image =skimage.transform.resize(im1,(224, 224))
        if image.ndim<3:
            print 'gray:'+filename
            X[i,0,:,:]=image[:,:]
            X[i,1,:,:]=image[:,:]
            X[i,2,:,:]=image[:,:]
        else:
            X[i,0,:,:]=image[:,:,0]
            X[i,1,:,:]=image[:,:,1]
            X[i,2,:,:]=image[:,:,2]
        i=i+1
    return X

def read_labels(labelfile):
    '''
    读取标签列表文件
    '''
    fin=open(labelfile)
    lines=fin.readlines()
    labels=np.empty((len(lines),))
    k=0;
    for line in lines:
        labels[k]=int(line)
        k=k+1;
    fin.close()
    return labels

def draw_roc_curve(fpr,tpr,acc,title='cosine',save_name='roc_lfw'):
    '''
    画ROC曲线图
    '''
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic using: '+title+' acc:'+str(acc))
    #plt.legend(loc="lower right")
    plt.show()
    plt.savefig(save_name+'.png')


def evaluate(metric='cosine'):
    '''
    @brief: 评测模型的性能
    @param：itera： 模型的迭代次数
    @param：metric： 度量的方法
    '''
    if use_feature == False:
        #设置为gpu格式
        caffe.set_mode_gpu()
        print 'loading caffemodel...'
        #net = caffe.Classifier(deploy_file, model_file, mean=np.load(fout))
        net = caffe.Classifier(deploy_file, model_file)
        print 'load succeed'
        mean_npy = np.ones([3, 256, 256], dtype=np.float)
        mean_npy[0,:,:] = 104
        mean_npy[1,:,:] = 117
        mean_npy[2,:,:] = 123
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', mean_npy.mean(1).mean(1))
        transformer.set_raw_scale('data', 255.0)
        transformer.set_channel_swap('data', (2,1,0))
    
        #需要对比的图像，一一对应
        
        print 'network input :' ,net.inputs  
        print 'network output： ', net.outputs
        #提取左半部分的特征
        X=read_imagelist(filelist_left)
        X.shape
        test_num=np.shape(X)[0]
        #data 是输入层的名字
        out = net.forward_all(data = X)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        feature1 = np.float64(out['deepid'])
        feature1 = np.reshape(feature1,(test_num,192))
        print 'feature1 forward finshed'
        np.savetxt('feature1_' + itera_str + '.txt', feature1, delimiter=',')
    
        #提取右半部分的特征
        X=read_imagelist(filelist_right)
        out = net.forward_all(data=X)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        feature2 = np.float64(out['deepid'])
        feature2=np.reshape(feature2,(test_num,192))
        print 'feature2 forward finished'
        np.savetxt('feature2_' + itera_str + '.txt', feature2, delimiter=',')
    else:
        feature1=np.loadtxt('feature1_' + itera_str + '.txt', dtype='float64', delimiter=',')
        feature2=np.loadtxt('feature2_' + itera_str + '.txt', dtype='float64', delimiter=',')
        test_num1=np.shape(feature1)[0]
        test_num2=np.shape(feature2)[0]
        assert(test_num1==test_num2)
        test_num=test_num1
 
    #提取标签    
    labels=read_labels(filelist_label)
    print 'read label finished'
    assert(len(labels)==test_num)
    #计算每个特征之间的距离
    mt=pw.pairwise_distances(feature1, feature2, metric=metric)
    print 'calculation of pairwise distance finished'
    predicts=np.empty((test_num,))
    for i in range(test_num):
          predicts[i]=mt[i][i]
        # 距离需要归一化到0--1,与标签0-1匹配
    for i in range(test_num):
            predicts[i]=(predicts[i]-np.min(predicts))/(np.max(predicts)-np.min(predicts))
    print 'accuracy is :',calculate_accuracy(predicts,labels,test_num)
    accuracy=calculate_accuracy(predicts,labels,test_num)
                 
    np.savetxt('predict.txt',predicts)           
    fpr, tpr, thresholds=sklearn.metrics.roc_curve(labels,predicts)
    print 'roc_curve finished'
    draw_roc_curve(fpr,tpr,acc=accuracy,title=metric,save_name='lfw_'+str(itera_str))
    print 'draw_roc_curve %r finished' %str('lfw_'+str(itera_str))
    
def calculate_accuracy(distance,labels,num):    
    '''
    #计算识别率,
    选取阈值，计算识别率
    '''    
    accuracy = []
    predict = np.empty((num,))
    threshold = 0.2
    while threshold <= 0.8 :
        for i in range(num):
            if distance[i] >= threshold:
                 predict[i] =1
            else:
                 predict[i] =0
        predict_right =0.0
        for i in range(num):
            if predict[i]==labels[i]:
              predict_right = 1.0+predict_right
        current_accuracy = (predict_right/num)
        accuracy.append(current_accuracy)
        threshold=threshold+0.001
    return np.max(accuracy)

if __name__=='__main__':
    metric='cosine'
    evaluate(metric)

