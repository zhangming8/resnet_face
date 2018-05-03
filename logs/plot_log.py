# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#先执行下面语句，用于生产.train与.test文件
#/data/Experiments/caffe/tools/extra/parse_log.py res50_3.log ./

log_name  = 'res50_3.log'
plot_loss_max = 5

test = open(log_name+'.test')
train = open(log_name+'.train')
#%% plot train
iteration ,loss, acc= [],[],[]
train = train.readlines()
for line in train[1:]:
    line = line.strip()
    split = line.split(',')
    iteration.append(int(float(split[0])))
    loss.append(float(split[4]))
    acc.append(float(split[3]))
# train loss
plt.plot(iteration,loss,label = 'train loss', color = 'blue',linewidth = 1)
plt.xlabel('iteration')
plt.ylabel('train loss')
plt.ylim(0,plot_loss_max)
plt.legend()
plt.savefig(log_name+'_train_loss.png',dpi = 300)
plt.show()
# train acc
plt.plot(iteration,acc,label='train acc', color = 'blue',linewidth = 1)
plt.xlabel('iteration')
plt.ylabel('train acc')
plt.ylim(0,1)
plt.legend(loc='lower right')
plt.savefig(log_name+'_train_acc.png',dpi = 300)
plt.show()

#%% plot val
iteration ,loss, acc= [],[],[]
test = test.readlines()
for line in test[1:]:
    line = line.strip()
    split = line.split(',')
    iteration.append(int(float(split[0])))
    loss.append(float(split[4]))
    acc.append(float(split[3]))
# val loss
plt.plot(iteration,loss,label='val loss', color = 'blue',linewidth = 1)
plt.xlabel('iteration')
plt.ylabel('val loss')
plt.ylim(0,plot_loss_max)
plt.legend()
plt.savefig(log_name+'_val_loss.png',dpi = 300)
plt.show()
# val acc
plt.plot(iteration,acc,label='val acc', color = 'blue',linewidth = 1)
plt.xlabel('iteration')
plt.ylabel('val acc')
plt.ylim(0,1)
plt.legend(loc='lower right')
plt.savefig(log_name+'_val_acc.png',dpi = 300)
plt.show()