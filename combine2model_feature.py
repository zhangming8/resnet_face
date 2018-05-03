# -*- coding: utf-8 -*-
import numpy as np

#%%
def pca(X = np.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""
	print ("Preprocessing the data using PCA...")
	(n, d) = X.shape
	X = X - np.tile(np.mean(X, 0), (n, 1))
	(l, M) = np.linalg.eig(np.dot(X.T, X))
	Y = np.dot(X, M[:,0:no_dims])
	return Y
 
if __name__ == '__main__':
    
    feature1 = 'vgg_fc7_features.txt'
    feature2 = 'res_pool5_features.txt'
    save_feature = 'combine_vgg_res_feature.txt'
    save_pca_feature = 'combine_vgg_res_pca_feature.txt'
    
    t1 = open(feature1).readlines()
    t2 = open(feature2).readlines()
    if len(t1) != len(t2):
        print('error!')
    with open(save_feature,'w') as f:
        for i in range(len(t1)):
            f.write(t1[i].strip() +' '+ t2[i])
    print('loading...')
    X = np.loadtxt(save_feature)
    print('file shape is:%d x %d' %(X.shape[0],X.shape[1]))
    X = pca(X, 200).real
    print('after PCA shape is:%d x %d' %(X.shape[0],X.shape[1]))
    np.savetxt(save_pca_feature,X)


