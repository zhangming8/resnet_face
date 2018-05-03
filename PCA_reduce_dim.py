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
    print('loading...')
    X = np.loadtxt('res_pool5_features_pca_400 0.9846.txt')
    print('file shape is:%d x %d' %(X.shape[0],X.shape[1]))
    X = pca(X, 400).real
    print('after PCA shape is:%d x %d' %(X.shape[0],X.shape[1]))
    np.savetxt('res_pool5_features_pca.txt',X)
