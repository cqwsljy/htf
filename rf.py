# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:00:11 2017

@author: liujiayong
"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import (manifold, decomposition, ensemble,random_projection)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd

def plot_embedding_3d(X, y,title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)
    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i,2],str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)

def plot_embedding_2d(X,y, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1],str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)


def loadtarget(X,Y,index=[4,7,9]):
    dflable = pd.DataFrame(Y,columns=["lable"])
    c = dflable["lable"].isin(index)
    Xnew = X[c,:]
    Ynew = Y[c]
    return [Xnew,Ynew]

mnist = input_data.read_data_sets('D:/Download/MNIST')

#X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)
## random forest
#traindata = mnist.train.images
#trainlable = mnist.train.labels


traindata,trainlable = loadtarget(mnist.train.images,mnist.train.labels)

#clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
#clf.fit(traindata,trainlable)
#index = [409,346,350]



X_train, X_test, y_train, y_test = train_test_split(traindata,trainlable, test_size=0.03, random_state=0)


##PCA
print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=3).fit_transform(X_test)
plot_embedding_2d(X_pca[:,0:2],y_test,"PCA 2D")
plot_embedding_3d(X_pca,y_test,"PCA 3D (time %.2fs)" %(time() - t0))



#%%
#LDA
print("Computing LDA projection")
X2 = X_test.copy()
X2.flat[::X_test.shape[1] + 1] += 0.01  # Make X invertible
t0 = time()
lda = LinearDiscriminantAnalysis(n_components=3)
X_lda = lda.fit_transform(X2, y_test)
plot_embedding_2d(X_lda[:,0:2],y_test,"LDA 2D" )
plot_embedding_3d(X_lda,y_test,"LDA 3D (time %.2fs)" %(time() - t0))


# MDS
print("Computing MDS embedding")
clf = manifold.MDS(n_components=3, n_init=1, max_iter=100)
t0 = time()
X_mds = clf.fit_transform(X_test)
print("Done. Stress: %f" % clf.stress_)
plot_embedding_2d(X_mds,y_test,"MDS (time %.2fs)" %(time() - t0))
plot_embedding_3d(X_mds,y_test,"MDS (time %.2fs)" %(time() - t0))


lable = trainlable
c = (lable == 4)
c2 = (lable == 9)
c3 = (lable == 7)
c = c|c2|c3
data = traindata[:,index]
data = data[c,:]
lable = lable[c]
lable = lable == 4

X_train, X_test, y_train, y_test = train_test_split(data,lable, test_size=0.03, random_state=0)

X = X_test[:,0]
Y = X_test[:,1]
Z = X_test[:,2]

#cmap = plt.cm.RdYlBu
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color = np.array([0.8]*len(y_test))
color[y_test==4] = 0.1
ax.scatter(X, Y, Z,c=color)
plt.show()

