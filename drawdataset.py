import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
#from munkres import Munkres
import os
import scipy
from PIL import Image



if __name__ == '__main__':
    # load face images and labels
    # data = sio.loadmat('./Data/YaleBCrop025.mat')
    # img = data['Y']
    # I = []
    # Label = []
    # for i in range(img.shape[2]):
    #     for j in range(img.shape[1]):
    #         temp = np.reshape(img[:,j,i],[42,48])
    #         Label.append(i)
    #         I.append(temp)
    # I = np.array(I)
    # Label = np.array(Label[:])
    # Img = np.transpose(I,[0,2,1])
    # # Img = np.expand_dims(Img[:],3)
    #
    # n_input = [48,42]
    # ims = []
    # for i in [15,17,21,22,30]:
    #     ims.append(Img[i])
    # im = np.hstack(ims)
    # im = Image.fromarray(im)
    # im.save('data-yale.png')


    # data = sio.loadmat('./Data/ORL_32x32.mat')
    # Img = data['fea']
    # Label = data['gnd']
    # n_input = [32, 32]
    # Img = np.reshape(Img,[Img.shape[0],n_input[0],n_input[1]])
    # Img = np.transpose(Img, [0, 2, 1])
	#
    # ims = []
    # for i in range(5):
    #     ims.append(Img[i])
    # im = np.hstack(ims)
    # im = Image.fromarray(im)
    # im.save('data-orl.png')



    # data = sio.loadmat('./Data/COIL20.mat')
    # Img = data['fea']
    # Label = data['gnd']
    # n_input = [32, 32]
    # Img = np.reshape(Img,[Img.shape[0],n_input[0],n_input[1]])
    # Img = np.transpose(Img, [0, 2, 1])
    # Img = Img * 255
    # Img = Img.astype(np.uint8)
	#
    # ims = []
    # for i in range(0,40,8):
    #     ims.append(Img[i])
    # im = np.hstack(ims)
    # im = Image.fromarray(im)
    # im.save('data-coil20.png')

    # data = sio.loadmat('./Data//COIL100.mat')
    # Img = data['fea'][0:40 * 72]
    # Label = data['gnd'][0:40 * 72]
    # n_input = [32, 32]
    # Img = np.reshape(Img, [Img.shape[0], n_input[0], n_input[1]])
    # Img = np.transpose(Img, [0, 2, 1])
    # Img = Img * 255
    # Img = Img.astype(np.uint8)
    #
    # ims = []
    # for i in range(0, 40, 8):
    #     ims.append(Img[i])
    # im = np.hstack(ims)
    # im = Image.fromarray(im)
    # im.save('data-coil40.png')



    from tensorflow.examples.tutorials.mnist import input_data
    sc = 100
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    Img = []
    Label = []
    num = mnist.train.num_examples
    rawImg = mnist.train._images
    rawLabel = mnist.train._labels
    for i in range(10):
        ind = [ii for ii in range(num) if rawLabel[ii] == i]
        ind = ind[0:sc]
        if i == 0:
            Img = rawImg[ind]
            Label = rawLabel[ind]
        else:
            Img = np.concatenate([Img, rawImg[ind]])
            Label = np.concatenate([Label, rawLabel[ind]])
    Label = np.reshape(Label, (-1, 1))
    rawLabel = np.reshape(rawLabel, (-1, 1))
    n_input = [28, 28]

    Img = np.reshape(Img, [Img.shape[0], n_input[0], n_input[1]])
    Img = Img * 255
    Img = Img.astype(np.uint8)
    ims = []
    for i in range(200, 240, 8):
        ims.append(Img[i])
    im = np.hstack(ims)
    im = Image.fromarray(im)
    im.save('data-mnist.png')

    data = sio.loadmat('./Data/umist-32-32.mat')
    Img = data['img']
    Label = data['label']
    n_input = [32, 32]
    Img = np.reshape(Img, [Img.shape[0], n_input[0], n_input[1]])
    Img = np.transpose(Img, [0, 2, 1])

    ims = []
    for i in range(0, 20, 4):
        ims.append(Img[i])
    im = np.hstack(ims)
    im = Image.fromarray(im)
    im.save('data-umist.png')