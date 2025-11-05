import keras
from keras.datasets import mnist
import ssl
import numpy as np


ssl._create_default_https_context = ssl._create_unverified_context

(xtrn,ytrn),(xtst,ytst) = mnist.load_data()

#split data into test and train
idx = np.argsort(np.random.random(ytrn.shape[0]))
xtrn = xtrn[idx]
ytrn = ytrn[idx]
idx = np.argsort(np.random.random(ytst.shape[0]))
xtst = xtst[idx]
ytst = ytst[idx]



np.save("learn_data/mnist_train_images.npy",xtrn)
np.save("learn_data/mnist_train_labels.npy",ytrn)
np.save("learn_data/mnist_test_images.npy",xtst)
np.save("learn_data/mnist_test_labels.npy",ytst)


#convert each shape into a vector of numbers
xtrnv = xtrn.reshape((60000,28**2))
xtstv = xtst.reshape((10000,28**2))
np.save("learn_data/mnist_train_vectors.npy",xtrnv)
np.save("learn_data/mnist_test_vectors.npy",xtstv)

#reorder each thing
idx = np.argsort(np.random.random(28**2))

for i in range(60000):
    xtrnv[i,:]=xtrnv[i,idx]
for i in range(10000):
    xtstv[i,:] = xtstv[i,idx]

np.save("learn_data/mnist_train_scrambled_vectors.npy",xtrnv)
np.save("learn_data/mnist_test_scrambled_vectors.npy",xtstv)


#convert back to images
t = np.zeros((60000,28,28))

for i in range(60000):
    t[i,:,:] = xtrnv[i,:].reshape((28,28))

np.save("learn_data/mnist_train_scrambled_images.npy",t)

t = np.zeros((10000,28,28))

for i in range(10000):
    t[i,:,:]=xtstv[i,:].reshape(28,28)
np.save("learn_data/mnist_test_scrambled_images.npy",t)