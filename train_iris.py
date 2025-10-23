import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

name = "iris"

x_trn = np.load("learn_data/"+name+"_train_data.npy")
y_trn = np.load("learn_data/"+name+"_train_label.npy")

x_tst = np.load("learn_data/"+name+"_test_data.npy")
y_tst = np.load("learn_data/"+name+"_test_label.npy")



def train_and_show(x_trn,y_trn,x_tst,y_tst,classifier):
    classifier.fit(x_trn,y_trn)
    print(f"{""}")
    print(f"predictions :{classifier.predict(x_tst)}")
    print(f"actual      :{y_tst}")
    print(f"score: {classifier.score(x_tst,y_tst)}")
    print()

print("nearest centroid")
train_and_show(x_trn,y_trn,x_tst,y_tst,NearestCentroid())
print("KNeighbors, 1")
train_and_show(x_trn,y_trn,x_tst,y_tst,KNeighborsClassifier(n_neighbors=1))
print("KNeighbors, 3")
train_and_show(x_trn,y_trn,x_tst,y_tst,KNeighborsClassifier(n_neighbors=3))

