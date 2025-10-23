import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier

import time


class Train:
    def __init__(self,name):
        self.name = name
        self.info = {}
    
    def make_train(self):
        x = np.load("data/"+self.name+"_data.npy")

        y = np.load("data/"+self.name+"_label.npy")

        print(x)

        y_length = y.size

        train_amount = int(y_length*.9)
        x_train = x[:train_amount,:]
        y_train = y[:train_amount]

        x_test = x[train_amount:,:]
        y_test = y[train_amount:]

        print("~~~")

        print(y)
        np.save("learn_data/"+self.name+"_train_data.npy",x_train)
        np.save("learn_data/"+self.name+"_train_label.npy",y_train)

        np.save("learn_data/"+self.name+"_test_data.npy",x_test)
        np.save("learn_data/"+self.name+"_test_label.npy",y_test)


    
    def train_data(self):
        x_trn = np.load("learn_data/"+self.name+"_train_data.npy")
        y_trn = np.load("learn_data/"+self.name+"_train_label.npy")

        x_tst = np.load("learn_data/"+self.name+"_test_data.npy")
        y_tst = np.load("learn_data/"+self.name+"_test_label.npy")

        print("nearest centroid")
        self.train_and_show(x_trn,y_trn,x_tst,y_tst,NearestCentroid())
        print("KNeighbors, 1")
        self.train_and_show(x_trn,y_trn,x_tst,y_tst,KNeighborsClassifier(n_neighbors=1))
        print("KNeighbors, 3")
        self.train_and_show(x_trn,y_trn,x_tst,y_tst,KNeighborsClassifier(n_neighbors=3))
        print("DecisionTreeClassifier")
        self.train_and_show(x_trn,y_trn,x_tst,y_tst,DecisionTreeClassifier())

    def train_and_show(self,x_trn,y_trn,x_tst,y_tst,classifier):
        before = time.time()
        classifier.fit(x_trn,y_trn)
        fit = time.time()-before
        before = time.time()
        print(f"{""}")
        print(f"predictions :{classifier.predict(x_tst)}")
        print(f"actual      :{y_tst}")
        print(f"score: {classifier.score(x_tst,y_tst)}")
        elapsed = time.time()-before
        print(f"time taken: {elapsed}")

        self.info[str(classifier)] = {
            "fitTime":fit,
            "predictTime":elapsed
        }



    