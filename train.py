import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier

import time


class Train:
    def __init__(self,name):
        self.name = name
        self.info = {}

        self.all_classifiers = [NearestCentroid(),KNeighborsClassifier(n_neighbors=1),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
    
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

        for cls in self.all_classifiers:
            self._train_and_show(x_trn,y_trn,x_tst,y_tst,cls)

    def _train_and_show(self,x_trn,y_trn,x_tst,y_tst,classifier):
        print(str(classifier))
        before = time.time()
        classifier.fit(x_trn,y_trn)
        fit = time.time()-before
        before = time.time()
        print(f"{""}")
        print(f"predictions :{classifier.predict(x_tst)}")
        elapsed = time.time()-before
        print(f"actual      :{y_tst}")
        score=classifier.score(x_tst,y_tst)
        print(f"score: {score}")
        
        print(f"time taken: {elapsed}")

        self.info[str(classifier)] = {
            "fitTime":fit,
            "predictTime":elapsed,
            "score":score
        }

    def visualize_info(self):
        for cls in self.info.keys():
            data_dict = self.info[cls]
            print(cls)
            for data in data_dict.keys():
                print(f"\t{data}: {self.info[cls][data]}")