import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn import decomposition

import time


class Train:
    def __init__(self,name):
        self.name = name
        self.info = {}

        self.all_classifiers = [NearestCentroid(), #get center(mean)
                                KNeighborsClassifier(n_neighbors=1), #nearest datapoint
                                KNeighborsClassifier(n_neighbors=3), #3 nearest datapoints
                                DecisionTreeClassifier(), #probabiliyy
                                RandomForestClassifier(), #probability
                                GaussianNB()#, # probability
                                # MultinomialNB() # probability
                                ]
    
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
            print("this line is running")
            self._train_and_show(x_trn,y_trn,x_tst,y_tst,cls,"regular")

    def _train_and_show(self,x_trn,y_trn,x_tst,y_tst,classifier,category):
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

        len(self.info.keys())==0:
            self.info[str(classifier)] = {}

        self.info[str(classifier)][category] = {
            "fitTime":fit,
            "predictTime":elapsed,
            "score":score
        }

    def visualize_info(self):
        print(self.name)
        for cls in self.info.keys():
            categories = self.info[cls]
            print(f"\t{cls}")
            print(categories.keys())
            for cat in categories.keys():
                data_dict = self.info[cls][cat]
                print(f"\t\t{cat}")

                for data in data_dict.keys():
                    print(f"\t\t\t{data}: {self.info[cls][cat][data]}")



                
    def get_start(self,ratio):
        total = 0
        for i in range(len(ratio)):
            total += ratio[i]
            if total>=.95:
                return i+1
    
    def generate(self,pca,x,start):
        # pca = decomposition.PCA() #REMOVE THIS BEFORE RUNNING!!!!!!!!!
        original = pca.components_.copy()
        
        num = pca.components_.shape[0]
        
        a = pca.transform(x)

        for i in range(start,num):
            pca.components_[i,:] += np.random.normal(scale=.1,size=num)
        
        b = pca.inverse_transform(a)

        pca.components_ = original.copy()

        return b
    
    def augment(self):

        x_trn = np.load(f"learn_data/{self.name}_train_data.npy")
        y_trn = np.load(f"learn_data/{self.name}_train_label.npy")

        columns = x_trn[0].size

        pca = decomposition.PCA(n_components=columns)

        pca.fit(x_trn)

        
        ratio = pca.explained_variance_ratio_
        print(ratio)

        start=self.get_start(ratio)
        print(f"the start is {start}")


        sets = 10

        new_train = np.zeros((x_trn.shape[0]*sets,columns))

        new_labels = np.zeros(y_trn.shape[0]*sets,dtype="uint8")


        samp = x_trn.shape[0]

        for i in range(sets):
            if i==0:
                new_train[0:samp,:]  = x_trn
                new_labels[0:samp] = y_trn
            else:
                new_train[(i*samp):(i*samp+samp),:] = self.generate(pca,x_trn,start)

                new_labels[(i*samp):(i*samp+samp)] = y_trn

        np.save(f"augmented_data/{self.name}_augment_data",new_train)
        np.save(f"augmented_data/{self.name}_augment_label",new_labels)

    def train_augmented(self):
        x_trn = np.load("augmented_data/"+self.name+"_augment_data.npy")
        y_trn = np.load("augmented_data/"+self.name+"_augment_label.npy")

        x_tst = np.load("learn_data/"+self.name+"_test_data.npy")
        y_tst = np.load("learn_data/"+self.name+"_test_label.npy")

        for cls in self.all_classifiers:
            self._train_and_show(x_trn,y_trn,x_tst,y_tst,cls,"augmented")

