import numpy as np
from sklearn import decomposition

self.name = "rice"

x_trn = np.load(f"learn_data/{self.name}_train_data.npy")
y_trn = np.load(f"learn_data/{self.name}_train_label.npy")

columns = x_trn[0].size

pca = decomposition.PCA(n_components=columns)

pca.fit(x_trn)


def get_start(ratio):
    total = 0
    for i in range(len(ratio)):
        total += ratio[i]
        if total>=.95:
            return i+1
        
        
ratio = pca.explained_variance_ratio_
print(ratio)

start=get_start(ratio)
print(f"the start is {start}")


sets = 10

new_train = np.zeros((x_trn.shape[0]*sets,columns))

new_labels = np.zeros(y_trn.shape[0]*sets,dtype="uint8")



def generate(pca,x,start):
    # pca = decomposition.PCA() #REMOVE THIS BEFORE RUNNING!!!!!!!!!
    original = pca.components_.copy()
    
    num = pca.components_.shape[0]
    
    a = pca.transform(x)

    for i in range(start,num):
        pca.components_[i,:] += np.random.normal(scale=.1,size=num)
    
    b = pca.inverse_transform(a)

    pca.components_ = original.copy()

    return b

samp = x_trn.shape[0]

for i in range(sets):
    if i==0:
        new_train[0:samp,:]  = x_trn
        new_labels[0:samp] = y_trn
    else:
        new_train[(i*samp):(i*samp+samp),:] = generate(pca,x_trn,start)

        new_labels[(i*samp):(i*samp+samp)] = y_trn

np.save(f"augmented_data/{self.name}_augment_data",new_train)
np.save(f"augmented_data/{self.name}_augment_label",new_labels)

