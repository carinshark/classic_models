import numpy as np

name = "bc"


x = np.load("data/"+name+"_data.npy")

y = np.load("data/"+name+"_label.npy")

print(x)

y_length = y.size

train_amount = int(y_length*.9)
x_train = x[:train_amount,:]
y_train = y[:train_amount]

x_test = x[train_amount:,:]
y_test = y[train_amount:]

print("~~~")

print(y)
np.save("learn_data/"+name+"_train_data.npy",x_train)
np.save("learn_data/"+name+"_train_label.npy",y_train)

np.save("learn_data/"+name+"_test_data.npy",x_test)
np.save("learn_data/"+name+"_test_label.npy",y_test)
