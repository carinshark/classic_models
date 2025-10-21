import numpy as np
import matplotlib.pyplot as plt

x_trn = np.load("learn_data/iris_train_data.npy")
y_trn = np.load("learn_data/iris_train_label.npy")

x_tst = np.load("learn_data/iris_test_data.npy")
y_tst = np.load("learn_data/iris_test_label.npy")


print(x_trn)
print("~~~`")
print(y_tst)

plt.boxplot(x_trn)

plt.show()