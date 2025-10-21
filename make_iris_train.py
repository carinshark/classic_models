import numpy as np

x = np.load("data/iris_data.npy")

y = np.load("data/iris_label.npy")

print(x)

y_length = y.size

if y_length==0 or y_length>150:
    print("error!!!")
    print(f"y length is {y_length}")

else:
    train_amount = int(y_length*.9)
    x_train = x[:train_amount,:]
    y_train = y[:train_amount]

    x_test = x[train_amount:,:]
    y_test = y[train_amount:]


    np.save("learn_data/iris_train_data.npy",x_train)
    np.save("learn_data/iris_train_label.npy",y_train)

    np.save("learn_data/iris_test_data.npy",x_test)
    np.save("learn_data/iris_test_label.npy",y_test)
