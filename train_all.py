from train import Train

"""
iris = Train("iris")
iris.train_data()
iris.train_augmented()

bc = Train("bc")
bc.train_data()
bc.train_augmented()

rice = Train("rice")
rice.train_data()
rice.train_augmented()

iris.visualize_info()
bc.visualize_info()
rice.visualize_info()

"""


letters = Train("mnist")

letters.trn_data_name = "mnist_train_images"
letters.trn_label_name = "mnist_train_labels"

letters.tst_data_name = "mnist_test_images"
letters.tst_label_name = "mnist_test_labels"

letters.train_data()

letters.visualize_info()