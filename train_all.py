from train import Train


# iris = Train("iris")
# iris.train_data()
# iris.train_augmented()

# bc = Train("bc")
# bc.train_data()
# bc.train_augmented()

# rice = Train("rice")
# rice.train_data()
# rice.train_augmented()

# iris.visualize_info()
# bc.visualize_info()
# rice.visualize_info()


#scrambled vectors

letters = Train("mnist_scrambled")

letters.trn_data_name = "mnist_train_scrambled_vectors"
letters.trn_label_name = "mnist_train_labels"

letters.tst_data_name = "mnist_test_scrambled_vectors"
letters.tst_label_name = "mnist_test_labels"

letters.train_data()

letters.visualize_info()

# images = Train("mnist_scrambled")

# images.trn_data_name = "mnist_train_scrambled_images"
# images.trn_label_name = "mnist_train_labels"

# images.tst_data_name = "mnist_test_scrambled_images"
# images.tst_label_name = "mnist_test_labels"

# images.train_data()

# images.visualize_info()