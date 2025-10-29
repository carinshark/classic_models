from train import Train


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