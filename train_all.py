from train import Train


iris = Train("iris")

iris.train_data()

bc = Train("bc")
bc.train_data()

rice = Train("rice")
rice.train_data()

iris.visualize_info()
bc.visualize_info()
rice.visualize_info()