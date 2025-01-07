import DataLoader
import Network

import numpy as np


training, validation, test = DataLoader.data_wrapper()

net = Network.network([784, 32, 10])

net.SGD(training, 30, 10, .01, test_data=test)
