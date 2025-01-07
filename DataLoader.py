import pickle
import gzip
import numpy as np

def load_data():
    """Scraped this code from github, but all its doing is uncompressing the gz file and streaming 
        bytes to the correspoding values. 
        
        Training data is originally in the format of a tuple, the first value being a huge ndarray of 50,000 x 784 
        where each row is essentially a new image but represented in pixels greyscale format. and the second value 
        being also a huge ndarray of 50,000 x 1 because it is an array where each row stores the value the image is representing
        pip 
        validation data and test data are the same thing but consists of 10,000 values"""
    with gzip.open('./mnist.pkl.gz', 'rb') as f:
        # Load the pickled data from the file
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    # Return the loaded data
    return (training_data, validation_data, test_data)

def data_wrapper():
    """After reading what the last method use is, it seems obvious we would want to rewrite how these tuples are stored.
        basically we would want to return training data in the form of a list of 50,000 tuples where the first value 
        in the tuple would be the matrix of greyscale pixels or "the image" and then the second value would be the correct 
        corresponding number in the form of a 10x1 matrix where 1.0 is the value found in the position the vector is representing.
        
        we are going to also do the same for test data and validation data
        
        the function returns 3 lists of all tuples consisting of values in training data, validation data and test data"""
    
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]] 
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    
    test_data = zip(test_inputs, te_d[1])
    return (list(training_data), list(validation_data), list(test_data))

def vectorized_result(j):
    """This code is basically saying given value j representing a number from 0-9, we want it in column vector form
        with standard format/ with the basis B = ((1,0,0,0,0,0,0,0,0,0)^T,(0,1,0,0,0,0,0,0,0,0)^T,....,(0,0,0,0,0,0,0,0,0,1)^T)"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e