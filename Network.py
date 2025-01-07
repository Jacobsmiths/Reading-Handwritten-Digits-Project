import numpy as np

class network:
    def __init__(self, sizes):
        """Sizes being a list of elements were the first element in the list will be
            neurons in the first layer, second element being the number of neurons in
            the second layer and so on... 
            
            Biases is a list containing the biases of each layer
            
            and Weights will also be a list of the weight matrices
            both being initialized to be random values from 0 - 1
            
            The weights being an m x n matrix where m = layer output nodes
            and n = number of connections made from input layer to output layer"""

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2 / x) for x, y in zip(sizes[:-1], sizes[1:])]

    def func(self, x):
        # print("the value of x passed to func: {0}".format(x))
        result = np.maximum(0,x)
        # print("the resutl of func: {0}".format(result))
        return result
    
    def relu_prime(self, z):
        """Derivative of the relu function."""
        array = np.where(z > 0, 1, 0)
        return array
    
    def feedforward(self, x):
        """goal of this function is to apply the equation y = relu(Ax+b)
            given the input is a vector x."""
        activation = x
        for b, w in zip(self.biases, self.weights):
            z = (np.dot(w, activation)+b)
            # print("The value of z: {0}".format(z))
            activation = self.func(z)
            # print("the value of activation: ")
            # print(activation)
        return activation
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """This method is to "learn" by taking the gradient decent. 
        
            Provided training data in the form of a tuple (x,y), x = input and y = expected output,
            mini_batch_size is the sample size of data, eta is the training rate, and epochs is the
            numbers of times we will run this program of learning. Test data, if supplied, will 
            reveal how well the network is running, displaying 
            
            mini batches will be a list of tuples of random samples from the training data 
            to then be updated """
        
        if test_data: 
            n_test = len(test_data)
        
        n = len(training_data)
        for i in range(0, epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] #   training_data[0:1] 
            for j in range (0, len(mini_batches)):
                #print("The following is the input (mini batch): {0}".format(mini_batches[i]))
                self.update_mini_batch(mini_batches[j], eta)
            if test_data:
                n_test = len(test_data)
                print("After epoch: {0}, the test data success rate is: {1}/{2}\n".format(i, self.evaluate(test_data),n_test))
            else:
                print("Mini epoch number: {0} completed.".format(i))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # will return corresponding list of matrix gradients for w and b
            #print("The following is the gradient of biases: {0} and weights: {1}".format(delta_nabla_b, delta_nabla_w))
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # this is formatting nabla matrices to store gradient
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)] # updates values to minimize cost based on step val.
        #print("The following are the updates weights of L1: {0}".format(self.weights[1]))
        #print("The following is the updated biases of L1: {0}".format(self.biases[1]))

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = (np.dot(w, activation)+b)
            zs.append(z)
            activation = self.func(z)
            activations.append(activation)
        # backward pass
        #print("The following are activations post relu of backprop: {0}".format([activations[1], activations[2]]))
        #print("The value of the cost derivative is: {0}\nThen the value of relu prime of that is: {1}".format(self.cost_derivative(activations[-1], y), self.relu_prime(zs[-1])))

        delta = (self.cost_derivative(activations[-1], y) * self.relu_prime(zs[-1]))
        # Delta is the partial derivative of cost times with respect to activations of the last layer of activations and
        # the partial derivative of the last layer of activations with respect to the relu function. Essentially this is
        # the rate of change of the cost funciton with respect to Zj(L) which is where we start branching off
        #print("the following is the first value of delta:{0}".format(delta))
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # np.dot(delta, activations[-2].transpose())
        # so the first values of gradient b and gradient w are correct where the gradient of weights is determined 
        # by the dot product of the cost function with respect to the Zj(L) and activations of the previous layer
        # to get the activation times the delta of the corresponding position

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.relu_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #print("the new delta is: {0}".format(delta))
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        #print("The testing data input and exepcted output as a tuple: {0}".format(test_data[0]))

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        
        for(x,y) in [test_data[int(np.random.randint(1,10))]]:
        #     print(self.feedforward(x),y)
        #     print(np.argmax(self.feedforward(x)),y)
            print("the values of the activation of a random value from test data is: \n{0}\nAnd the compared values are: {1}, {2}".format(
                self.feedforward(x), np.argmax(self.feedforward(x)), y))


        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


        