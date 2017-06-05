from numpy import exp, array, random, dot
from adamoptimizer import AdamOptimizer


class NeuralNetwork():

    def __init__(self, l1=3, l2=5, l3=4):
        random.seed(2)

        # assign random weights to matrices in network
        # format is (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.synaptic_weights1 = 2 * random.random((l1, l2)) - 1
        self.synaptic_weights2 = 2 * random.random((l2, l3)) - 1
        self.synaptic_weights3 = 2 * random.random((l3, 1)) - 1
        
        self.adam1 = AdamOptimizer(self.synaptic_weights1, alpha=0.1)
        self.adam2 = AdamOptimizer(self.synaptic_weights2, alpha=0.1)
        self.adam3 = AdamOptimizer(self.synaptic_weights3, alpha=0.1)

    def __sigmoid(self, x):
        return 1/(1+exp(-x))

    def __sigmoid_derivative(self, x):
        return x*(1-x)

    def train(self, training_set_inputs, training_set_outputs, training_iter):
        for iteration in range(training_iter):
            # pass training set through our neural network
            # a2 means the activations fed to second layer

            a2 = self.__sigmoid(dot(training_set_inputs, self.synaptic_weights1))
            a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
            output = self.__sigmoid(dot(a3, self.synaptic_weights3))

            # calculate 'error'
            del4 = (output - training_set_outputs)*self.__sigmoid_derivative(output)

            # find 'errors' in each layer
            del3 = dot(self.synaptic_weights3, del4.T)*(self.__sigmoid_derivative(a3).T)
            del2 = dot(self.synaptic_weights2, del3)*(self.__sigmoid_derivative(a2).T)

            # get adjustments (gradients) for each layer
            adjustment3 = dot(a3.T, del4)
            adjustment2 = dot(a2.T, del3.T)
            adjustment1 = dot(training_set_inputs.T, del2.T)

            # adjust weights accordingly
            self.synaptic_weights1 = self.adam1.backward_pass(adjustment1)
            self.synaptic_weights2 = self.adam2.backward_pass(adjustment2)
            self.synaptic_weights3 = self.adam3.backward_pass(adjustment3)
            print (self.adam1.m)
            #self.synaptic_weights1 -= adjustment1
            #self.synaptic_weights2 -= adjustment2
            #self.synaptic_weights3 -= adjustment3
            
            
    def forward_pass(self, inputs):
        a2 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
        a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
        output = self.__sigmoid(dot(a3, self.synaptic_weights3)) 
        return output

if __name__ == "__main__":

    # initialise single neuron neural network
    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights (layer 1): ")
    print (neural_network.synaptic_weights1)
    print ("\nRandom starting synaptic weights (layer 2): ")
    print (neural_network.synaptic_weights2)
    print ("\nRandom starting synaptic weights (layer 3): ")
    print (neural_network.synaptic_weights3)

    # the training set.
    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = array([[1,1,0,0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 700)


    print ("\nNew synaptic weights (layer 1) after training: ")
    print (neural_network.synaptic_weights1)
    print ("\nNew synaptic weights (layer 2) after training: ")
    print (neural_network.synaptic_weights2)
    print ("\nNew synaptic weights (layer 3) after training: ")
    print (neural_network.synaptic_weights3)

    # test with new input
    print ("\nConsidering new situation [1,0,0] -> ?")
    print (neural_network.forward_pass(array([1,0,0])))