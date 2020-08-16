import numpy as np


class NeuralNetwork:
    def __init__(self, input):
        self.initialConfig = input
        self.nLayer = self.initialConfig.size
        self.nHiddenLayer = self.nLayer - 2
        self.learningRate = 0.01
        self.momentumCoeff = 0.95
        self.outputConnect = 1
        self.output = "softmax"

        self.weight = []
        self.velocity = []
        self.grad = []
        self.c = []
        # initiate weights

        for i in range(self.nLayer - 2):
            weight1 = np.random.rand(input[i + 1], input[i] + 1)
            self.weight.append(weight1)
            self.velocity.append(np.zeros(weight1.shape))
            self.grad.append(np.zeros(weight1.shape))
            self.c.append(np.random.rand(input[i + 1], 1))

        self.weightSoftmax = []
        self.momentumSoftmax = []
        self.gradSoftmax = []
        self.beta = []
        self.betaOld = []
        self.p = []

        for ing in range(self.nHiddenLayer):
            weight2 = np.random.rand(input[self.nHiddenLayer, self.initialConfig[ing + 1] + 1])
            self.weightSoftmax.append(weight2)
            self.momentumSoftmax.append(np.zeros(weight2.shape))
            self.gradSoftmax.append(np.zeros(weight2.shape))
            self.beta.append(1)
            self.betaOld.append(1)
            self.p.append(1)
        return self

    def netInitWinner(self, layer):
        self.initialConfig = layer
        self.nLayer = numel(self.initialConfig)
        self.learningRate = 0.01
        self.momentumCoeff = 0.95
        return self


def meanstditer(miu_old, var_old, x, k):
    miu = miu_old + np.divide(x - miu_old, k)
    var = var_old + np.multiply(x - miu_old, x - miu)
    std = np.sqrt(var / k)
    return miu, std, var


def prohibit(miu, std):
    p = np.divide(miu, 1 + np.multiply(np.pi, ((std ** 2) / 8) ** 0.5))
    return p


def stableSoftmax(activation, weight):
    output = activation * np.transpose(weight)
    output = np.exp(output - max(np.amax(output, axis=1)))
    output = np.divide(output, np.sum(output, axis=1))
    return output



    # def feedforward(self):
    #     self.layer1 = sigmoid(np.dot(self.input, self.weights1))
    #     self.output = sigmoid(np.dot(self.layer1, self.weights2))
    #
    # def backprop(self):
    #     # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
    #     d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
    #     d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
    #                                               self.weights2.T) * sigmoid_derivative(self.layer1)))
    #
    #     # update the weights with the derivative (slope) of the loss function
    #     self.weights1 += d_weights1
    #     self.weights2 += d_weights2
