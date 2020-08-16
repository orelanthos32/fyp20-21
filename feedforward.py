import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def netFeedForward(net, x, output):
    nLayer = net.nLayer
    batchSize = np.size(x, 0)
    y = np.ones(batchSize)
    y.append(x)
    net.activity[0] = y
    array2 = np.ones(batchSize)
    # feed forward
    for i in range(nLayer - 2):
        net2 = sigmoid(net.activity[i] * np.transpose(net.weight[i]))
        array2.append(net2)
        net.activity[i + 1] = array2

    # propagate to output layer
    for i in range(net.nHiddenLayer):
        if net.beta[i] != 0:
            net.activityOutput[i] = stableSoftmax(net.activity[i+1],net.weightSoftmax[i])

        # calculate Error
        net.error[i] = output - net.activityOutput[i]
        # calculate loss function
        net.loss[i] = -np.sum(np.multiply(output, np.log(net.activityOutput[i])))/batchSize
    return net
