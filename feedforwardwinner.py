import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def netFeedForwardWinner(net, input2, output):
    nLayer = net.nLayer
    batchSize = np.size(input2, 0)
    net.activity[0] = input2
    array2 = np.ones(batchSize)
    # feed forward
    for i in range(nLayer - 2):
        net2 = sigmoid(net.activity[i] * np.transpose(net.weight[i]))
        array2.append(net2)
        net.activity[i + 1] = array2

    # propagate to output layer
    net.activity[nLayer] = stableSoftmax(net.activity[nLayer-1],net.weightSoftmax[nLayer-1])

    # calculate Error
    net.error = output - net.activity[nLayer]
    return net
