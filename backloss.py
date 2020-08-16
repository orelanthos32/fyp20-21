import numpy as np


def lossBackward(net):
    nLayer = net.nLayer
    backPropSignal = net.grad

    # if net.output == "sigmf":
    #     result1 = np.matmul(net.activity[nLayer],(1 - net.activity[nLayer]))
    #     backPropSignal = np.matmul(-net.error, result1)
    # else:
    backPropSignal[nLayer-1] = -net.error

    for i in range(nLayer - 2, 1, -1):
        actFuncDerivative = np.multiply(net.activity[i], 1 - net.activity[i])

        if i + 1 == nLayer:
            backPropSignal[i] = np.multiply((net.weight[i]*backPropSignal[i+1]), actFuncDerivative)
        else:
            backPropSignal[i] = np.multiply(net.weight[i]*(backPropSignal[i+1][:, 1:]), actFuncDerivative)

    for i in range(nLayer-1):
        if i + 1 == nLayer-1:
            net.grad[i] = net.activity[i] * np.transpose(backPropSignal[i+1])
        else:
            net.grad[i] = net.activity[i] * np.transpose(backPropSignal[i+1][:, 1:])
    return net
