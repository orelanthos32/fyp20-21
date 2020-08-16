def optimizerStep(net):
    for i in range(net.nLayer -1):
        grad = net.grad[i]
        net.velocity[i] = net.momentumCoeff*net.velocity[i] + net.learningRate*grad
        finalGrad = net.velocity[i]
        net.weight[i] = net.weight[i] - finalGrad
    return net
