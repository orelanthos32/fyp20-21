import numpy as np


def testing(net, input, trueClass, ev):
    net = netFeedForward(net, input, trueClass)
    nData, m2 = trueClass.shape
    decreasingFactor = 0.001
    # label
    acttualLabel = np.argmax(trueClass, axis = 1)
    net.sigma = np.zeros([nData, m2])


    for i in range(nData):
        for j in range(net.nHiddenLayer):
            if net.beta[j] != 0:
                # obtain the predicted label, according equation 4.1

                net.sigma[i,:] = net.sigma[i,:] + net.activityOutput[j][i,:]*net.betaOld[j]
                # find out what this does
                net.classlabel[j][i,:] = np.argmax(net.activityOutput[j][i,:], axis=1)


                # train the dynamic voting weight beta
                compare = acctualLabel[i,:] - net.classlabel[j][i,:]
                if compare !=0:
                    net.p[j] = max(net.p[j]-decreasingFactor,decreasingFactor)
                    net.beta[j] = max(net.beta[j]*net.p[j], decreasingFactor)

                else:
                    net.p[j] = min(net.p[j]+decreasingFactor,1)
                    net.beta[j] = min(net.beta[j]*(1 +net.p[j]),1)
            if i == nData-1:
                if net.beta[j] !=0:
                    c, d = net.weightSoftmax[j].shape
                    vw = 1
                else:
                    c = 0
                    d = 0
                    vw = 0
                a, b = size(net.weight[j])
                nop = a*b + c*d +vw
                # calculate num of node in each hidden layer
                # matlab structure
                net.nodes[j][net.t] = ev.K[j]

    net.nop[net.t] = sum(nop)
    net.mnop = [mean(net.nop), np.std(net.nop)]
    # update voting weight
    net.beta = net.beta/sum(net.beta)
    net.betaOld = net.beta
    net.index = np.argmax(net.beta)

    # calculate classification rate

    multiClassProb = np.amax(net.sigma, axis=1)
    classPerdiction = np.argmax(net.sigma, axis=1)
    net.wrongClass = np.where(classPerdiction != acttualLabel)
    net.cr = 1 - np.size(net.wrongClass)/nData
    net.resudial_error = 1 - multiClassProb
    net.classPerdiction = classPerdiction
    net.acttualLabel = acttualLabel
    return net