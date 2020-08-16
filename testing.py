def testing(net, input, trueClass, ev):
    net = netFeedForward(net, input, trueClass)
    nData, m2 = trueClass.shape
    decreasingFactor = 0.001
    # tbc
    [~, acttualLabel] = max(np.amax(trueClass, axis=1))