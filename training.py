def training(parameter, y, nEpoch, dataProportion):
    bb = parameter.net.weight[parameter.net.index].shape[1]
    grow = 0
    prune = 0

    # initiate performance matrix
    ly = parameter.net.index
    kp = parameter.ev.kp[0]
    miu_x_old = parameter.ev.miu_x_old[0]
    var_x_old = parameter.ev.var_x_old[0]
    kl = parameter.ev.kl[ly]
    K = parameter.ev.K[ly]
    node = parameter.ev.node[ly]
    growingThreshold = parameter.ev.BIAS2[ly]
    pruningThreshold = parameter.ev.VAR[ly]
    miu_NS_old = parameter.ev.miu_NS_old[ly]
    var_NS_old = parameter.ev.var_NS_old[ly]
    miu_NHS_old = parameter.ev.miu_NHS_old[ly]
    var_NHS_old = parameter.evvar_NHS_old[ly]
    miumin_NS = parameter.ev.miumin_NS[ly]
    miumin_NHS = parameter.ev.miumin_NHS[ly]
    stdmin_NS = parameter.ev.stdmin_NS[ly]
    stdmin_NHS = parameter.ev.stdmin_NHS[ly]

    # initiate Training model
    net = netInitWinner([np.ones(3)])
    net.activationFunction = parameter.net.activationFunction
    net.output = parameter.net.output

    # substitute weight to training model
    net.weight[0] = parameter.net.weight[ly]
    net.velocity[0] = parameter.net.velocity[ly]
    net.grad[0] = parameter.net.grad[ly]
    net.weight[1] = parameter.net.weightSoftmax[ly]
    net.velocity[1] = parameter.net.momentumSoftmax[ly]
    net.grad[1] = parameter.net.gradSoftmax[ly]

    # load data for training
    x = parameter.net.activity[ly]
    nData, I = x.shape
    kk = np.random.permutation(nData)
    x = x[kk, :]
    y = y[kk, :]
    nLabeledData = np.round(dataProportion * nData)
    x = x[0:nLabeledData, :]
    y = y[0:nLabeledData, :]
    nData = x.shape[0]
    if ly > 1:
        n_in = parameter.ev.K[ly - 1]
    else:
        n_im = parameter.net.initialConfig[0]

    # main loop
    for i in range(nData):
        kp = kp + 1
        kl = kl + 1

        # increment bias and variance
        miu_x, std_x, var_x = meanstditer(miu_x_old, var_x_old, parameter.net.activity[0][i, :], kp)
        miu_x_old = miu_x
        var_x_old = var_x

        # expectation of output
        py = prohibit(miu_x, std_x)
        for j in range(parameter.net.index):
            if j == parameter.net.index - 1:
                py = sigmoid(net.weight[0] * py)
            else:
                py = sigmoid(parameter.net.weight[j] * py)
            py = [1, py]
            if j == 0:
                Ey2 = np.power(py, 2)

        Ey = py
        Ez = net.weight[1] * Ey
        Ez = np.exp(Ez - max(Ez))
        Ez = Ez / sum(Ez)

        if parameter.net.nHiddenLayer > 1:
            py = Ey2
            for j in range(1, parameter.net.index):
                if j == parameter.net.index - 1:
                    py = sigmoid(net.weight[0] * py)
                else:
                    py = sigmoid(parameter.net.weight[j] * py)
                py = np.array([1, py])
            Ey2 = py
        Ez2 = net.weight[1] * Ey2
        Ez2 = np.exp(Ez2 - max(Ez2))
        Ez2 = Ez2 / sum(Ez2)
        # Network mean calculation
        bias2 = (Ez - np.transpose(y[i, :])) ** 2
        ns = bias2
        NS = np.norm(ns)

        miu_NS, std_NS, var_NS = meanstditer(miu_NS_old, var_NS_old, NS, kl)
        miu_NS_old = miu_NS
        var_NS_old = var_NS
        miustd_NS = miu_NS + std_NS
        if kl <= 1 or grow == 1:
            miumin_NS = miu_NS
            stdmin_NS = std_NS
        else:
            if miu_NS < miumin_NS:
                miumin_NS = miu_NS
            if std_NS < stdmin_NS:
                stdmin_NS = std_NS
        miustdmin_NS = miumin_NS + (1.3 * np.exp(-NS)) * stdmin_NS
        growingThreshold[kp, :] = miu_NS
        if miustd_NS >= miustdmin_NS and kl > 1:
            grow = 1
            K = K + 1
            print("The new node no {} is FORMED around sample {}".format(K, kp))
            node[kp] = K
            # augment the weight
            net.weight[0] = np.array([net.weight[0], np.random.randn(1, bb)])
            net.velocity[0] = np.array([net.velocity[0], np.zeros(bb)])
            net.grad[0] = np.array([net.grad[0], np.zeros(bb)])
            net.weight[1] = np.array(
                [net.weight[1], np.random.randn(parameter.net.initialConfig[parameter.net.index - 1], 1)])
            net.velocity[1] = np.array(
                [net.velocity[1], np.zeros(parameter.net.initialConfig[parameter.net.index - 1], 1)])
            net.grad[1] = np.array(net.grad[1], np.zeros(parameter.net.initialConfig[parameter.net.index - 1], 1))

            if ly < parameter.net.nHiddenLayer:
                wNext = parameter.net.weight[ly + 1][0]
                parameter.net.weight[ly + 1] = np.array([parameter.net.weight[ly + 1], np.random.randn(wNext, 1)])
                parameter.net.velocity[ly + 1] = np.array([net.velocity[ly + 1], np.zeros(wNext, 1)])
                parameter.net.grad[ly + 1] = np.array(net.grad[ly + 1], np.zeros(wNext, 1))

        else:
            grow = 0
            node[kp] = K

        # Network variance calculation
        var = Ez2 - Ez ** 2
        NHS = np.norm(var)

        # incremental calc of NHS mean and variance
        miu_NHS, std_NHS, var_NHS = meanstditer(miu_NHS_old, var_NHS_old, NHS, kl)
        miu_NHS_old = miu_NHS
        var_NHS_old = var_NHS
        miustd_NHS = miu_NHS + std_NHS
        if kl <= I + 1 or prune == 1:
            miumin_NHS = miu_NHS
            stdmin_NHS = std_NHS
        else:
            if miu_NHS < miumin_NHS:
                miumin_NHS = miu_NHS
            if std_NHS < stdmin_NHS:
                stdmin_NHS = std_NHS

        miustdmin_NH = miumin_NHS + (2.6 * np.exp(-NHS) + 1.4) * stdmin_NHS
        pruningThreshold[kp, :] = miu_NHS

        if grow == 0 and K > 1 and miustd_NHS >= miustd_NHS and kl > I + 1:
            HS = Ey[1:nData - 1]
            BB = argmin(HS)
            print('The node no {} is PRUNED around sample {}\n'.format(BB, kp))
            prune = 1
            K = K - 1
            node[kp] = K
                # delete the weight
            net.weight[0][BB, :] = []
            net.velocity[0][BB, :] = []
            net.grad[0][BB, :] = []
            net.weight[1][:, BB + 1] = []
            net.velocity[1][:, BB + 1] = []
            net.grad[1][:, BB + 1] = []
            if ly < parameter.net.nHiddenLayer:
                parameter.net.weight[ly + 1][:, BB + 1] = []
                parameter.net.velocity[ly + 1][:, BB + 1] = []
                parameter.net.grad[ly + 1][:, BB + 1] = []
        else:
            node[kp] = K
            prune = 0

        # feedforward
        net = lossBackward(net)
        net = optimizerStep(net)
    # iterative learning
    if nEpoch >1:
        for i in range(nEpoch -1):
            kk = np.random.permutation(nData)
            x = x[kk, :]
            y = y[kk, :]
            for j in range(nData):
                net = netFeedForwardWinner(net, x[j, :], y[j, :])
                net = lossBackward(net)
                net = optimizerStep(net)
    #substitute weight back to main model
    parameter.net.weight[ly] = net.weight[0]
    parameter.net.weightSoftmax[ly] = net.weight[1]

    # reset momentumCoeff and gradient
    parameter.net.velocity[ly] = 0
    parameter.net.grad[ly] = 0
    parameter.net.momentumSoftmax[ly] = 0
    parameter.net.gradSoftmax[ly] = 0

    parameter.ev.kp[0] = kp
    parameter.ev.miu_x_old[0] = miu_x_old
    parameter.ev.var_x_old[0] = var_x_old
    parameter.ev.kl[ly] = kl
    parameter.ev.K[ly] = K
    parameter.ev.node[ly] = node
    parameter.ev.BIAS2[ly] = growingThreshold
    parameter.ev.VAR[ly] = pruningThreshold
    parameter.ev.miu_NS_old[ly] = miu_NS_old
    parameter.ev.var_NS_old[ly] = var_NS_old
    parameter.ev.miu_NHS_old[ly] = miu_NHS_old
    parameter.ev.var_NHS_old[ly] = var_NHS_old
    parameter.ev.miumin_NS[ly] = miumin_NS
    parameter.ev.miumin_NHS[ly] = miumin_NHS
    parameter.ev.stdmin_NS[ly] = stdmin_NS
    parameter.ev.stdmin_NHS[ly] = stdmin_NHS
