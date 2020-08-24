import time
def ADL(data,I,chunkSize,epoch,alpha_w,alpha_d,delta):
    dataProportion = 1
    print("Autonomous deep learning started")
    nData,mn = data.shape
    M = mn - I
    l = 0
    nFolds = np.round(len(data[0])/chunkSize)
    chunk_size = np.round(nData/nFolds)
    round_nFolds = np.floor(nData/chunk_size)
    Data = []
    if round_nFolds == nFolds:
        if nFolds == 1:
            Data.append(data)
        else:
            for i in range(nFolds):
                l = l + 1
                if i<nFolds:
                    Data1 = data[i * chunk_size + 1:i * chunk_size, :]
                elif i == nFolds - 1:
                    Data1 = data[i * chunk_size + 1: , :]
                Data.append(Data1)
    else:
        if nFolds == 1:
            Data[0] = data
        else:
            for i in range(nFolds -1):
                l = l + 1
                Data1 = data[i * chunk_size + 1:i * chunk_size, :]
                Data.append(Data1)
            Data[nFolds-1] = data[((nFolds-1)*chunk_size +1): , :]

    buffer_x = []
    buffer_T = []
    tTest = []
    acttualLabel = []
    classPerdiction = []

    parameter = Parameter()
    K = 1
    parameter.net = netInit([I,K,M])
    parameter.ev = EV(mn,K)
    layer = 1
    alpha = alpha_d
    covariance =  np.zeros((M,M,M))
    covariance_old = covariance
    threshold = delta

    for iFolds in range(nFolds):
        x = Data[iFolds][:, 0: I -1]
        T = Data[iFolds][:,I:mn -1]
        bd = T.shape[0]

        # neural network testing
        start_time = time.time()
        print("====Chunk {} of {} =====".format(iFolds,Data.shape[1]))
        print('Discriminative Testing: running ...')
        parameter.net.t = iFolds
        testing(x,T,parameter.ev)
        parameter.net.test_time[iFolds] = time.time() - start_time

        # metrics calculation
        parameter.loss[iFolds] = parameter.net.loss[parameter.net.index]
        tTest[bd*iFolds+1+(1-bd):bd*(iFolds+1),:] = parameter.net.sigma
        parameter.residual_error[bd*iFolds+(1-bd):bd*iFolds,:] = parameter.net.residual_error
        parameter.cr[iFolds] = parameter.net.cr


        # last chunk testing
        if iFolds == nFolds -1:
            print("'=========Parallel Autonomous Deep Learning is finished=========\n'")
            break

        start_time = time.time()
        outputCovar = np.zeros((layer,layer,M))

        for jj in range(layer):
            for kk in range(layer):
                if parameter.net.beta[jj] != 0 and parameter.nrt.bet[kk] != 0:
                    for ll in range(M):
                        temp = np.cov(parameter.net.activityOutput[kk][:,ll],parameter.net.activityOutput[jj][:,ll])
                        outputCovar[jj,kk,ll] = temp[0,1]
                        covariance[jj,kk,ll] = ((covariance_old[jj,kk,ll]*(iFolds)) + (iFolds/iFold + 1)*outputCovar[jj,kk,ll] )/(iFolds + 1)

        covariance_old = covariance
        if layer>1:
            merged_list = np.empty([2,2])
            for iter1 in range(layer -1):
                for hh in range(layer -1 - iter1):
                    if parameter.net.beta[-2] != 0 or parameter.net.beta[hh] != 0:
                        MCI = np.zeros(M)
                        for o in range(M):
                            pearson = covariance[-2,hh,o]/np.sqrt(covariance[-2,-2,o] * covariance[hh,hh,o])
                            MCI[o] = (0.5*(covariance[hh,hh,o] + covariance[end - l,end - l,o]) - np.sqrt((covariance[hh,hh,o] + covariance[end - l,end - l,o])**2 - 4*covariance[end - l,end - l,o]*covariance[hh,hh,o]*(1 - pearson**2)))
                        if max(abs(MCI)) < threshold:
                            if merged_list[0,0] == False:
                                merged_list[0,0] = layer -1
                                merged_list[0,1] = hh
                            else:
                                nu = np.where(merged_list[:,1:-2]==layer -1)
                                nu1 = np.where(merged_list[:,1:-2]== hh -1)
                                if np.array(nu).size == 0 and np.array(nu1).size == 0:
                                    n_list = np.array([layer-1,hh-1])
                                    merged_list = np.array(merged_list,nlist,axis=0)
                            break
            del_list = []
            for itt in range(merged_list.shape[0]):
                noOfHighlyCorrelated = np.where(merged_list[itt,:] == 0)
                if np.array(noOfHighlyCorrelated).size == 0 :
                    if parameter.net.beta[merged_list[itt,0]] > parameter.net.beta[merged_list[itt,1]]:
                        deleteLayer = merged_list[itt,1]
                    else:
                        deleteLayer = merged_list[itt,0]
                    del_list.append(deleteLayer)

            if len(del_list)>0 and parameter.net.beta[del_list] != 0:
                print("'The Hidden Layer no {} is PRUNED around chunk {}]\n'".format(del_list, iFolds))
                parameter.net.beta[del_list] = 0
            parameter.prune_list = parameter.prune_list + len[del_list]
            parameter.prune_list_index = np.append(parameter.prune_list_index,del_list)

        # drift detection
        if iFolds >1:
            cuttingpoint = 0
            pp = length(T)
            F_cut = np.zeros(pp)
            F_cut[parameter.net.wrongClass,:] = 1
            Fupper = np.max(F_cut)
            Flower = np.min(F_cut)
            miu_F = np.mean(F_cut)
            for cut in range(pp):
                miu_G =np.mean(F_cut[0:cut,:])
                Gupper = np.max(F_cut[0:cut,:])
                Glower = np.min(F_cut[0:cut,:])
                epsilon_G = (Gupper - Glower)*np.sqrt((pp/(2*cut*pp))*np.log(1/alpha))
                epsilon_F = (Fupper - Flower) * np.sqrt((pp / (2 * cut * pp)) * np.log(1 / alpha))
                if (epsilon_G + miu_G) >= (miu_F + epsilon_F):
                    cuttingpoint = cut
                    miu_H = np.mean(F_cut[cuttingpoint+1:,:])
                    epsilon_D = (Fupper - Flower)*np.sqrt(((pp-cuttingpoint)/(2*cuttingpoint*(pp-cuttingpoint)))*np.log(1/alpha_d))
                    epsilon_W = (Fupper - Flower)*np.sqrt(((pp-cuttingpoint)/(2*cuttingpoint*(pp-cuttingpoint)))*np.log(1/alpha_w))
                    break

            if cuttingpoint == 0:
                miu_H = 0
                epsilon_D = 0
                epsilon_W = 0

            if np.abs(miu_G - miu_H) >epsilon_D and 1 < cuttingpoint < pp:
                st = 1
                print("Drift state: DRIFT")
                layer = layer + 1
                parameter.net.nLayer = parameter.net.nLayer + 1
                parameter.net.index = parameter.net.nnHiddenLayer
                print("New layer no {} is FORMED arojnf chunk {}".format(layer,iFolds))

                # initiate NN weight parameter
                wei = parameter.net.weight[layer - 2].shape[0]
                parameter.net.weight[layer-1] = np.random.normal(0,np.sqrt(2/(wei+1)),[wei +1])
                parameter.net.velocity[layer-1] = np.zeros(wei + 1)
                parameter.net.grad[layer -1] = np.zeros(wei + 1)

                #initiate new classifier weight
                parameter.net.weightSoftmax[layer -1] = np.random.normal(0,1,[M,2])
                parameter.net.momentumSoftmax[layer-1] = np.zeros(M,2)
                parameter.net.gradSoftmax[layer -1] = np.zeros(M,2)

                # initiate new voting weight
                parameter.net.beta[layer -1] = 1
                parameter.net.betaOld[layer -1] = 1
                parameter.net.p[layer -1] = 1

                # inititate iterative parameters
                parameter.ev.layer[layer-1] = layer
                parameter.ev.kl[layer - 1] = 0
                parameter.ev.K[layer - 1] = 1
                parameter.ev.cr[layer - 1] = 0
                parameter.ev.node[layer - 1] = []
                parameter.ev.miu_NS_old[layer - 1] = 0
                parameter.ev.var_NS_old[layer - 1] = 0
                parameter.ev.miu_NHS_old[layer - 1] = 0
                parameter.ev.var_NHS_old[layer - 1] = 0
                parameter.ev.miumin_NS[layer - 1] = []
                parameter.ev.miumin_NHS[layer - 1] = []
                parameter.ev.stdmin_NS[layer - 1] = []
                parameter.ev.stdmin_NHS[layer - 1] = []
                parameter.ev.BIAS2[layer - 1] = []
                parameter.ev.VAR[layer - 1] = []

                covariance = np.zeros((M, M, M))
                covariance_old = covariance

                if len(buffer_x) <1:
                    pass
                else:
                    # prepare input for training
                    x = np.array([parameter.net.activity[0],buffer_x])
                    T = np.array([T,buffer_T])
                    parameter.net.T = T
                    parameter.net = netFeedForward(parameter.net,x,T)

                buffer_x = []
                buffer_T = []
            elif epsilon_W <= np.abs(miu_G - miu_H) < epsilon_D and st !=2:
                print("drift state: WARNING")
                st = 2

                buffer_x = x
                buffer_T = T

            else:
                st = 3
                print("drift state stable")

                if len(buffer_x) <1:
                    pass
                else:
                    x = np.array(buffer_x,x)
                    T = [buffer_T,T]
                    parameter.net.T = T
                    parameter.net = netFeedForward(parameter.net,x,T)

                buffer_x = []
                buffer_T = []

        else:
            st = 3
            print("drift state STABLE")
            buffer_T = []
            buffer_x = []

        driftState = st
        nHidLayer = size(np.where(parameter.net.beta!=0))
        parameter.wl[iFolds -1] = parameter.net.index

        #training for winning layer
        if st != 2:
            parameter =  training(parameter,T,epoch,dataProportion)
            print("=========Hidden layer number %d was updated=========".format(parameter.net.index))

        parameter.net.update_time[iFolds-1] = time.time() - start_time
        # claer current data chunk
        # data{t} = 0
        parameter.net.activity = []

    # save numerical result
    parameter.drift = driftState
    parameter.nFolds = nFolds
    update_time = [np.mean(parameter.net.update_time),np.std(parameter.net.update_time)]
    test_time = [np.mean(parameter.net.test_time),np.std(parameter.net.test_time)]
    classification_rate = [np.mean(parameter.cr[1:-1]),np.std(parameter.cr[1:-1])]
    rlayer = [np.mean(nHidLayer), np.mean(nHidLayer)]
    LayerWeight = parameter.net.beta
    meanode = []
    stdnode = []
    for i in range(parameter.net.nnHiddenLayer):
        a = np.count_nonzero(parameter.net.nodes[i]==0)
        parameter.net.nodes[i] = parameter.net.nodes[i][a:iFolds]
        meanode.append(np.mean(parameter.net.nodes[i]))
        stdnode.append(np.std(parameter.net.nodes[i]))

    numberOfParameters = parameter.net.mnop
    HL = nHidLayer
    return parameter, update_time, test_time, classification_rate, rlayer, LayerWeight, meanode, stdnode, numberOfParameters, HL