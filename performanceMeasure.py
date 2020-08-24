import numpy as np


def performanceMeasure(trueClass, rawOutput, nClass):
    label = index2vector(trueClass, nClass)
    predictedLabel = index2vector(rawOutput, nClass)

    recall = calculate_recall(label, predictedLabel, nClass)
    result = np.dot(predictedLabel, label)
    error = 1 - np.sum(result.diagonal())/np.sum(np.dot(predictedLabel, label))
    precision = calculate_precision(label, predictedLabel, nClass)
    gMean = calculate_g_mean(recall,nClass)
    fMeasure = calculate_f_measure(lbel, predictedLabel, nClass)
    return error, precision,gMean,fMeasure

def index2vector(inputclass, nClass):
    output = np.zeros([len(inputclass), nClass])
    for i in range(len(inputclass)):
        output[i][input[i]] = 1
    return output


def calculate_recall(label, predictedLabel, nClass):
    recall = np.zeros(nClass)
    for i in range(nClass):
        recall[i] = np.dot(label[:, i], predictedLabel[:, i]) / np.sum(label[:, i], axis=0)
    for a in recall:
        if np.isnan(a):
            a = 1
    return recall


def calculate_precision(label, predictedLabel, nClass):
    precision = np.zeros(nClass)
    for i in range(nClass):
        precision[i] = np.dot(label[:, i], predictedLabel[:, i]) / np.sum(predictedLabel[:, i], axis=0)
    for a in precision:
        if np.isnan(a):
            a = 1
    return precision


def calculate_f_measure(label, predictedLabel, nClass):
    fMeasure = zeros(nClass)
    for i in range(nClass):
        fMeasure[i] = np.dot(2*label[:, i], predictedLabel[:, i])/np.sum(predictedLabel[:, i], axis=0) + np.sum(label[:, i], axis=0)
    for a in fMeasure:
        if np.isnan(a):
            a = 1
    return fMeasure


def calculate_g_mean(recall, nClass):
    gMean = np.prod(recall, axis=1)
    gMean2 = np.linalg.matrix_power(gMean, 1/nClass)
    return gMean2


