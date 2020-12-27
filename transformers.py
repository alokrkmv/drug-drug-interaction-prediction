'''
Author : Alok Kumar
Date : 24/12/2020
This file contains all the functions which are required to transform data and create integrated similarity matrix from the data
'''
import numpy as np
import math
def FindDominantSet(W, K):
    m, n = W.shape
    DS = np.zeros((m, n))
    for i in range(m):
        index = np.argsort(W[i, :])[-K:]  # get the closest K neighbors
        DS[i, index] = W[i, index]  # keep only the nearest neighbors

    # normalize by sum
    B = np.sum(DS, axis=1)
    B = B.reshape(len(B), 1)
    DS = DS / B
    return DS


def normalized(W, ALPHA):
    m, n = W.shape
    W = W + ALPHA * np.identity(m)
    return (W + np.transpose(W)) / 2


def SNF(Wall, K, t, ALPHA=1):
    C = len(Wall)
    m, n = Wall[0].shape

    for i in range(C):
        B = np.sum(Wall[i], axis=1)
        len_b = len(B)
        B = B.reshape(len_b, 1)
        Wall[i] = Wall[i] / B
        Wall[i] = (Wall[i] + np.transpose(Wall[i])) / 2

    newW = []

    for i in range(C):
        newW.append(FindDominantSet(Wall[i], K))

    Wsum = np.zeros((m, n))
    for i in range(C):
        Wsum += Wall[i]

    for iteration in range(t):
        Wall0 = []
        for i in range(C):
            temp = np.dot(np.dot(newW[i], (Wsum - Wall[i])), np.transpose(newW[i])) / (C - 1)
            Wall0.append(temp)

        for i in range(C):
            Wall[i] = normalized(Wall0[i], ALPHA)

        Wsum = np.zeros((m, n))
        for i in range(C):
            Wsum += Wall[i]

    W = Wsum / C
    B = np.sum(W, axis=1)
    B = B.reshape(len(B), 1)
    W /= B
    W = (W + np.transpose(W) + np.identity(m)) / 2
    return W

def read_Sim_Calc_Entropy(fname, cutoff):
    entropy_exclude_zero_sumRow = []
    max_entropy = 0.0
    cutoff = float(cutoff)
    entropy = []
    small_number = 1 * pow(10, -16)
    arr = np.loadtxt(fname, delimiter=',')
    np.fill_diagonal(arr, 0)
    row, col = arr.shape
    aIndices_nonZero = []
    max_entropy = float(math.log(row, 2))

    for i in range(row):
        for j in range(col):
            if arr[i][j] < cutoff:
                arr[i][j] = 0

    for i in range(len(arr)):
        row_sum = arr[i].sum()
        row_entropy = 0

        if row_sum == 0:
            entropy.append(0)

        if row_sum > 0:
            aIndices_nonZero.append(i)
            arr[i] += small_number
            row_sum = arr[i].sum()
    for j in range(len(arr[i])):
        v = arr[i][j]
        cell_edited = (v) / row_sum
        # print 'cell_edited',cell_edited
        row_entropy = row_entropy + (cell_edited * math.log(cell_edited, 2))
        # print 'row_entropy',row_entropy
        row_entropy = row_entropy * -1
        entropy.append(row_entropy)

    for x in aIndices_nonZero:
        entropy_exclude_zero_sumRow.append(entropy[x])

    return np.mean(entropy), np.mean(entropy_exclude_zero_sumRow), round(max_entropy, 2)



def removeRedundancy(ranked_entropy_simType, all_euclideanDist_Sim):
    flT = 0.6
    m = 0
    iMEnd = len(ranked_entropy_simType)
    while m < iMEnd:
        A_simType = ranked_entropy_simType[m]
        n = m + 1
        iNEnd = len(ranked_entropy_simType)
        while n < iNEnd:
            B_simType = ranked_entropy_simType[n]

            if A_simType + ',' + B_simType in all_euclideanDist_Sim:
                key = A_simType + ',' + B_simType
            if B_simType + ',' + A_simType in all_euclideanDist_Sim:
                key = B_simType + ',' + A_simType

            flMax = all_euclideanDist_Sim[key]
            if flMax > flT:
                # oMC.deleteMotif(sMotB)
                del ranked_entropy_simType[n]
            else:
                n += 1
            iNEnd = len(ranked_entropy_simType)
        m += 1
        iMEnd = len(ranked_entropy_simType)
    print('ranked_entropy_simType', ranked_entropy_simType)

    return ranked_entropy_simType