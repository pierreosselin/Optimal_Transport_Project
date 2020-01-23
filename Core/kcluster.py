import random
import ot
import hott
import numpy as np
import scipy
from sklearn.cluster import KMeans
## Perform Clustering into k clusters according to wassertein barycenter, return partition

def computeTf(Data):
    data = Data / (Data.sum(axis = 1)[:,None])
    N = Data.shape[0]
    l = np.log(N/(Data >= 1).sum(axis = 0))
    data = data * l[None,:]
    return data

def clusterDistanceInfo(C1,C2,k1,k2): #np array length n each element label
        matrixk = np.zeros((k1,k2))
        vector1 = np.zeros(k1)
        vector2 = np.zeros(k2)
        n = C1.shape[0]
        for el1,el2 in zip(C1,C2):
            vector1[int(el1)] += 1.
            vector2[int(el2)] += 1.
            matrixk[int(el1),int(el2)] += 1.
        vector1 *= 1/n
        vector2 *= 1/n
        matrixk *= 1/n
        return -(vector1*np.log(vector1)).sum() - (vector2*np.log(vector2)).sum() - 2*(matrixk * ((np.ma.log(((matrixk/vector1[:,None])/vector2[None,:]))).filled(0))).sum()



def buildContingency(dataLabels, y, kClusters, nLabels):
    contingency = np.zeros((kClusters, nLabels))
    for el in zip(dataLabels, y):
        contingency[int(el[0]), int(el[1])] += 1
    return scipy.stats.chi2_contingency(contingency)[1]


def kclustering(k, Data, C, y, max_iter = 100, reg = 0.01, pVal = True, verbose = False): ## Data = (nSample, nfeatures), C = Cost matrix computed in Loader
    n = Data.shape[0]
    nbIter = 0
    nLabels = np.unique(y).shape[0]
    print("Loop n° :", nbIter)
    ### Random intialization : take k element as barycenters
    barycenters = Data[random.sample(range(Data.shape[0]), k)]

    ### Compute distance everyelement to the centers and label every every element
    distances = np.zeros((n,k)) # array (nData, k) = distance every element to the barycenter of the clusters
    datalabels = np.zeros(n)
    dataDist = np.zeros(n)
    dataHist = []
    listPvalue = []
    listInfoDist = []
    for i, data in enumerate(Data):
        minDistance = np.Inf
        for j, bar in enumerate(barycenters):
            distances[i,j] = hott.hoftt(data, bar, C)
            if distances[i,j] < minDistance:
                minDistance, datalabels[i] = distances[i,j], j
        dataDist[i] = minDistance
    dataHist.append(dataDist.sum())
    if pVal:
        listPvalue.append(buildContingency(datalabels, y, k, nLabels))
    listInfoDist.append(clusterDistanceInfo(y,datalabels,nLabels,k))
    while nbIter < max_iter:

        nbIter += 1
        print("Loop n° :", nbIter)
        ## Compute Centroids: wassertein Barycenters:
        for i in range(k):
            DataLabel_k = Data[datalabels == i]
            barycenters[i] = ot.bregman.barycenter_sinkhorn(DataLabel_k.T, C, reg, np.ones(DataLabel_k.shape[0])/DataLabel_k.shape[0], verbose = verbose)
        for i, data in enumerate(Data):
            minDistance = np.Inf
            for j, bar in enumerate(barycenters):
                distances[i,j] = hott.hott(data, bar, C, threshold=None)
                if distances[i,j] < minDistance:
                    minDistance, datalabels[i] = distances[i,j], j
            dataDist[i] = minDistance
        print("Distance to Barycenters")
        print(dataDist.sum())
        print("p-value")
        if pVal:
            listPvalue.append(buildContingency(datalabels, y, k, nLabels))
        dataHist.append(dataDist.sum())
        listInfoDist.append(clusterDistanceInfo(y,datalabels,nLabels,k))

    return dataDist, datalabels, barycenters, dataHist, listPvalue, listInfoDist

def kclustertf(k,Data,y, n_init = 1): #Data under the form of BOW
    data = computeTf(Data) # Convert BOW to tfidf
    result = KMeans(n_clusters=k,n_init = n_init, max_iter = 25).fit(data).labels_
    distInfoFinal = clusterDistanceInfo(y, result, k,k)
    distPvalue = buildContingency(result, y, k, k)
    return result, distInfoFinal, distPvalue

def kclusterLDA(k,Data,y): #Data under the form of (doc,TopicProportion)
    labels = Data.argmax(axis = 1)
    distInfoFinal = clusterDistanceInfo(y, labels, k,k)
    distPvalue = buildContingency(labels, y, k, k)
    return labels, distInfoFinal, distPvalue

def kclusterEmbedding(k,Data, y, n_init = 1): #Give embedding (nDoc, nEmbed)
    result = KMeans(n_clusters=k,n_init = n_init, max_iter = 25).fit(Data).labels_
    distInfoFinal = clusterDistanceInfo(y, result, k,k)
    distPvalue = buildContingency(result, y, k, k)
    return result, distInfoFinal, distPvalue
