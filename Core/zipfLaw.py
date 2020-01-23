from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from data import loader, load_preprocessing
from knn_classifier import knn

import distances
import hott
import numpy as np
# Download datasets used by Kusner et al from
# https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0
# and put them into
data_path = './data/'

# Download GloVe 6B tokens, 300d word embeddings from
# https://nlp.stanford.edu/projects/glove/
# and put them into
embeddings_path = './data/glove.6B/glove.6B.300d.txt'

# Pick a dataset (uncomment the line you want)
data_name = 'bbcsport-emd_tr_te_splitdict70.pkl'
# data_name = 'twitter-emd_tr_te_split.mat'
# data_name = 'r8-emd_tr_te3.mat'
# data_name = 'amazon-emd_tr_te_split.mat'
# data_name = 'classic-emd_tr_te_split.mat'
# data_name = 'ohsumed-emd_tr_te_ix.mat'

# Pick a seed
# 0-4 for bbcsport, twitter, amazon, classic
# r8 and ohsumed have a pre-defined train/test splits - just set seed=0
data = load_preprocessing(data_path + data_name)
p = 1
C = data['cost_T']
Data = data['proportions']
y = data['y']

matrixPVal = []
matrixDist = []
for i in range(1,10):
    dataDist, datalabels, barycenters, dataHist, listPvalue, listInfoDist = kclustering(5, Data, C, y, max_iter = 25)
    np.save(data_path + "ArrayPvalExp" + str(i) + ".npy", np.array(listPvalue))
    np.save(data_path + "ArrayDistExp" + str(i) + ".npy", np.array(dataHist))
    np.save(data_path + "ArrayDistInfoExp" + str(i) + ".npy", np.array(listInfoDist))
    matrixPVal.append(listPvalue)
    matrixDist.append(dataHist)


matrixPVal = np.array(matrixPVal)
matrixDist = np.array(matrixDist)
