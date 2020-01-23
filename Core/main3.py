from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from data import loader, load_preprocessing
from knn_classifier import knn

import distances
import hott
from kcluster import kclustering

# Download datasets used by Kusner et al from
# https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0
# and put them into
data_path = './data/'

# Download GloVe 6B tokens, 300d word embeddings from
# https://nlp.stanford.edu/projects/glove/
# and put them into
embeddings_path = './data/glove.6B/glove.6B.300d.txt'

# Pick a dataset (uncomment the line you want)
# data_name = 'bbcsport-emd_tr_te_split.mat'
# data_name = 'twitter-emd_tr_te_split.mat'
# data_name = 'r8-emd_tr_te3.mat'
# data_name = 'amazon-emd_tr_te_split.mat'
# data_name = 'classic-emd_tr_te_split.mat'
# data_name = 'ohsumed-emd_tr_te_ix.mat'

# Pick a seed
# 0-4 for bbcsport, twitter, amazon, classic
# r8 and ohsumed have a pre-defined train/test splits - just set seed=0

p = 1
K_lda = 70

seed = 0
data_name = 'bbcsport-emd_tr_te_split.mat'
data = loader(data_path + data_name, embeddings_path, p=p, K_lda = 50)

seed = 1
data_name = 'twitter-emd_tr_te_split.mat'
#data = loader(data_path + data_name, embeddings_path, p=p, K_lda = 3)

seed = 2
data_name = 'amazon-emd_tr_te_split.mat'
#data = loader(data_path + data_name, embeddings_path, p=p, K_lda = 4)

seed = 3
data_name = 'classic-emd_tr_te_split.mat'
#data = loader(data_path + data_name, embeddings_path, p=p, K_lda = 4)

seed = 0
data_name = 'r8-emd_tr_te3.mat'
#data = loader(data_path + data_name, embeddings_path, p=p, K_lda = 8)

seed = 0
data_name = 'ohsumed-emd_tr_te_ix.mat'
#data = loader(data_path + data_name, embeddings_path, p=p, K_lda = 10)
