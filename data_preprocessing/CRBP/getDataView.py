import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # from keras.backend.tensorflow_backend import set_session
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import scipy.io as sio
# import matlab.engine
import torch

from utils import *
from data_preprocessing.CRBP.getSequence import *
from data_preprocessing.CRBP.getCircRNA2Vec import *
from data_preprocessing.CRBP.AnalyseFASTA import analyseFixedPredict
from data_preprocessing.CRBP.DProcess import convertRawToXY

def get_data(protein):

    Kmer, dataY = dealwithdata1(protein)
    Embedding = dealwithCircRNA2Vec(protein)

    np.random.seed(4)
    indexes = np.random.choice(Kmer.shape[0], Kmer.shape[0], replace=False)

    seqpos_path='Datasets/circRNA-RBP/' + protein + '/positive'
    seqneg_path ='Datasets/circRNA-RBP/' + protein + '/negative'

    pos_data, pos_ids, pos_poses = analyseFixedPredict(seqpos_path, window=20, label=1)
    neg_data, neg_ids, neg_poses = analyseFixedPredict(seqneg_path, window=20, label=0)
    #
    train_All2 = pd.concat([pos_data, neg_data])
    train_data = train_All2
    train_All = train_data

    trainX_PSTNPss_NCP, trainY_PSTNPss_NCP = convertRawToXY(train_All.values, train_data.values,
                                                            codingMode='PSTNPss_NCP_EIIP_Onehot')

    data_dict = dict()
    data_dict["samples1"] = torch.from_numpy(Kmer[indexes])
    data_dict["samples2"] = torch.from_numpy(Embedding[indexes])
    data_dict["samples3"] = torch.from_numpy(trainX_PSTNPss_NCP[indexes])#！！
    data_dict["labels"] = torch.from_numpy(dataY[indexes])

    return data_dict






