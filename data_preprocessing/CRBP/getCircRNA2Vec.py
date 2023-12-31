import numpy as np
import gensim
from gensim.models import Word2Vec
# from keras.preprocessing.sequence import pad_sequences



def seq2ngram(seqs, k, s, wv):
    list22 = []
    print('need to n-gram %d lines' % len(seqs))

    for num, line in enumerate(seqs):
        if num < 3000000:
            line = line.strip()
            l = len(line)
            list2 = []
            for i in range(0, l, s):
                if i + k >= l + 1:
                    break
                list2.append(line[i:i + k])
            list22.append(convert_data_to_index(list2, wv))
    return list22


def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data

def dealwithCircRNA2Vec(protein):
    dataX_pos = []
    dataX_neg=[]
    with open('Datasets/circRNA-RBP/' + protein + '/positive') as f:
        for line in f:
            if '>' not in line:
                dataX_pos.append((line.strip()).replace('T', 'U'))
    with open('Datasets/circRNA-RBP/' + protein + '/negative') as f:
        for line in f:
            if '>' not in line:
                dataX_neg.append((line.strip()).replace('T', 'U'))
    dataX_pos = np.array(dataX_pos)
    dataX_neg = np.array(dataX_neg)


    k = 10
    s = 1
    vector_dim = 30
    MAX_LEN = 101
    model1 = gensim.models.Doc2Vec.load('circRNA2Vec/circRNA2Vec_model')
    pos_list = seq2ngram(dataX_pos, k, s, model1.wv)
    neg_list = seq2ngram(dataX_neg, k, s, model1.wv)
    seqs = pos_list + neg_list

    X = [seq+[0] *(MAX_LEN-len(seq)) for seq in seqs]
    X = np.array(X)

    embedding = np.zeros((X.shape[0], X.shape[1], vector_dim))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # if X[i][j] != "None":
            num = int(X[i][j])
            embedding_vector = model1.wv[model1.wv.index2word[num]]
            embedding[i][j] = embedding_vector


    return embedding





