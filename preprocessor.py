'''
This file contains all the code for preprocessing the data to our neural network model
'''
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder


def prepare_data(seperate=False):
    train = []
    tmp_fea = []
    drug_fea_tmp = []
    drug_fea = np.genfromtxt('datasets/DS2/didiMatrix.csv', delimiter=',')[:, :-1]
    interaction = np.genfromtxt("datasets/DS2/simMatrix.csv", delimiter=',')[:, :-1]
    flattened_interaction = interaction.ravel()
    for i in range(0, interaction.shape[0]):
        for j in range(0, interaction.shape[1]):
            drug_fea_tmp = list(drug_fea[i])
            if seperate:

                tmp_fea = (drug_fea_tmp, drug_fea_tmp)

            else:
                tmp_fea = drug_fea_tmp + drug_fea_tmp
            train.append(tmp_fea)

    return np.array(train), flattened_interaction


def transfer_array_format(data):
    formated_matrix1 = []
    formated_matrix2 = []
    for val in data:
        formated_matrix1.append(val[0])
        formated_matrix2.append(val[1])
    return np.array(formated_matrix1), np.array(formated_matrix2)

def preprocess_labels(labels):
    enc = OneHotEncoder(handle_unknown='ignore', categorical_features=[0])
    reshaped_labels = np.array(labels).reshape(-1, 1)
    enc_f = enc.fit_transform(reshaped_labels).toarray()
    return enc_f

