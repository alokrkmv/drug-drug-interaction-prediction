'''
Author : Alok Kumar
Date : 24/12/2020
This file contains the logic to train and evaluate the model. The model is trained on DS2 dataset and a post trained model is saved as keras model for hot reload.
For training the model on other datasets(DS1 or DS3) the input vector size of the neural network  needs to be adjusted as per the size of intergrated similarity matrix or
else the model will not compile.
'''
import numpy as np
from preprocessor import prepare_data,preprocess_labels,transfer_array_format
from model import DNN
from evaluator import calculate_performace
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
if __name__ == '__main__':

    X, labels = prepare_data(seperate=True)
    X_data1, X_data2 = transfer_array_format(X)
    y = preprocess_labels(labels)
    X = np.concatenate((X_data1, X_data2), axis=1)
    X = np.concatenate((X_data1, X_data2), axis=1)
    num = np.arange(len(y))
    np.random.shuffle(num)
    X_data1 = X_data1[num]
    X_data2 = X_data2[num]
    y = y[num]
    num_cross_val = 2

    num_cross_val = 2

    for fold in range(num_cross_val):
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val == fold])
        train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
        test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])

        zerotest = 0
        nozerotest = 0
        zerotrain = 0
        nozerotrain = 0
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                nozerotest = nozerotest + 1
                real_labels.append(0)
            else:
                zerotest = zerotest + 1
                real_labels.append(1)
        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                zerotrain = zerotrain + 1
                train_label_new.append(0)
            else:
                nozerotrain = nozerotrain + 1
                train_label_new.append(1)

        prefilter_train = np.concatenate((train1, train2), axis=1)
        prefilter_test = np.concatenate((test1, test2), axis=1)

        train_label_new_forDNN = np.array([[0, 1] if i == 1 else [1, 0] for i in train_label_new])

        test_label_new_forDNN = np.array([[0, 1] if i == 1 else [1, 0] for i in real_labels])

        model_DNN = DNN()
        model_DNN.fit(prefilter_train, train_label_new_forDNN, epochs=30, batch_size=200)
        model_DNN.save("my_model")
        # model_DNN = models.load_model("my_model")
        print(model_DNN.summary())
        print(model_DNN.summary())
        proba = model_DNN.predict_classes(prefilter_test, batch_size=200, verbose=True)
        ae_y_pred_prob = model_DNN.predict_proba(prefilter_test, batch_size=200, verbose=True)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba, real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob[:, 1])
        auc_score = auc(fpr, tpr)

        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob[:, 1])
        aupr_score = auc(recall, precision1)
        # f = f1_score(real_labels, transfer_label_from_prob(ae_y_pred_prob[:,1]))
        all_F_measure = np.zeros(len(pr_threshods))

        print(auc_score)
        print("+++++")
        print(aupr_score)
        print("-----")
        # print(f"F-measure {f}")
        print(recall)
        print("+++++=====")
        print(precision)
        print("------++++++")
