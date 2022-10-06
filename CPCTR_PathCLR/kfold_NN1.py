from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
from sklearn.utils import shuffle
import random
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
import copy
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif 
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold

project_path = ""

n_patches = 200
best_acc = 0
n_features = 512
num = 1
skf = StratifiedKFold(10)


def create_baseline1():
    model = Sequential()
    model.add(Dense(300, input_dim=n_features, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.06, l2=0.06)))
    return model

tp_sum = 0
tn_sum = 0
fn_sum = 0
fp_sum = 0

def compute_CI(nums):
    avg = sum(nums) / len(nums)
    print(avg)

    s = 0
    for i in nums:
        s += (i - avg) * (i - avg)

    s /= (len(nums) - 1)
    s = s ** 0.5
    e = (1.96 * s) / (len(nums) ** 0.5)
    print("CI: ", avg, " + ", e)
    return avg, e

def result_by_patient(y_pred, y_true, case_id, fname, image_num, cutoff, best_acc):
    # labels_validation = np.reshape(labels_validation, (labels_validation.shape[0], 1))
    df = pd.DataFrame(
        {"caseid": case_id.ravel(), "votes": y_pred.ravel(), "labels": y_true.ravel(), "image_num": image_num.ravel()})

    by_image_prediction = df.groupby(['image_num'], as_index=False).mean()

    by_image_prediction["votes"] = (by_image_prediction["votes"] >= cutoff).astype(int)


   # final_vote = by_image_prediction.groupby(['caseid'], as_index=False).min()
    # final_vote = by_image_prediction.groupby(['caseid'], as_index=False).max()
    final_vote = by_image_prediction.groupby(['caseid'], as_index=False).mean()
    final_vote["labels"] = (final_vote["labels"]>=0.5).astype(int)
    final_vote["votes"] = final_vote["votes"].astype(int)
    # final_vote["labels"] = final_vote["labels"].astype(int) 
    cm = confusion_matrix(final_vote["labels"], final_vote["votes"])
    # print(cm)
    accuracy = (cm[0][0] + cm[1][1]) / len(final_vote["labels"])
    if best_acc <= accuracy:
        best_acc = accuracy
        global tn_tmp
        global fn_tmp
        global tp_tmp
        global fp_tmp

        tn, fp, fn, tp = confusion_matrix(final_vote["labels"], final_vote["votes"]).ravel()
        tn_tmp = tn
        fn_tmp = fn
        tp_tmp = tp
        fp_tmp = fp
    return best_acc, accuracy, cm

def read_label(filename):
    tmp = pd.read_csv(filename)
    labels = []
    case_id = []
    image_name = []
    for index, row in tmp.iterrows():
        n = row['ImageName'].split("/")[-1]
        ci = int(n.split("-")[2].split("_")[0])
        image_name.append(n)
        case_id.append(ci)
        labels.append(row['Reccured'])

    return labels, case_id, image_name

def read_data(project_path, path1, path2):
    embedding = np.load(project_path + path1)
    labels, case_id, image_name = read_label(project_path + path2)
    image_name, case_id, labels, embedding = zip(*sorted(zip(image_name, case_id, labels, embedding)))

    image_num = []
    counter = 0
    labels = np.array(labels)
    # embedding = embedding.reshape((s,2048))
    final_embedding = []
    for i in range(len(case_id)):
        e = np.array(embedding[i][:])
        final_embedding.append(e)
        if i % n_patches == 0:
            counter += 1
            if case_id[i] != case_id[i + n_patches - 1]:
                print("something is wroooong!!!")
                print(0 / 0)
        image_num.append(counter)

    final_embedding = np.array(final_embedding)
    print("Embedding Shape: ", final_embedding.shape)
    image_num, case_id, labels, final_embedding = shuffle(image_num, case_id, labels, final_embedding)
    print("shuffled!")
    image_num = np.array(image_num)
    image_num = np.reshape(image_num, (image_num.shape[0], 1))
    labels = np.array(labels)
    labels = np.reshape(labels, (labels.shape[0], 1))
    case_id = np.array(case_id)
    case_id = np.reshape(case_id, (case_id.shape[0], 1))
    return image_num, case_id, labels, final_embedding


def evaluate_NN(model, resolution, image_num_validation, case_id_validation, labels_validation, embedding_validation):
    predictions = model.predict(embedding_validation)
    y_pred = np.array(predictions)

    print("===================================================")
    print("Evaluation results for ", resolution)
    print("===================================================")
    print("Prediction = 0.5")
    pos = []
    neg = []
    for i in range(len(labels_validation)):
        if labels_validation[i] == 1:
            pos.append(y_pred[i])
        else:
            neg.append(y_pred[i])
    print("Avg Pos:", sum(pos) / len(pos))
    print("Avg Neg:", sum(neg) / len(neg))

    s = (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2

    y_pred_f = (y_pred >= 0.5).astype(int)

    cm = confusion_matrix(labels_validation, y_pred_f)
    accuracy = (cm[0][0] + cm[1][1]) / len(y_pred_f)

    # plot_tsne(embedding_validation, labels_validation, y_pred_f, "validation_" + resolution)
    print("**********************************")
    print("Number of case IDs: ", len(y_pred_f))
    print('Accuracy: %.2f' % accuracy)
    print("confusion matrix: ", cm)
    print("===================================================")

    best_acc = 0
    all_accs = []
    all_cms = []
    cutoff = 0
    print("Itereate over cutoff values..")
    for i in range(0, 100):
        best_acc, acc, cm = result_by_patient(y_pred, labels_validation, case_id_validation, False,
                                              image_num_validation,
                                              i / 100.0, best_acc)
        all_accs.append(acc)
        all_cms.append(cm)
        if best_acc == acc:
            cutoff = i
    global tn_tmp
    global fn_tmp
    global tp_tmp
    global fp_tmp
    global tn_sum
    global fn_sum
    global tp_sum
    global fp_sum
    tn_sum += tn_tmp
    fn_sum += fn_tmp
    tp_sum += tp_tmp
    fp_sum += fp_tmp
    return best_acc, all_accs, cutoff, all_cms


def train_NN(model, resolution, image_num_train, case_id_train, labels_train, embedding_train, labels_validation,
             embedding_validation):
    es1 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    # mc1 = ModelCheckpoint(save_path + 'best_model_' + resolution + '.h5', monitor='val_accuracy', mode='max', verbose=1,
    #                       save_best_only=True)

    history = model.fit(embedding_train, labels_train, epochs=40, batch_size=32,
                        validation_data=(embedding_validation, labels_validation), callbacks=[es1])

    _, accuracy = model.evaluate(embedding_train, labels_train)
    predictions = np.array(model.predict(embedding_train))
    print("===================================================")
    print("Prediction = 0.5")
    predictions_f = (predictions >= 0.5).astype(int)

    pos = []
    neg = []
    for i in range(len(labels_train)):
        if labels_train[i] == 1:
            pos.append(predictions[i])
        else:
            neg.append(predictions[i])
    print("Avg Pos:", sum(pos) / len(pos))
    print("Avg Neg:", sum(neg) / len(neg))

    best_acc = 0
    all_accs = []
    cutoff = 0
    all_cm = []
    print("Itereate over cutoff values..")
    for i in range(0, 100):
        # print("train result by patient: ", i / 100.0)
        best_acc, acc, cm = result_by_patient(predictions, labels_train, case_id_train, True, image_num_train,
                                              i / 100.0, best_acc)
        all_accs.append(acc)
        all_cm.append(cm)
        if best_acc == acc:
            cutoff = i
    return best_acc, all_accs, cutoff, all_cm, model



p1 = "./train_40x_HR2_sampled_200_128x128_ALL_pretrained_noidea.npy"
p2 = "./train_40x_HR_sampled_200_128x128_ALL_pretrained_noidea.csv"
image_num_train, case_id_train, labels_train, embedding_train = read_data(project_path, p1, p2)

p1 = "./validation_40x_HR2_sampled_200_128x128_ALL_pretrained_noidea.npy"
p2 = "./validation_40x_HR_sampled_200_128x128_ALL_pretrained_noidea.csv"
image_num_validation, case_id_validation, labels_validation, embedding_validation = read_data(project_path, p1, p2)

all_embedding = np.vstack((embedding_train, embedding_validation))
all_image_num = np.vstack((image_num_train, image_num_validation))
all_case_id = np.vstack((case_id_train, case_id_validation))
all_labels = np.vstack((labels_train, labels_validation))

df = pd.DataFrame({"caseID": list(all_case_id[:, 0]), "label": list(all_labels[:, 0])})
df2 = df.drop_duplicates()

def calculate_acc(all_acc_list, index, final_acc_list):
    accuracy = all_acc_list[index]
    if accuracy < 0.50:
        final_acc_list.append(50)
    else:
        final_acc_list.append(accuracy * 100)
    return


def save_results(list_accs, path_name):
    data = {}
    for i in range(len(list_accs)):
        data["val_acc_" + str(i)] = list_accs[i]

    df = pd.DataFrame.from_dict(data)
    df.to_csv(path_name + str(num) + ".csv", index=False)


val_acc_all = [[] for i in range(100)]
j = 0
kfold_caseid_validation = []
for train_index, test_index in skf.split(df2["caseID"], df2["label"]):
    print("fold:", j)
    j += 1
    model_NN1 = create_baseline1()
    model_NN1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    case_ids_set_train = df2.iloc[train_index]["caseID"]
    case_ids_set_validation = df2.iloc[test_index]["caseID"]
    bootstrap_image_num_train = []
    bootstrap_case_id_train = []
    bootstrap_labels_train = []
    bootstrap_embedding_train = []
    bootstrap_image_num_validation = []
    bootstrap_case_id_validation = []
    bootstrap_labels_validation = []
    bootstrap_embedding_validation = []
    case_ids_set_validation = list(case_ids_set_validation[:])
    kfold_caseid_validation.append(case_ids_set_validation)
    for i in range(all_case_id.shape[0]):
        if  all_case_id[i, 0] in case_ids_set_validation:
            bootstrap_image_num_validation.append(all_image_num[i, 0])
            bootstrap_case_id_validation.append(all_case_id[i, 0])
            bootstrap_labels_validation.append(all_labels[i, 0])
            bootstrap_embedding_validation.append(all_embedding[i])
        else:
            bootstrap_image_num_train.append(all_image_num[i, 0])
            bootstrap_case_id_train.append(all_case_id[i, 0])
            bootstrap_labels_train.append(all_labels[i, 0])
            bootstrap_embedding_train.append(all_embedding[i])

    bootstrap_image_num_train = np.array(bootstrap_image_num_train, dtype=np.int64)
    bootstrap_case_id_train = np.array(bootstrap_case_id_train)
    bootstrap_labels_train = np.array(bootstrap_labels_train, dtype=np.int64)
    bootstrap_embedding_train = np.array(bootstrap_embedding_train)
    bootstrap_image_num_validation = np.array(bootstrap_image_num_validation, dtype=np.int64)
    bootstrap_case_id_validation = np.array(bootstrap_case_id_validation)
    bootstrap_labels_validation = np.array(bootstrap_labels_validation, dtype=np.int64)
    bootstrap_embedding_validation = np.array(bootstrap_embedding_validation)
    best_accuracy, all_accuracys, cutoff, all_cms, trained_model = train_NN(model_NN1, "40x", bootstrap_image_num_train,
                                                                            bootstrap_case_id_train,
                                                                            bootstrap_labels_train,
                                                                            bootstrap_embedding_train,
                                                                            bootstrap_labels_validation,
                                                                            bootstrap_embedding_validation)
    best_accuracy2, all_accuracys2, cutoff2, all_cms2 = evaluate_NN(trained_model, "40x",
                                                                    bootstrap_image_num_validation,
                                                                    bootstrap_case_id_validation,
                                                                    bootstrap_labels_validation,
                                                                    bootstrap_embedding_validation)


    for i in range(100):
        calculate_acc(all_accuracys2, i, val_acc_all[i])

    save_results(val_acc_all, "./embd_accs")




tmp = pd.read_csv(open("./embd_accs" + str(num) + ".csv", 'rb'))
best  = 0
f_e = 0
f_c = 0
all_avgs = []
tags = []
for c in range(100):
    l = tmp['val_acc_'+ str(c)]
    avg, e = compute_CI(l)
    if avg >= best:
      best = avg
      f_e = e
      f_c = 'val_acc_'+ str(c)
    tags.append(str(c))
    all_avgs.append(avg)

print("======================================")
print("======================================")
print("Best for " , num)
print(best, e, f_c)



print("======================")
print("TP: ", tp_sum)
print("TN: ", tn_sum)
print("FP: ", fp_sum)
print("FN: ", fn_sum)
