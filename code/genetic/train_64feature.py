#coding=utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tqdm

import random
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score,precision_score,f1_score,roc_curve

from sklearn.preprocessing import MinMaxScaler

from sklearn import svm
import function_lpp_new
import util
import scipy.io as scio
import xlwt
from sklearn.decomposition import PCA
MAX_EPOCHS = 500
def svm_lpp():
    #parameters={'kernel': ['linear'], 'C': [0.5, 1, 2,10]}
    #svr=svm.SVC(probability=True,class_weight='balanced')
    #clf=GridSearchCV(svr, parameters, cv=3,iid=True)
    clf = svm.SVC(kernel='linear',probability=True,C=1)
    return clf
def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)

    i=0
    for data in datas:
        for j in range(len(data)):
            sheet1.write(i, j, data[j])
        i =i + 1
    f.save(file_path)
def specificity(y_true,y_test):
    TN=0
    FP=0
    for i in range(0,y_true.shape[0]):
        if int(y_true[i])== 0 and int(y_test[i])==0:
            TN = TN+1
        if int(y_true[i])==0 and int(y_test[i])==1:
            FP =FP+1
    return  float(TN)/(TN+FP)
def sensitivity(y_true,y_test):
    FN=0
    TP=0
    for i in range(0,y_true.shape[0]):
        if int(y_true[i])== 1 and int(y_test[i])==0:
            FN = FN+1
        if int(y_true[i])==1 and int(y_test[i])==1:
            TP =TP+1
    return  float(TP)/(FN+TP)
def F1(y_true,y_test):
    spe = specificity(y_true,y_test)
    sen = sensitivity(y_true,y_test)
    return (2*spe*sen) / (spe+sen)
def label_trans(label_pro, bound):
    label = []
    for i in range(label_pro.shape[0]):
        if label_pro[i]>=bound:
            label.append(1)
        else:
            label.append(0)
    label = np.array(label)
    return label
def wrapper_fea(X_train,X_test, y):
    estimator = svm.SVR(kernel="linear")
    selector = RFE(estimator, 10, step=1)
    selector = selector.fit(X_train, y)
    elements = selector.support_
    index = np.argwhere(elements == True)
    index = index.squeeze(axis=-1).tolist()
    fea_train = X_train[:,index]
    fea_test = X_test[:,index]
    return fea_train,fea_test

def fea_gene_generator(x_train,x_test,i):
    estimator = PCA(n_components=i)
    estimator.fit(x_train)
    x_train = estimator.transform(x_train)
    x_test = estimator.transform(x_test)
    return x_train,x_test
def cross(new_group_half1, num):
    group = []
    for i in range(num):
        gene1 = new_group_half1[random.randint(0,len(new_group_half1)-1)]
        gene2 = new_group_half1[random.randint(0,len(new_group_half1)-1)]
        gene = gene1[0:int(len(gene1)/2)] + gene2[int(len(gene1)/2):]
        group.append(gene)
    return group
def variation(group):
    for i in range(0,int(len(group)*0.2)):
        index1 = random.randint(1,len(group)-1)
        for j in range(0,int(len(group[0])*0.2)):
            index2 = random.randint(0,len(group[0])-1)
            group[index1][index2] = abs(group[index1][index2]-1)
    return group
def compare(list1,list2):
    for i in range(len(list1)):
            if list1[i] != list2[i]:  # 元素不相等时
                return False
    return True
def obtain_flod_predict(x_train,y_train,x_test,y_test):
    y_train = y_train.argmax(1)
    y_test = y_test.argmax(1)
    clf = svm_lpp().fit(x_train, y_train)
    y_predict_score = clf.decision_function(x_test)
    y_predict = clf.predict(x_test)
            #fpr, tpr, threshold = roc_curve(y_test, y_predict)
    return [y_test,y_predict_score,y_predict]


def caculate_fitness(x_train,y_train,x_test,y_test):
    y_train = y_train.argmax(1)
    y_test = y_test.argmax(1)
    clf = svm_lpp().fit(x_train, y_train)
    y_test_score = clf.decision_function(x_test)
    #acc = accuracy_score(y_test, clf.predict(x_test))
    #f1 = F1(y_test, clf.predict(x_test))
    #acc = accuracy_score(y_test, clf.predict(x_test))
    roc_auc = function_lpp_new.draw_roc(y_test, y_test_score)
    return roc_auc

def caculate_fitness_plot(x_train,y_train,x_test,y_test):
    y_train = y_train.argmax(1)
    y_test = y_test.argmax(1)
    clf = svm_lpp().fit(x_train, y_train)
    train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)
    test_predict_proba = clf.decision_function(x_test)

    print(confusion_matrix(y_test, test_predict))
    print('train sensitivity:',sensitivity(y_train,train_predict),'train specifity:', specificity(y_train, train_predict))
    print('test sensitivity:',sensitivity(y_test,test_predict),'test specifity:', specificity(y_test, test_predict))
    print('f1_score:', F1(y_test, test_predict))
    print('acc:', accuracy_score(y_test,test_predict))

    roc_auc = function_lpp_new.draw_roc(y_test, test_predict_proba)
    print('roc:', roc_auc)

    print('over')
if __name__ == '__main__':

    params = util.config()
    combine = list(zip(params['ecg_path_all'],params['pcg_path_all']))
    combine_index = 0
    for lll,j in enumerate(combine):

        params['ecg_path'] = j[0]
        params['pcg_path'] = j[1]


        scaler = MinMaxScaler()

        save_true = []
        save_predict = []
        y_predict_label = []
        result_eva = []
        tmp = ['sensitivity:', 'specificity:', 'precision', 'f1-score:', 'accuracy:', 'roc:']
        result_eva.append(tmp)
        # ******************************  1.载入数据  ***************************
            #fold = 0#五折交叉验证中的第几折
        for fold in range(0,5):
            feature_num = 128 #特征个数32+79=111 64+79=143
            group_num = 128#生成初始种群 200个
            best_score = 0#最好的分
            best_feature = []#最好特征
            times = 0

            # ******************************  2.载入数据  ***************************
            x_train,y_train,x_test,y_test = function_lpp_new.load_data(params, fold)
            scaler.fit_transform(np.vstack([x_train,x_test]))
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            # ****************************** 3.PCA
            score = 0
            index = 0
            for pca_i in range(1,129):
                x_train_pca,x_test_pca = fea_gene_generator(x_train, x_test, pca_i)
                pca_score = caculate_fitness(x_train_pca, y_train, x_test_pca, y_test)
                if pca_score>score:
                    index = pca_i
                    score = pca_score
            x_train_pca, x_test_pca=fea_gene_generator(x_train, x_test, index)
            result = obtain_flod_predict(x_train_pca, y_train, x_test_pca, y_test)
            save_true.append(result[0])
            save_predict.append(result[1])
            y_predict_label.append(result[2])
            print(str(fold), 'over')

        for i in range(5):
            result_eva.append(
                [recall_score(save_true[i], y_predict_label[i]), specificity(save_true[i], y_predict_label[i]),
                 precision_score(save_true[i], y_predict_label[i]), f1_score(save_true[i], y_predict_label[i]),
                 accuracy_score(save_true[i], y_predict_label[i]),
                 function_lpp_new.draw_roc(save_true[i], save_predict[i])])

        for i in range(1,5):
            save_true[0] = np.hstack([save_true[0],save_true[i]])
            save_predict[0] = np.hstack([save_predict[0],save_predict[i]])
            y_predict_label[0] = np.hstack([y_predict_label[0],y_predict_label[i]])
        save_true = save_true[0]
        save_predict = save_predict[0]

        y_predict_label = y_predict_label[0]
        scio.savemat(params['path_PCA']+str(lll)+'predict_genetic_multi.mat',{'y_predict':save_predict,'y_true':save_true})
        print('*******final********')
        print(confusion_matrix(save_true, y_predict_label))
        print('sensitivity:', recall_score(save_true, y_predict_label))
        print('specificity:', specificity(save_true, y_predict_label))
        print('precision', precision_score(save_true, y_predict_label))
        print('f1-score:', f1_score(save_true, y_predict_label))
        print('accuracy:', accuracy_score(save_true, y_predict_label))
        print('auc:', function_lpp_new.draw_roc(save_true, save_predict))

        result_eva.append([recall_score(save_true, y_predict_label), specificity(save_true, y_predict_label),
                           precision_score(save_true, y_predict_label), f1_score(save_true, y_predict_label),
                           accuracy_score(save_true, y_predict_label), function_lpp_new.draw_roc(save_true, save_predict)])
        data_write(params['path_PCA'] +str(lll)+ 'evaluate_genetic_multi.xls', result_eva)
