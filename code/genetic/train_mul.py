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

def fea_gene_generator(gene,x_pcg):
    gene_index = []
    #获得基因为1的索引
    for index in range(len(gene)):
        if gene[index]!=0:
            gene_index.append(index)

    x = x_pcg[:,gene_index]
    return x
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


    scaler = MinMaxScaler()

    save_true = []
    save_predict = []
    y_predict_label = []
    result_eva = []
    tmp = ['sensitivity:', 'specificity:', 'precision', 'f1-score:', 'accuracy:', 'roc:']
    result_eva.append(tmp)
    # ******************************  1.载入数据  ***************************
        #fold = 0#五折交叉验证中的第几折
    hot_map = []
    #for fold in range(0,5):
    feature_num = 128 #特征个数64+64
    group_num = 400#生成初始种群 200个
    best_score = 0#最好的分
    best_feature = []#最好特征
    times = 0
    group = [[random.randint(1,2) for i in range(feature_num)] for j in range(group_num)]#生成初始种群
    for each1 in range(len(group)):
        for each2 in range(len(group[each1])):
            if group[each1][each2]>0:
                group[each1][each2]=1
    # ******************************  2.载入数据  ***************************
    x_train,y_train,x_test,y_test = function_lpp_new.load_data(params)
    #x_train = x_train[:,0:64]
    #x_test = x_test[:,0:64]
    scaler.fit_transform(np.vstack([x_train,x_test]))
    #x_train = scaler.transform(x_train)
    #x_test = scaler.transform(x_test)
    # ****************************** 3.遗传算法
    #为每个种群打分
    schedule = range(0,10,1)
    for sch in tqdm.tqdm(schedule):
        score = []
        for i in range(group_num):
            x_train_tmp = fea_gene_generator(group[i], x_train)
            x_test_tmp = fea_gene_generator(group[i], x_test)
            score.append(caculate_fitness(x_train_tmp, y_train, x_test_tmp, y_test))

        group_score = []
        for index,score in enumerate(score):
            group_score.append([index,score])
        #根据每个种群的分数排序
        group_score = sorted(group_score, key = lambda x: x[1],reverse=True)
        hot_map.append(group_score)
        #最好的分数
        best_score = group_score[0][1]
        #取前四分之一最好的
        new_group_half1 = [group[i[0]] for i in group_score[0:int(group_num/4)]]
        #交叉产生后四分之三
        new_group_half2 = cross(new_group_half1,int(group_num*3/4))
        #相加产生所有种群
        group = new_group_half1+new_group_half2
        #对种群进行变异。最好的一个基因组不变
        group = variation(group)
        print(times, best_score, compare(best_feature, group[0]))

    best_feature = group[0]
    #print(best_feature)
        #times = times + 1
    x_train_tmp = fea_gene_generator(best_feature, x_train)
    x_test_tmp = fea_gene_generator(best_feature, x_test)
    caculate_fitness_plot(x_train_tmp, y_train, x_test_tmp, y_test)
    #result = obtain_flod_predict(x_train_tmp, y_train, x_test_tmp, y_test)
    #save_true.append(result[0])
    #save_predict.append(result[1])
