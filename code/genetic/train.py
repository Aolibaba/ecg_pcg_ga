import tqdm
import random
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import function
import util

def svm_lpp():
    clf = svm.SVC(kernel='linear',probability=True,C=1)
    return clf

def specificity(y_true,y_test):
    TN = 0
    FP = 0
    for i in range(0,y_true.shape[0]):
        if int(y_true[i]) == 0 and int(y_test[i]) == 0:
            TN = TN+1
        if int(y_true[i]) == 0 and int(y_test[i]) == 1:
            FP = FP+1
    return  float(TN)/(TN+FP)
def sensitivity(y_true,y_test):
    FN = 0
    TP = 0
    for i in range(0,y_true.shape[0]):
        if int(y_true[i]) == 1 and int(y_test[i]) == 0:
            FN = FN+1
        if int(y_true[i]) == 1 and int(y_test[i]) == 1:
            TP = TP+1
    return  float(TP)/(FN+TP)
def F1(y_true,y_test):
    spe = specificity(y_true,y_test)
    sen = sensitivity(y_true,y_test)
    return (2*spe*sen) / (spe+sen)


def fea_gene_generator(gene,x_pcg):
    gene_index = []
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
            if list1[i] != list2[i]:
                return False
    return True

def caculate_fitness(x_train,y_train,x_test,y_test):
    y_train = y_train.argmax(1)
    y_test = y_test.argmax(1)
    clf = svm_lpp().fit(x_train, y_train)
    y_test_score = clf.decision_function(x_test)
    roc_auc = function.draw_roc(y_test, y_test_score)
    return roc_auc

def caculate_fitness_output(x_train,y_train,x_test,y_test):
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

    roc_auc = function.draw_roc(y_test, test_predict_proba)
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
    # ******************************1.parameters setting***************************
    feature_num = 128 #number of features: ecg+pcg 64 + 64
    group_num = 400 #number of feature subset in population
    best_feature = []#the optimal subset
    best_score = 0 #the score of optimal subset
    times = 0
    group = [[random.randint(1,2) for i in range(feature_num)] for j in range(group_num)]#生成初始种群
    for each1 in range(len(group)):
        for each2 in range(len(group[each1])):
            if group[each1][each2]>0:
                group[each1][each2]=1
    # ******************************  2.data loading  ***************************
    x_train,y_train,x_test,y_test = function.load_data(params)
    #scaler.fit_transform(np.vstack([x_train,x_test]))
    # ****************************** 3.genetic algorithm
    schedule = range(0,100,1)
    for sch in tqdm.tqdm(schedule):
        score = []
        for i in range(group_num):
            x_train_tmp = fea_gene_generator(group[i], x_train)
            x_test_tmp = fea_gene_generator(group[i], x_test)
            score.append(caculate_fitness(x_train_tmp, y_train, x_test_tmp, y_test))
        group_score = []
        for index,score in enumerate(score):
            group_score.append([index,score])
        #Sort by the score of each subset
        group_score = sorted(group_score, key = lambda x: x[1],reverse=True)
        #optimal subset
        best_score = group_score[0][1]
        #the optimal quarter of the population
        new_group_half1 = [group[i[0]] for i in group_score[0:int(group_num/4)]]
        #generate the rest three quarters to complete the population by using the optimal quarter. cross operation.
        new_group_half2 = cross(new_group_half1,int(group_num*3/4))
        group = new_group_half1+new_group_half2
        #variation opration.
        group = variation(group)
        #output the evolution states。
        print(times, best_score, compare(best_feature, group[0]))
        times = times + 1
    best_feature = group[0]
    x_train_tmp = fea_gene_generator(best_feature, x_train)
    x_test_tmp = fea_gene_generator(best_feature, x_test)
    caculate_fitness_output(x_train_tmp, y_train, x_test_tmp, y_test)

