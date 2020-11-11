import numpy as np
import scipy.io as scio

from sklearn.metrics import roc_curve,auc


def load_pcg_auto(path):
    train = scio.loadmat(path + 'pcg_train.mat')
    test = scio.loadmat(path + 'pcg_test.mat')
    x_train  = train['x']
    #x_train = x_train.reshape([x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
    x_test  = test['x']
    #x_test = x_test.reshape([x_test.shape[0], x_test.shape[1]*x_test.shape[2]])
    return x_train,x_test


def load_ecg_auto(path):
    train = scio.loadmat(path + 'ecg_train.mat')
    test = scio.loadmat(path + 'ecg_test.mat')
    x_train, y_train = train['x'],train['y']
    #x_train = x_train.reshape([x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
    x_test,y_test = test['x'], test['y']
    #x_test = x_test.reshape([x_test.shape[0], x_test.shape[1]*x_test.shape[2]])
    return x_train,y_train,x_test,y_test

def load_data(params):

    x_ecg_train, y_ecg_train, x_ecg_test, y_ecg_test = load_ecg_auto(params['ecg_path'])
    x_pcg_train, x_pcg_test = load_pcg_auto(params['pcg_path'])

    x_train = np.hstack([x_ecg_train,x_pcg_train])
    x_test  = np.hstack([x_ecg_test,x_pcg_test])
    return   x_train, y_ecg_train, x_test, y_ecg_test



def draw_roc(y,y_pre):
    fpr, tpr, thresholds = roc_curve(y, y_pre, pos_label=None, sample_weight=None,drop_intermediate=True)
    auc_area = auc(fpr, tpr)
    #plt.plot(fpr, tpr, lw=1, label='(area = %0.2f)'%auc_area)
    #plt.show()
    return auc_area

if __name__=="__main__":
    print('over')
