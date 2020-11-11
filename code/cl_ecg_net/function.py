import keras
import numpy as np
import scipy.io as scio
from sklearn.metrics import roc_curve,auc

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = {'batch': [], 'epoch': []}
        self.acc = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.loss['batch'].append(logs.get('loss'))
        self.acc['batch'].append(logs.get('acc'))
        #self.val_loss['batch'].append(logs.get('val_loss'))
        #self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.loss['epoch'].append(logs.get('loss'))
        self.acc['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.loss[loss_type]))


    def save_result(self,path):
        loss_batch = np.array(self.loss['batch'])
        loss_epoch = np.array(self.loss['epoch'])
        acc_batch = np.array(self.acc['batch'])
        acc_epoch = np.array(self.acc['epoch'])

        val_loss_epoch = np.array(self.val_loss['epoch'])

        val_acc_epoch = np.array(self.val_acc['epoch'])

        scio.savemat(path, {'loss_batch': loss_batch,'loss_epoch':loss_epoch,'acc_batch':acc_batch,'acc_epoch':acc_epoch,
                            'val_loss_epoch':val_loss_epoch, 'val_acc_epoch':val_acc_epoch})

def specificity(y_true,y_test):
    TN = 0
    FP = 0
    for i in range(0,y_true.shape[0]):
        if int(y_true[i]) == 0 and int(y_test[i]) == 0:
            TN = TN + 1
        if int(y_true[i]) == 0 and int(y_test[i]) == 1:
            FP = FP + 1
    return  float(TN)/(TN+FP)

def cal_auc(y,y_pre):
    fpr, tpr, thresholds = roc_curve(y, y_pre, pos_label=None, sample_weight=None,drop_intermediate=True)
    auc_area = auc(fpr, tpr)
    return auc_area