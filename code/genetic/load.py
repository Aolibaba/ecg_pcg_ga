from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import keras
import numpy as np
import os
import random
import scipy.io as sio
import tqdm
from sklearn.preprocessing import StandardScaler
STEP = 256
def batch_generate(batch_size, x, y):
    batch_size = int(batch_size/2)
    examples = zip(x, y)
    positive = []
    negative = []
    index = 0
    for label in y:
        if label[0] == '1':
            positive.append(index)
        else:
            negative.append(index)
        index = index+1
    set_positive = [examples[i] for i in positive]
    set_negative = [examples[i] for i in negative]
    batches_positive = [set_positive[i:i+batch_size]
                        for i in range(0, len(set_positive)-batch_size+1, batch_size)]
    [batches_positive[-1].append(i) for i in set_positive[i+batch_size:]]
    index = 0
    batches_negative = []
    for batch_positive in batches_positive:
        batch_size = len(batch_positive)
        if (index + batch_size)<len(set_negative):
            batches_negative.append(set_negative[index:index+batch_size])
            index = index +batch_size
        else:
            temp = set_negative[index:]
            index = len(set_negative)-index
            batch_negative = temp + set_negative[:index]
            batches_negative.append(batch_negative)
    batches= []
    for i in range(len(batches_positive)):
        batches.append(batches_positive[i]+batches_negative[i])
    return batches

def data_generator(batch_size, preproc, x_signal,x_feature, label):
    batches_ecg = batch_generate(batch_size, x_signal, label)
    batches_pcg = batch_generate(batch_size, x_feature, label)
    while True:
        for index in range(len(batches_ecg)):
            x_signal1, y_signal = zip(*batches_ecg[index])
            x_signal2, y_signal = zip(*batches_pcg[index])
            x1, y1 = preproc.process(x_signal1, y_signal)
            x2 = np.array(x_signal2)
            scaler = StandardScaler()
            x2 = scaler.fit_transform(x2)
            #x = {'ecg_inputs': x1,'pcg_inputs': x2}
            x = {'ecg_inputs': x1}
            y = {'outputs': y1}
            yield x, y
def data_generator_dev(batch_size, preproc, x_signal,x_feature,label):
    while True:
        x1, y1=preproc.process(x_signal, label)
        x2 = np.array(x_feature)
        scaler = StandardScaler()
        x2 = scaler.fit_transform(x2)
        #x = {'ecg_inputs': x1,'pcg_inputs': x2}
        x={'ecg_inputs': x1}
        y = {'outputs': y1}
        yield x, y
def load_predict_batch(batch_size,x,y):
    num_examples=len(x)
    examples=zip(x, y)
    #examples=sorted(examples, key=lambda x: x[0].shape[0])
    end=num_examples - batch_size + 1
    batches=[examples[i:i + batch_size]
             for i in range(0, end, batch_size)]
    [batches[-1].append(i) for i in examples[i + batch_size:]]
    return batches

class Preproc:

    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        #self.classes.append(unicode(0))
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x

    def process_y(self, y):
        # TODO, awni, fix hack pad with noise for cinc
        y=pad([[self.class_to_int[c] for c in s] for s in y], val=1, dtype=np.int32)
        y = keras.utils.np_utils.to_categorical(
                y, num_classes=len(self.classes))
        return y

def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded

def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32),
           np.std(x).astype(np.float32))

def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []; ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels

def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]

def load_dataset2(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []; ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg2(d['ecg']))
    return ecgs, labels

def load_ecg2(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['features'].squeeze()
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)
    return ecg