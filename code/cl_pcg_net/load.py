import json
import keras
import numpy as np
import random
import scipy.io as sio
import tqdm

def data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = list(zip(x, y))
    random.shuffle(examples)
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    while True:
        for batch in batches:
            x_bat, y_bat = zip(*batch)
            x_bat = np.array(x_bat)
            input = []
            for i in range(4):
                tmp = x_bat[:,:,i]
                tmp = tmp[:,:,np.newaxis]
                input.append(tmp)
            y_bat = preproc.process_y(y_bat)
            yield input, y_bat

def data_generator2(preproc, x, y):
    x = np.array(x)
    input = []
    for i in range(4):
        tmp = x[:, :, i]
        tmp = tmp[:, :, np.newaxis]
        input.append(tmp)
    y = preproc.process_y(y)
    return input, y

class Preproc:

    def __init__(self, pcg, labels):

        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        return np.array(x)

    def process_y(self, y):
        y = pad([[self.class_to_int[c] for c in s] for s in y], val=1, dtype=np.int32)
        y = keras.utils.np_utils.to_categorical(
                y, num_classes=len(self.classes))
        return y

    def process_y1(self,y):
        y = [[self.class_to_int[c] for c in s] for s in y]
        y = np.array(y)
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
    labels = []; pcgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        pcgs.append(load_pcg(d['ecg']))
    return pcgs, labels

def load_pcg(record):
    pcg = sio.loadmat(record)['val'].squeeze()
    return pcg

