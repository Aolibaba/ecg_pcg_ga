
from __future__ import print_function



import argparse
import numpy as np
import keras
import os

import load
import util
import evaluate

def predict(data_json, model_path):
    preproc = util.load(os.path.dirname(model_path))
    dataset = load.load_dataset(data_json)

    x, y = preproc.process(*dataset)
    y_test = []
    for e,i in enumerate(dataset[1]):
        for j in range(len(i)):
            y_test.append(y[e,j,:])
    y_result=np.array(y_test)

    model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)
    #update start
    y_test = []
    y_predict=[]
    for e,i in enumerate(dataset[1]):
        for j in range(len(i)):
            y_test.append(y[e,j,:])
            y_predict.append(probs[e,j,:])
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    #update stop

    return y_test , y_predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", help="path to data json")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    y_test,y_score = predict(args.data_json, args.model_path)
    evaluate.roc_plot(y_test,y_score)
    acc_val = evaluate.acc_cal(y_test,y_score)
    print("over")