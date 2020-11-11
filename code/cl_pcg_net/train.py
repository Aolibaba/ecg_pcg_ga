from function import specificity,cal_auc,LossHistory
import network
import load
import util
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.models import Model
import scipy.io as scio
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score,precision_score,f1_score

MAX_EPOCHS = 160
batch_size = 32
if __name__ == '__main__':
    params = util.config()
    save_dir = params['save_dir']

    print("Loading training set...")
    train = load.load_dataset(params['train'])
    print("Loading dev set...")
    dev = load.load_dataset(params['dev'])
    print("Building preprocessor...")
    preproc = load.Preproc(*train)
    print("Training size: " + str(len(train[0])) + " examples.")
    print("Dev size: " + str(len(dev[0])) + " examples.")

    params.update({
        "input_shape": [8000, 1],
        "num_categories": len(preproc.classes)
    })

    #create the cl-pcg-net
    model = network.build_network(**params)

    #learning rate reduce strategy
    def scheduler(epoch):
        if epoch % 80 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            model.load_weights(save_dir + 'best.hdf5')
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model.optimizer.lr)
    reduce_lr = LearningRateScheduler(scheduler)

    #choose best model to save
    checkpointer = keras.callbacks.ModelCheckpoint(
        mode='max',
        monitor='val_acc',
        filepath=save_dir + 'best.hdf5',
        save_best_only=True)

    #variable to save the loss_acc_iter value
    history = LossHistory()

    #data generator
    train_gen = load.data_generator(batch_size, preproc, *train)
    dev_gen = load.data_generator(len(dev[0]), preproc, *dev)

    #fit the model
    model.fit_generator(
        train_gen,
        steps_per_epoch=int(len(train[0]) / batch_size),
        epochs=MAX_EPOCHS,
        validation_data=dev_gen,
        validation_steps=1,
        verbose=False,
        callbacks=[checkpointer, reduce_lr, history])

    #save loss_acc_iter
    history.save_result(params['save_dir'] + 'loss_acc_iter.mat')

    #extract and save deep coding features
    x_train, y_train = load.data_generator2(preproc,*train)
    x, y_t = load.data_generator2(preproc,*dev)
    model.load_weights(save_dir + 'best.hdf5')
    new_model = Model(inputs=model.input, outputs=model.layers[-3].output)
    feature_train = new_model.predict(x_train)
    feature_test = new_model.predict(x)
    scio.savemat(save_dir +  'pcg_train.mat', {'x': feature_train, 'y':y_train})
    scio.savemat(save_dir +  'pcg_test.mat', {'x': feature_test, 'y':y_t})
    print('deep coding features of pcg saved')

    #evaluate model
    y_p = model.predict(x)
    print(confusion_matrix(y_t.argmax(1), y_p.argmax(1)))
    print('sensitivity:', recall_score(y_t.argmax(1), y_p.argmax(1)))
    print('specificity:', specificity(y_t.argmax(1), y_p.argmax(1)))
    print('f1-score:', f1_score(y_t.argmax(1), y_p.argmax(1)))
    print('accuracy:', accuracy_score(y_t.argmax(1), y_p.argmax(1)))
    print('roc:', cal_auc(y_t.argmax(1), y_p[:, 1]))
    #evaluate.evaluate(save_dir)
