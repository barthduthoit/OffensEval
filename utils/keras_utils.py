from keras.callbacks import Callback
import keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax((np.asarray(self.model.predict(self.validation_data[0]))), axis=1)
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        print(" — val_macro_f1: {}".format(_val_f1), average="macro")
        
        
def f1_loss(y_true, y_pred):    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred),  'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)
    
    