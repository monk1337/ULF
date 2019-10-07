# Research_paper
# https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8527

import numpy as np
import sklearn.metrics.pairwise as pairwise
from sklearn import preprocessing
from keract import get_activations
import tensorflow as tf

import keras
import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout,Activation
from keras.optimizers import Adam, Nadam, RMSprop, Adadelta
from keras import regularizers

class GraphClustering(object):

    def __init__(
                self,
                dataframe,
                epoch = 100,
                ):
    
        self.dataframe   = dataframe
        self.epoch       = epoch

        self.first_layer  = self.dataframe.shape[0]
        self.second_layer = int(self.first_layer*0.7)
        self.third_layer  = int(self.second_layer*0.5)

        
        self.TModel = Sequential()

        self.TModel.add(Dense(
                         self.first_layer, 
                         input_dim = self.first_layer, 
                         name= 'first', 
                         activation='sigmoid')
                        )
        # TModel.add(Dropout(0.8))

        self.TModel.add(Dense(self.second_layer, name='second', activation='sigmoid'))
        # TModel.add(Dropout(0.8))

        self.TModel.add(Dense(self.third_layer, name='embed', activation='sigmoid'))
        # TModel.add(Dropout(0.8))

        self.TModel.add(Dense(self.second_layer, name='four', activation='sigmoid'))
        # TModel.add(Dropout(0.8))

        self.TModel.add(Dense(self.first_layer, name='five', activation='sigmoid'))
        # TModel.add(Dropout(0.8))

    
    def preprocessing(self, data):

        # x = normalize(x)
        min_max_scaler = preprocessing.MinMaxScaler()
        x = min_max_scaler.fit_transform(X=data)

        # # Compute the rbf (gaussian) kernel between X and Y:
        # # K(x, y) = exp(-gamma ||x-y||^2)
        # xtrain = pairwise.rbf_kernel(x, x, gamma=1.0/(2*490))
        xtrain = pairwise.cosine_similarity(x, x)

        D = np.diag(1.0 / np.sqrt(xtrain.sum(axis=1)))
        train_data = D.dot(xtrain).dot(D)

        return train_data


    def sae_square_loss(self, beta, p):

        def layer_activations(layername):
            return tf.reduce_mean(self.TModel.get_layer(layername).output, axis=0)

        def sparse_result(rho, layername):
            rho_hat = layer_activations(layername)
            return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

        def KL(p):
            First = tf.reduce_sum(sparse_result(p, 'first'))
            Second = tf.reduce_sum(sparse_result(p, 'second'))
            Embed = tf.reduce_sum(sparse_result(p, 'embed'))
            return First + Second + Embed

        def loss(y_true, y_pred):
            # res = (K.sum(K.l2_normalize(y_true - y_pred))) + beta*(p * K.log(p / K.mean(activations)) + (1.0 - p)*K.log((1.0-p)/(1.0-K.mean(activations))))
            # res =  K.sqrt( K.sum( (y_true - y_pred)**2 ) ) + beta*(p * K.log(p / K.mean(activations)) + (1.0 - p)*K.log((1.0-p)/(1.0-K.mean(activations))))
            # res = K.sqrt(K.sum((y_true - y_pred)**2)) + beta * KL(p)
            res = tf.reduce_mean(tf.reduce_sum((y_true - y_pred)**2, axis=1)) + beta * KL(p)
            # res = tf.reduce_mean(tf.reduce_sum((y_true - y_pred)**2, axis=1))
            return res
        return loss

    def training(self):

        train_x = self.preprocessing(self.dataframe)
        self.TModel.compile(optimizer=Adam(lr=0.005, 
                                           decay=1e-2), 
                                           loss=self.sae_square_loss(beta=0.01, p = 0.5))
        self.TModel.fit(train_x, train_x, nb_epoch=self.epoch, batch_size=2, verbose=2)
        model_result = Model(input = self.TModel.input, output=self.TModel.get_layer('embed').output )
        
        return  model_result.predict(train_x)