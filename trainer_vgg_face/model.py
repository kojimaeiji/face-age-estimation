# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.models import Model
#import cv2
from keras.utils import Sequence
from keras.utils import np_utils
from scipy.io import loadmat
import tensorflow as tf
import subprocess
from keras.optimizers import Optimizer, SGD, Adam
from keras.legacy import interfaces
import glob
import tensorflow
from keras.initializers import he_normal, glorot_normal
from keras.callbacks import EarlyStopping
from _io import BytesIO
from tensorflow.python.lib.io import file_io
from vggface import VGGFace
"""Implements the Keras Sequential model."""


import keras
from keras import backend as K, metrics
from keras import models
from keras.layers import Input

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import numpy as np
import logging

img_rows, img_cols = 224, 224

seed = 1

def model_fn(lam, dropout):
    """Create a Keras Sequential model with layers."""
    input_tensor = Input(shape=(img_rows, img_cols, 3))
    vggface16 =VGGFace(include_top=True, model='vgg16',weights='vggface',
                  input_tensor=input_tensor)
    vggface16.layers.pop()
    vggface16.layers.pop()
    #vgg16.outputs = [vgg16.layers[-1].output]
    #vgg16.layers[-1].outbound_nodes = []
    # 最後のconv層の直前までの層をfreeze
    #vgg16.output_shape = vgg16.layers[-1].output_shape
    #top_model = Flatten()(vgg16.output)
    #top_model = Dense(1024, activation='relu', name='last_2', kernel_initializer=he_normal(seed))(top_model)
#     top_model.add(
    print(vggface16.layers[-1].output)
    top_model = Dropout(dropout)(vggface16.layers[-1].output)
    top_model = Dense(101, activation='softmax',
                      kernel_initializer=glorot_normal(seed), name='last')(top_model)
    model = Model(inputs=vggface16.input, outputs=top_model)
    #for layer in model.layers[:18]:
    #     layer.trainable = False

    #compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate):
    last_layer_variables = list()
    for layer in model.layers:
        if layer.name in ['last', 'last_2','last_3']:
            last_layer_variables.extend(layer.weights)
    model.compile(loss='categorical_crossentropy',
#                 optimizer=MultiSGD(lr=learning_rate, momentum=0.9,
#                                   decay=0.0005,
#                                    exception_vars=last_layer_variables,
#                                    multiplier=10),
                optimizer=SGD(lr=learning_rate, momentum=0.9, nesterov=True),
                #optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy', age_mae])
    return model

CONST_LIST = [float(_i) for _i in range(101)]

def age_mae(y_true, y_pred):
    y_true = tf.cast(K.argmax(y_true, axis=1),'float')
    labels = K.constant(CONST_LIST, dtype=tf.float32)
    y_pred = labels * y_pred
    y_pred = K.sum(y_pred, axis=1)
    return K.mean(K.abs(y_true-y_pred), axis=0)

class MultiSGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, exception_vars=None, multiplier=10, **kwargs):
        super(MultiSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.exception_vars = exception_vars
        self.multiplier = multiplier

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            multiplied_lr = lr * self.multiplier if p in self.exception_vars \
                else lr
            if p in self.exception_vars:
                print(p)
            v = self.momentum * m - multiplied_lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - multiplied_lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  #'exception_vars': self.exception_vars,
                  #'multiplier': self.multiplier
                  }
        base_config = super(MultiSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                      outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
        builder.save()


def convert_to_ordinal(age):
    age_vecs = []
    for i in range(80):
        if age > i:
            age_vec = np.array([0.0, 1.0])
        else:
            age_vec = np.array([1.0, 0.0])
        age_vecs.append(age_vec)
    #print('age=%s, age_vec = %s' % (age, age_vec))
    return age_vecs


def get_filenum(file_prefix):
    files = tf.gfile.Glob(file_prefix)
    return len(files)


def create_y_encode(y):
    return [1.0, 0.0] if y > 0.5 else [0.0, 1.0]


class DataSequence(Sequence):
    def __init__(self, x, y, batch_size):
        # コンストラクタ
        #self.data_file_path = input_file
        #data = get_meta(input_file)
        self.x = x
        self.y = y
#         for i in range(len(self.y)):
#             age_vec = convert_to_ordinal(self.y[i])
#             self.y[i] = age_vec
        self.batch_size = batch_size
        self.length = len(self.x) // batch_size if len(
            self.x) % batch_size == 0 else (len(self.x) // batch_size) + 1

    def __getitem__(self, idx):
        # データの取得実装
        logger = logging.getLogger()
        #logger.info('idx=%s' % idx)
        #age = self.data.loc[:, 2]

        batch_size = self.batch_size

        # last batch or not
        if idx != self.length - 1:
            X, Y = convert_to_minibatch(
                self.x, self.y, idx * batch_size, (idx + 1) * batch_size)
        else:
            X, Y = convert_to_minibatch(
                self.x, self.y, idx * batch_size, None)
        return X, Y

    def __len__(self):
        # バッチ数
        return self.length

    def on_epoch_end(self):
        # epoch終了時の処理
        pass


class FileDataSequence(Sequence):
    def __init__(self, file_prefix):
        self.file_prefix = file_prefix

        self.length = get_filenum(file_prefix)

    def __getitem__(self, idx):
        # データの取得実装
        #logger = logging.getLogger()
        #logger.info('idx=%s' % idx)

        #batch_size = self.batch_size
        file_prefix = self.file_prefix.split('/')[-1][0:-1]
        root = self.file_prefix.split(file_prefix)[0]
        filename = '%s-%s.npz' % (file_prefix, idx)
        full_path = '%s%s' % (root, filename)
        #print('full_path=%s' % full_path)
        x, y = load_data(full_path)
        #x = x / 255.0
        x = x.astype(np.float32)
        mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 1, 3)
        x -= mean
        #y = np.array([create_y_encode(y[i]) for i in range(len(y))])
        y = np_utils.to_categorical(y, 101)
        #print('return x,y')
        return x, y

    def __len__(self):
        # バッチ数
        return self.length

    def on_epoch_end(self):
        # epoch終了時の処理
        pass


def convert_to_minibatch(X, Y, start_idx, end_idx):
    if end_idx:
        X_mini = X[start_idx:end_idx]
        #Y_mini = [Y[i][start_idx:end_idx] for i in range(len(Y))]
        Y_mini = Y[start_idx:end_idx]
    else:
        X_mini = X[start_idx:]
        #Y_mini = [Y[i][start_idx:] for i in range(len(Y))]
        Y_mini = Y[start_idx:]

    return X_mini, Y_mini


def unpack(xy):
    x, y = xy
    return x, y


def download_mats(train_file_prefix, val_file_prefix):
    if train_file_prefix.startswith('gs://'):
        cmd = 'gsutil cp %s /tmp' % train_file_prefix
        subprocess.check_call(cmd.split())
        cmd = 'cat /tmp/imdb_face_vgg_all.tar.gz-* > /tmp/imdb_face_vgg_all.tar.gz'
        subprocess.check_call(cmd, shell=True)
        cmd = 'tar -zxvf /tmp/imdb_face_vgg_all.tar.gz -C /tmp'
        subprocess.check_call(cmd.split())
        return '/tmp/wiki_face_vgg_all/imdb_224_all-tr*','/tmp/imdb_face_vgg_all/imdb_224_all-cv*'
        #return '/tmp/%s' % file_prefix.split('/')[-1]
    else:
        return train_file_prefix, val_file_prefix
    

def load_data(mat_path):
    with BytesIO(file_io.read_file_to_string(mat_path, binary_mode=True)) as f:
        data = np.load(f)
        x, y = data["image"], data["age"]
    return x, y


def load_data_split(mat_path):
    d = loadmat(mat_path)

    x, y = d["image"], d["age"][0]
    length = len(x)
    train_len = int(length * 0.8)
    x_train = x[0:train_len]
    y_train = y[0:train_len]
    x_cv = x[train_len:]
    y_cv = y[train_len:]
    return (x_train, y_train), (x_cv, y_cv)


def convert_to_column_list(y):
    columns = []
    column_len = len(y[0])
    for j in range(column_len):
        column = []
        for i in range(len(y)):
            one_ele = y[i][j]
            column.append(one_ele)
        column = np.array(column)
        columns.append(column)
    return columns


def create_data(input_file):
    (X_train, y_train), (X_test, y_test) = load_data_split(input_file)

    # データをfloat型にして正規化する
    # BGR
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 1, 3)
    X_train = X_train.astype('float32') - mean
    X_test = X_test.astype('float32') - mean

#     img_rows = 60
#     img_cols = 60

    # image_data_formatによって畳み込みに使用する入力データのサイズが違う
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(-1, 3, img_rows, img_cols)
        X_test = X_test.reshape(-1, 3, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(-1, img_rows, img_cols, 3)
        X_test = X_test.reshape(-1, img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')
    #y_train = np_utils.to_categorical(y_train, 10)
#     y_test = np_utils.to_categorical(y_test, 10)
#     y_train = [convert_to_ordinal(y_train[i])
#                for i in range(len(y_train))]
#    y_train = convert_to_column_list(y_train)
    y_train = np_utils.to_categorical(y_train, 101)

#     y_test = [convert_to_ordinal(y_test[i])
#               for i in range(len(y_test))]
#     y_test = convert_to_column_list(y_test)
    y_test = np_utils.to_categorical(y_test, 101)
    return X_train, y_train, X_test, y_test, input_shape


if __name__ == '__main__':
    #file_prefix = download_mats('/home/jiman/data/wiki_process_10000.mat')
    #x_tr, y_tr, x_t, y_t, input_shape = create_data(
    #    file_prefix)
    #     print(x_tr.shape, input_shape)
    #     print(len(y_tr))
    #     print(y_tr[0])
    #mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 1, 3)
    #print(x_tr[1][1][1][0])
    #print(x_tr[1][1][1][1])
    #print(x_tr[1][1][1][2])
    #x_tr = x_tr - mean
    #print(x_tr[1][1][1][0])
    #print(x_tr[1][1][1][1])
    #print(x_tr[1][1][1][2])
    model = model_fn(lam=0.0, dropout=0.5)
    for layer in model.layers[:-6]:
        layer.trainable = False
    compile_model(model, learning_rate=0.001)
    print(model.summary())
    #print(type(np_utils.to_categorical(5, 10)[0]))
    #     data = get_meta(
    #         ['gs://kceproject-1113-ml/intermediate/csv/path_age.csv-00000-of-00221'])
#    seq = DataSequence('/Users/saboten/data/wiki_process_60_128*')
#     seq = DataSequence(x_tr, y_tr, 64)
#     x_tr, y_tr = seq.__getitem__(0)
#     print(seq.length)
#     print(x_tr.shape)
#     print(y_tr.shape)
#     print(x_tr[0])
#     print(y_tr[0])
#     cv_seq = DataSequence(x_t, y_t, 64)
#     print('cv_length=%s' % cv_seq.length)
#     print(y_t.shape)
#     print(y_t.shape)
    #print(x_t[0])
    #print(y_t[0])

#     data=model.evaluate_generator(
#                     seq,
#                     steps=seq.length)
#     print(data)
#     img_mat = x_t[0]
#     print('shape=%s' % ((img_mat.shape),))
#     print(type(x_t[0][0][0][0]))
#     print(type(y_t[0][0]))
    # model_fn()
#     img_mat = img_mat* 255.0
#     img_mat = img_mat.astype(np.uint8)
#     cv2.imshow('image',img_mat)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
