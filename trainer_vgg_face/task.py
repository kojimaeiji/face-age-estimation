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
import sys
print(sys.path)

from trainer_vgg_face import model
from trainer_vgg_face.model import DataSequence, download_mats, create_data,\
    FileDataSequence
import keras
import logging
from logging import StreamHandler
from sys import stdout
from keras.models import load_model
from keras.callbacks import EarlyStopping
import subprocess
"""This code implements a Feed forward neural network using Keras API."""

import argparse
import glob
import os

import multiprocessing
from tensorflow.python.lib.io import file_io


# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
FACE_AGE_MODEL = 'face_age.hdf5'

class ContinuousEval(keras.callbacks.Callback):
    """Continuous eval callback to evaluate the checkpoint once
       every so many epochs.
    """

    def __init__(self,
                 eval_frequency,
                 data_sequence,
                 learning_rate,
                 job_dir):
        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.data_sequence = data_sequence
#        self.validation_prefix = vaidation_prefix

    def on_epoch_begin(self, epoch, logs={}):
        if epoch > 0 and epoch % self.eval_frequency == 0:

            # Unhappy hack to work around h5py not being able to write to GCS.
            # Force snapshots and saves to local filesystem, then copy them
            # over to GCS.
            model_path_glob = 'checkpoint.*'
            if not self.job_dir.startswith("gs://"):
                model_path_glob = os.path.join(self.job_dir, model_path_glob)
            checkpoints = glob.glob(model_path_glob)
            if len(checkpoints) > 0:
                checkpoints.sort()
                face_age_model = load_model(checkpoints[-1], compile=False)
                face_age_model = model.compile_model(
                    face_age_model, self.learning_rate)
                # data_sequence = DataSequence(
                #    self.validation_prefix)
                loss, acc, mae = face_age_model.evaluate_generator(
                    self.data_sequence,
                    steps=self.data_sequence.length)
                print('\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}, {:.2f}] {}'.
                      format(
                          epoch, loss, acc, mae, face_age_model.metrics_names))
                if self.job_dir.startswith("gs://"):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])
            else:
                print(
                    '\nEvaluation epoch[{}] (no checkpoints found)'.
                    format(epoch))


def dispatch(train_prefix,
             validation_prefix,
             job_dir,
             learning_rate,
             num_epochs,
             checkpoint_epochs,
             lam,
             dropout,
             model_file
             ):
    print(model_file)
    # download train data
    #train_tmp_prefix, val_tmp_prefix = download_mats(train_prefix, validation_prefix)
    train_tmp_prefix = train_prefix
    val_tmp_prefix = validation_prefix
    print(train_tmp_prefix, val_tmp_prefix)

    # create train datasequence
    train_data_sequence = FileDataSequence(
        train_tmp_prefix
    )
    
    # create val datasequence
    val_datasequence = FileDataSequence(val_tmp_prefix)

    #train_x, train_y, cv_x, cv_y, input_shape = create_data(train_tmp_prefix)

    logger = logging.getLogger()
    sh = StreamHandler(stdout)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    logger.info('learning_rate=%s' % learning_rate)
    
    # create job dir if not exists
    try:
        os.makedirs(job_dir)
    except Exception:
        pass



    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to
    # GCS.
    checkpoint_path = FILE_PATH
    if not job_dir.startswith("gs://"):
        checkpoint_path = os.path.join(job_dir, checkpoint_path)
        verbose = 1
        multi = False
        num_worker = 1
    else:
        verbose = 2
        multi = True
        num_worker = 4 #multiprocessing.cpu_count()
        

    # Model checkpoint callback
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        period=checkpoint_epochs,
        mode='max')

    # Continuous eval callback
#     evaluation = ContinuousEval(eval_frequency,
#                                 # validation_tmp_prefix,
#                                 val_datasequence,
#                                 learning_rate,
#                                 job_dir,
#                                 )

    # Tensorboard logs callback
    tblog = keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    callbacks = [checkpoint, tblog]

    #x_train, y_train = train_data_sequence.__getitem__(0)
#     test_data_sequence = DataSequence(
#         validation_tmp_prefix
#     )

    # create model and compile
    if model_file is not None:
        if model_file.startswith('gs://'):
            cmd = 'gsutil cp %s /tmp' % model_file
            subprocess.check_call(cmd.split())
            real_model_file = '/tmp/%s' % model_file.split('/')[-1]
        else:
            real_model_file = model_file
        face_age_model = load_model(real_model_file, compile=False)
        #face_age_model = model.compile_model(face_age_model, learning_rate)
        initial_epoch = 1
    else:
        face_age_model = model.model_fn(lam, dropout)
        
        #first train last layer
        for layer in face_age_model.layers[:-1]:
            layer.trainable =False
        face_age_model = model.compile_model(face_age_model, learning_rate)
        face_age_model.fit_generator(
            train_data_sequence,
            validation_data=val_datasequence,
            validation_steps=val_datasequence.length,
            steps_per_epoch=train_data_sequence.length,
            verbose=verbose,
            epochs=1,
            callbacks=callbacks)
        initial_epoch=2


    for layer in face_age_model.layers[:-6]:
        layer.trainable = True
    face_age_model = model.compile_model(face_age_model, learning_rate)
    
    face_age_model.fit_generator(
        train_data_sequence,
        validation_data=val_datasequence,
        validation_steps=val_datasequence.length,
        steps_per_epoch=train_data_sequence.length,
        verbose=verbose,
        epochs=num_epochs,
        callbacks=callbacks,
        initial_epoch=initial_epoch)

    # plot_history(history)
    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to
    # GCS.
    if job_dir.startswith("gs://"):
        face_age_model.save(FACE_AGE_MODEL)
        copy_file_to_gcs(job_dir, FACE_AGE_MODEL)
    else:
        face_age_model.save(os.path.join(job_dir, FACE_AGE_MODEL))

    # Convert the Keras model to TensorFlow SavedModel
    model.to_savedmodel(face_age_model, os.path.join(job_dir, 'export'))

# h5py workaround: copy local models over to GCS if the job_dir is GCS.

# def plot_history(history):
#     # print(history.history.keys())
#
#     # 精度の履歴をプロット
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.legend(['acc', 'val_acc'], loc='lower right')
#     plt.show()
#
#     # 損失の履歴をプロット
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend(['loss', 'val_loss'], loc='lower right')
#     plt.show()
#
# # 学習履歴をプロット
#     plot_history(history)


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') \
                as output_f:
            output_f.write(input_f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-prefix', '-tr',
                        required=True,
                        type=str,
                        help='Training files prefix local or GCS')
    parser.add_argument('--validation-prefix', '-cv',
                        required=True,
                        type=str,
                        help='Validation files prefix local or GCS')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and '
                        'export model')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate')
    parser.add_argument('--lam',
                        type=float,
                        default=0.0,
                        help='l2 regularizaion lambda')
    parser.add_argument('--dropout',
                        type=float,
                        default=1.0,
                        help='dropout')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Maximum number of epochs on which to trainer')
    parser.add_argument('--checkpoint-epochs',
                        type=int,
                        default=1,
                        help='Checkpoint per n training epochs')
    parser.add_argument('--model-file',
                        type=str,
                        default=None,
                        help='Model file to continue to train ')
    parse_args, unknown = parser.parse_known_args()
    dispatch(**parse_args.__dict__)
