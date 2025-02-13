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
from trainer_face_recog import model
from trainer_face_recog.model import FileDataSequence, download_mats
import keras
import logging
from logging import StreamHandler
from sys import stdout
from keras.models import load_model
"""This code implements a Feed forward neural network using Keras API."""

import argparse
import glob
import os

import multiprocessing
from tensorflow.python.lib.io import file_io


# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
FACE_AGE_MODEL = 'face_age_reco.hdf5'

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
             eval_frequency,
             num_epochs,
             batch_size,
             checkpoint_epochs,
             lam,
             dropout,
             trainable
             ):

    # download train data
    train_tmp_prefix, val_tmp_prefix = download_mats(train_prefix, validation_prefix)
    print(train_tmp_prefix, val_tmp_prefix)
    # download train data
    #validation_tmp_prefix = download_mats(validation_prefix)

    #train_x, train_y, cv_x, cv_y, input_shape = create_data(train_tmp_prefix)

    logger = logging.getLogger()
    sh = StreamHandler(stdout)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    logger.info('learning_rate=%s' % learning_rate)
    if job_dir.startswith('gs://'):
        model_file='/tmp/fr_model.h5'
        verbose = 2
    else:
        model_file='trainer_face_recog/fr_model.h5'
        verbose = 1

    face_age_model = model.model_fn(learning_rate, lam, dropout, model_file=model_file, trainable=trainable)

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
#
#     meta_data = get_meta(train_files)
#     indexes = [i for i in range(len(meta_data))]
#     random.shuffle(indexes)
#     meta_data = meta_data.loc[indexes].reset_index(drop=True)

    # Model checkpoint callback
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        period=checkpoint_epochs,
        mode='max')

    # Continuous eval callback
    val_data_sequence = FileDataSequence(val_tmp_prefix)
    evaluation = ContinuousEval(eval_frequency,
                                # validation_tmp_prefix,
                                val_data_sequence,
                                learning_rate,
                                job_dir
                                )

    # Tensorboard logs callback
    tblog = keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    callbacks = [checkpoint, tblog]

    train_data_sequence = FileDataSequence(
        train_tmp_prefix
    )
    #x_train, y_train = train_data_sequence.__getitem__(0)
#     test_data_sequence = DataSequence(
#         validation_tmp_prefix
#     )

    face_age_model.fit_generator(  # x_train, y_train,
        #model.generator_input(train_files, chunk_size=CHUNK_SIZE),
        train_data_sequence,
        validation_data=val_data_sequence,
        validation_steps=val_data_sequence.length,
        steps_per_epoch=train_data_sequence.length,
        verbose=verbose,
        epochs=num_epochs,
        workers=multiprocessing.cpu_count(),
        use_multiprocessing=True,
        callbacks=callbacks)

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
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='minibatch size')
    parser.add_argument('--eval-frequency',
                        default=1,
                        help='Perform one evaluation per n epochs')
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
    parser.add_argument('--trainable',
                        type=int,
                        default=-1,
                        help='trainable last layer')
    parse_args, unknown = parser.parse_known_args()
    dispatch(**parse_args.__dict__)
