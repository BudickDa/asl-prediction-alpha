#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import shutil
import tensorflow.contrib.learn as learn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

tf.logging.set_verbosity(tf.logging.INFO)

TIMESERIES_COL = 'rawdata'
N_OUTPUTS = 1
SEQ_LEN = None
DEFAULTS = None
N_INPUTS = None


def init(hparams):
    global SEQ_LEN, DEFAULTS, N_INPUTS
    SEQ_LEN = hparams['sequence_length']
    DEFAULTS = [[0.0] for x in xrange(0, SEQ_LEN)]
    N_INPUTS = SEQ_LEN - N_OUTPUTS


# read data and convert to needed format
def read_dataset(filename, mode, batch_size):
    def _input_fn():
        # could be a path to one file or a file pattern.
        input_file_names = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(
            input_file_names, num_epochs=None, shuffle=True)

        reader = tf.TextLineReader(skip_header_lines=True)
        _, value = reader.read_up_to(filename_queue, num_records=batch_size)

        value_column = tf.expand_dims(value, -1)
        # print ('readcsv={}'.format(value_column))

        # all_data is a list of tensors
        all_data = tf.decode_csv(value_column, record_defaults=DEFAULTS, field_delim=';')
        inputs = all_data[1:]
        label = all_data[0]

        # from list of tensors to tensor with one more dimension
        inputs = tf.concat(inputs, axis=1)
        label = tf.concat(label, axis=1)
        # print ('inputs={}'.format(inputs))

        return {TIMESERIES_COL: inputs}, label  # dict of features, label

    return _input_fn


def rnn_model(features, mode, params):
    LSTM_SIZE = 3
    x = tf.split(features[TIMESERIES_COL], N_INPUTS, 1)

    # 1. Configure the RNN
    lstm_cell = rnn.BasicLSTMCell(LSTM_SIZE, forget_bias=1.0)
    outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Slice to keep only the last cell of the RNN
    outputs = outputs[-1]

    # Output is result of linear activation of last layer of RNN
    weight = tf.Variable(tf.random_normal([LSTM_SIZE, N_OUTPUTS]))
    bias = tf.Variable(tf.random_normal([N_OUTPUTS]))
    predictions = tf.matmul(outputs, weight) + bias
    return predictions


def lstmN_model(features, mode, params):
    # dynamic_rnn needs 3D shape: [BATCH_SIZE, N_INPUTS, 1]
    x = tf.reshape(features[TIMESERIES_COL], [-1, N_INPUTS, 1])

    # 2. configure the RNN
    lstm_cell1 = rnn.BasicLSTMCell(N_INPUTS * 2, forget_bias=1.0)
    lstm_cell2 = rnn.BasicLSTMCell(N_INPUTS // 2, forget_bias=1.0)
    lstm_cells = rnn.MultiRNNCell([lstm_cell1, lstm_cell2])
    outputs, _ = tf.nn.dynamic_rnn(lstm_cells, x, dtype=tf.float32)

    # 3. make lstm output a 2D matrix and pass through a dense layer
    # so that the dense layer is shared for all outputs
    lstm_flat = tf.reshape(outputs, [-1, N_INPUTS, lstm_cells.output_size])
    h1 = tf.layers.dense(lstm_flat, lstm_cells.output_size, activation=tf.nn.relu)
    h2 = tf.layers.dense(h1, lstm_cells.output_size // 2, activation=tf.nn.relu)
    predictions = tf.layers.dense(h2, 1, activation=None)  # (?, N_INPUTS, 1)
    predictions = tf.reshape(predictions, [-1, N_INPUTS])
    return predictions


def serving_input_fn():
    feature_placeholders = {
        TIMESERIES_COL: tf.placeholder(tf.float32, [None, N_INPUTS])
    }

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    features[TIMESERIES_COL] = tf.squeeze(features[TIMESERIES_COL], axis=[2])

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)
