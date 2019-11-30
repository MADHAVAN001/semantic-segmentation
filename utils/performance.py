from tensorflow.keras.callbacks import Callback
import os
import six
import io
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import time


class PerformanceMetrics(Callback):
    def __init__(self, filename, append=False):
        super(PerformanceMetrics).__init__()
        self.append_header = True
        self.filename = filename
        self.append = append
        self.start_train_time = None
        if six.PY2:
            self.file_flags = 'b'
            self._open_args = {}
        else:
            self.file_flags = ''
            self._open_args = {'newline': '\n'}

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_begin(self, epoch, logs=None):
        self.start_train_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        trainable_count = int(
            np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))
        non_trainable_count = int(
            np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)]))

        row = list()
        row.append(str(trainable_count))
        row.append(str(non_trainable_count))
        row.append(str(self.estimate_flops()))
        row.append(str(time.time() - self.start_train_time))

        self.csv_file.write(",".join(row))

    def estimate_flops(self):
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()

        flops = tf.profiler.profile(graph=K.get_session().graph,
                                    run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops
