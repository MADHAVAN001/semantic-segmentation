import io
import time

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


class PerformanceMetrics(Callback):
    def __init__(self, filename, append=False):
        super(PerformanceMetrics).__init__()
        self.append_header = True
        self.filename = filename
        self.append = append
        self.start_train_time = None
        self.csv_file = None

    def on_train_begin(self, logs=None):
        self.csv_file = io.open(self.filename, 'w')

    def headers(self):
        row = list()
        row.append("epoch")
        row.append("trainable_parameters")
        row.append("non_trainable_parameters")
        row.append("total_parameters")
        row.append("epoch_training_time")

        self.csv_file.write(",".join(row))

    def on_epoch_begin(self, epoch, logs=None):
        self.start_train_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        trainable_count = np.sum([K.count_params(w) for w in self.model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in self.model.non_trainable_weights])

        row = list()
        row.append(str(epoch))
        row.append(str(trainable_count))
        row.append(str(non_trainable_count))
        row.append(str(trainable_count + non_trainable_count))
        row.append(str(time.time() - self.start_train_time))

        self.csv_file.write(",".join(row))
