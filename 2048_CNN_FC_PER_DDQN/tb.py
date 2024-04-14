import numpy as np
import tensorflow as tf
import datetime

class TensorBoardLogger:

    def __init__(self, log_dir=None):
        self.time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if log_dir is None:
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, name, value, step):
        with self.writer.as_default():
            tf.summary.scalar('batch_' + name, value, step=step)
            self.writer.flush()

logger = TensorBoardLogger()

#warning:If you file name =="tensorboard", which will lead to wrong
#I spand so much time on it, and finally find this problem on stack overflow


