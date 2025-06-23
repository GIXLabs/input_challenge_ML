import tensorflow as tf

class CustomTensorBoard:
    def __init__(self, log_dir, **kwargs):
        self.writer = tf.summary.create_file_writer(log_dir)

    def log(self, step, **stats):
        with self.writer.as_default():
            for name, value in stats.items():
                tf.summary.scalar(name, value, step=step)
