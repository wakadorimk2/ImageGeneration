import tensorflow as tf
import datetime

def plotLossFunction():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/gradient_tape/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    return summary_writer