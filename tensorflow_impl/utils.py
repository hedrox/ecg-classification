import os, sys

import tensorflow as tf
import numpy as np


class TensorBoardHandler(object):
    def __init__(self, logs_path=None):
        self.logs_path = logs_path if logs_path else "tensorboard_data/others/"
        self.writer = tf.summary.FileWriter(self.logs_path)

    def add_histograms(self, dct):
        for k, v in dct.items():
            tf.summary.histogram(str(k), v)

    def add_scalar(self, name, obj):
        return tf.summary.scalar(name, obj)

    def merge_all(self):
        return tf.summary.merge_all()


class ModelSaver(object):
    model_ext = ".ckpt"
    
    def __init__(self, save_dir=None, *args, **kwargs):
        self.save_dir = save_dir if save_dir else "saved_models/others/"

        # Create directory to store models
        if not os.path.isdir(self.save_dir):
            print("Saved model dir not found")
            print("Creating {}".format(self.save_dir))
            os.makedirs(self.save_dir)
        self.saver = tf.train.Saver(*args, **kwargs)

    def save(self, sess, model_name="model"):
        model_dir = self.save_dir + str(model_name) + self.model_ext
        self.saver.save(sess, model_dir)
        print("Model saved to {}".format(model_dir))
        
    def restore(self, sess, model_name="model"):
        model_dir = self.save_dir + str(model_name) + self.model_ext
        self.saver.restore(sess, model_dir)
        print("Model restored from {}".format(model_dir))


def shuffle_tensors(x, y):
    assert len(x) == len(y), "Lengths don't match"
    if type(x) == list or type(y) == list:
        x = np.array(x)
        y = np.array(y)

    perm = np.random.permutation(len(x))
    return x[perm], y[perm]

def next_minibatch(x, y, minibatch_size):
    assert x.shape[0] == y.shape[0], "Shapes don't match"
    for i in range(0, x.shape[0] - minibatch_size + 1, minibatch_size):
        slice_range = slice(i, i + minibatch_size)
        yield x[slice_range], y[slice_range]

def get_labels(datasets):
    nr_classes = len(datasets)
    labels = []
    for i in range(nr_classes):
         for _ in range(len(datasets[i])):
             class_label = [0] * nr_classes
             class_label[i] = 1
             labels.append(class_label)
    return np.array(labels)

def get_datasets(diseases, nr_inputs):
    datasets = []
    for disease in diseases:
        with open('datasets/processed/{}.csv'.format(disease)) as dis:
            dataset = np.loadtxt(dis)
            datasets.append(dataset)

    for idx, dataset in enumerate(datasets):
        _cutoff_val = reduce_ds(len(dataset), nr_inputs)
        _dataset = dataset[:_cutoff_val]
        datasets[idx] = _dataset.reshape(-1, nr_inputs)
        
    return datasets

def check_processed_dir_existance():
    if not os.listdir('datasets/processed'):
        print("You have to first process the data, " +
              "please call the download_data script")
        sys.exit(1)

def reduce_ds(dataset_len, nr_inputs):
    # fit the dataset length to the number of input neurons
    return dataset_len - (dataset_len % nr_inputs)

