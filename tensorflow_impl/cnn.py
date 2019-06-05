import time
import argparse

import tensorflow as tf
import numpy as np

from utils import (shuffle_tensors, next_minibatch, get_labels,
                   get_datasets, TensorBoardHandler, ModelSaver,
                   check_processed_dir_existance)


par = argparse.ArgumentParser(description="ECG Convolutional " +
                                           "Neural Network implementation")

par.add_argument("-lr", dest="learning_rate",
                 type=float, default=0.001,
                 help="Learning rate used by the model")

par.add_argument("-e", dest="epochs",
                 type=int, default=50,
                 help="The number of epochs the model will train for")

par.add_argument("-bs", dest="batch_size",
                 type=int, default=32,
                 help="The batch size of the model")

par.add_argument("--display-step", dest="display_step",
                 type=int, default=10,
                 help="The display step")

par.add_argument("--dropout", type=float, default=0.5,
                 help="Dropout probability")

par.add_argument("--restore", dest="restore_model",
                 action="store_true", default=False,
                 help="Restore the model previously saved")

par.add_argument("--freeze", dest="freeze",
                 action="store_true", default=False,
                 help="Freezes the model")

par.add_argument("--heart-diseases", nargs="+",
                 dest="heart_diseases",
                 default=["apnea-ecg", "svdb", "afdb"],
                 choices=["apnea-ecg", "mitdb", "nsrdb", "svdb", "afdb"],
                 help="Select the ECG diseases for the model")

par.add_argument("--verbose", dest="verbose",
                 action="store_true", default=False,
                 help="Display information about minibatches")

args = par.parse_args()

# Parameters
learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
display_step = args.display_step
dropout = args.dropout
restore_model = args.restore_model
freeze = args.freeze
heart_diseases = args.heart_diseases
verbose = args.verbose

# Network Parameters
nr_inputs = 350 # changing this will also have to change the shape from wdense1
nr_classes = len(heart_diseases)

# TF Graph input
x = tf.placeholder(tf.float32, shape=[None, nr_inputs], name="X_input")
y = tf.placeholder(tf.float32, shape=[None, nr_classes], name="Y_classes")
keep_prob = tf.placeholder(tf.float32)

check_processed_dir_existance()


class CNN(object):
    weights = {
        # 10x1 conv filter, 1 input, 64 outputs
        'wconv1': tf.Variable(tf.random_normal([10, 1, 64])),
        # 10x64 conv filter, 64 inputs, 128 outputs
        'wconv2': tf.Variable(tf.random_normal([10, 64, 128])),
        # 10x128 conv filter, 128 inputs, 128 outputs
        'wconv3': tf.Variable(tf.random_normal([10, 128, 128])),
        # 10x128 conv filter, 128 inputs, 256 outputs
        'wconv4': tf.Variable(tf.random_normal([10, 128, 256])),
        # fully connected, 1024 outputs
        'wdense1': tf.Variable(tf.random_normal([5376, 1024])),
        # fully connected, 1024 inputs, 2048 outputs
        'wdense2': tf.Variable(tf.random_normal([1024, 2048])),
        # 2048 inputs, class prediction
        'wout': tf.Variable(tf.random_normal([2048, nr_classes]))
    }

    biases = {
        'bconv1': tf.Variable(tf.random_normal([64])),
        'bconv2': tf.Variable(tf.random_normal([128])),
        'bconv3': tf.Variable(tf.random_normal([128])),
        'bconv4': tf.Variable(tf.random_normal([256])),
        'bdense1': tf.Variable(tf.random_normal([1024])),
        'bdense2': tf.Variable(tf.random_normal([2048])),
        'bout': tf.Variable(tf.random_normal([nr_classes]))
    }

    def __init__(self, weights=None, biases=None):
        self.weights = weights if weights else self.weights
        self.biases = biases if biases else self.biases
        self.datasets = get_datasets(heart_diseases, nr_inputs)
        self.label_data = get_labels(self.datasets)

        self.saver = ModelSaver(save_dir="saved_models/cnn/")

        logs_path = "tensorboard_data/cnn/"
        self.tensorboard_handler = TensorBoardHandler(logs_path)
        self.tensorboard_handler.add_histograms(self.weights)
        self.tensorboard_handler.add_histograms(self.biases)

        self.build()

    def build(self):
        dataset_len = []
        for dataset in self.datasets:
            dataset_len.append(len(dataset))

        validation_size = int(0.1 * sum(dataset_len))

        print("Validation size: {}".format(validation_size))
        print("Total samples: {}".format(sum(dataset_len)))
        print("Heart diseases: {}".format(', '.join(heart_diseases)))

        # Shuffle the input, helps training
        concat_dataset = np.concatenate(self.datasets)
        concat_dataset, self.label_data = shuffle_tensors(concat_dataset, self.label_data)

        # split training and testing sets
        self.X_train, self.X_test = np.split(concat_dataset,
                                             [len(concat_dataset)-validation_size])

        self.Y_train, self.Y_test = np.split(self.label_data,
                                             [len(self.label_data)-validation_size])

        if verbose:
            print("X_train shape: {}".format(self.X_train.shape))
            print("Y_train shape: {}".format(self.Y_train.shape))
            print("X_test shape: {}".format(self.X_test.shape))
            print("Y_test shape: {}".format(self.Y_test.shape))

    def train(self, x):
        is_training = not freeze
        # Reshape input so that we can feed it to the first conv layer
        x = tf.reshape(x, shape=[-1, nr_inputs, 1])
        
        # Convolution Layer 1
        conv1 = self.conv1d(x, self.weights['wconv1'], self.biases['bconv1'])
        conv1 = self.maxpool1d(conv1)
        # Batch Norm Layer 1
        conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training)

        # Convolution Layer 2
        conv2 = self.conv1d(conv1, self.weights['wconv2'], self.biases['bconv2'])
        conv2 = self.maxpool1d(conv2)
        # Batch Norm Layer 2
        conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training)

        # Convolution Layer 3
        conv3 = self.conv1d(conv2, self.weights['wconv3'], self.biases['bconv3'])
        conv3 = self.maxpool1d(conv3)
        # Batch Norm Layer 3
        conv3 = tf.contrib.layers.batch_norm(conv3, is_training=is_training)

        # Convolution Layer 4
        conv4 = self.conv1d(conv3, self.weights['wconv4'], self.biases['bconv4'])
        conv4 = self.maxpool1d(conv4)
        # Batch Norm Layer 4
        conv4 = tf.contrib.layers.batch_norm(conv4, is_training=is_training)

        # Fully connected layer
        # Reshape conv4 output to fit fully connected layer input
        # shape_size is a cause for errors, it is determined using
        # conv4.shape[1]*conv4.shape[2] and also has to be changed in weight definition
        shape_size = conv4.shape[1] * conv4.shape[2]
        fc1 = tf.reshape(conv4, [-1, shape_size])

        # Fully connected layer 1
        fc1 = tf.add(tf.matmul(fc1, self.weights['wdense1']), self.biases['bdense1'])
        fc1 = tf.contrib.layers.batch_norm(fc1, is_training=is_training)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)

        # Fully connected layer 2
        fc2 = tf.add(tf.matmul(fc1, self.weights['wdense2']), self.biases['bdense2'])
        fc2 = tf.contrib.layers.batch_norm(fc2, is_training=is_training)
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc2, self.weights['wout']), self.biases['bout'])
        return out

    def conv1d(self, x, W, b, strides=1):
        # conv1d needs a 3-D input([batch, in_width, in_channels]) and
        # filter tensors([filter_width, in_channels, out_channels])
        x = tf.nn.conv1d(x, W, stride=strides, padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool1d(self, x, pool_size=2):
        # [batch, height, width, channels] input type: tf.float32
        return tf.contrib.keras.layers.MaxPool1D(pool_size=pool_size)(x)

    def cost(self, pred):
        softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,
                                                             labels=y)
        return tf.reduce_mean(softmax)

    def optimizer(self, cost):
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return adam.minimize(cost)

    def evl(self, pred):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def get_data(self):
        return (self.X_train, self.X_test,
                self.Y_train, self.Y_test)


# Construct model
model = CNN()
pred = model.train(x)

# Define loss and optimizer
cost = model.cost(pred)

# Add scalar summary to cost tensor
model.tensorboard_handler.add_scalar("training_loss", cost)

# Create optimier
optimizer = model.optimizer(cost)

# Evaluate model
accuracy = model.evl(pred)

# Add scalar summary to accuracy tensor
model.tensorboard_handler.add_scalar("training_accuracy", accuracy)
# testing_acc = model.tensorboard_handler.add_scalar("testing_accuracy", accuracy)

# Merge tensorboard data
merged = model.tensorboard_handler.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()

X_train, X_test, Y_train, Y_test = model.get_data()


# Launch the graph
with tf.Session() as sess:
    # Initialize the variables for the current session
    sess.run(init)

    # Add the graph to tensorboard writer
    model.tensorboard_handler.writer.add_graph(sess.graph)
    step = 1

    # If restore_model flag True, restore the model
    if restore_model:
        model.saver.restore(sess)

    # Set start time
    total_time = time.time()
    epoch_time = time.time()

    print("-"*50)
    # Train
    for epoch in range(1, epochs):
        for X_train_batch, Y_train_batch in next_minibatch(X_train, Y_train, batch_size):
            sess.run(optimizer, feed_dict={x: X_train_batch,
                                           y: Y_train_batch,
                                           keep_prob: dropout})

            # Once a few steps run the accuracy for the training model
            if verbose and (step % display_step) == 0:
                loss, acc = sess.run([cost, accuracy],
                                     feed_dict={x: X_train,
                                                y: Y_train,
                                                keep_prob: 1.0})

                print("Step: {}".format(step))
                print("Training loss: {:.4f}".format(loss))
                print("Training Accuracy: {:.4f}".format(acc))

            step += 1

        print("#"*50)
        print("Epoch summary:")
        print("Epoch: {}".format(epoch))
        print("Training took: {0:.2f}s".format(time.time() - epoch_time))
        summary, acc = sess.run([merged, accuracy],
                                feed_dict={x: X_train,
                                           y: Y_train,
                                           keep_prob: 1.0})
        print("Training accuracy: {0:.4f}".format(acc))

        # Run testing accuracy
        acc = sess.run(accuracy, feed_dict={x: X_test,
                                            y: Y_test,
                                            keep_prob: 1.0})
        print("Testing accuracy: {0:.4f}".format(acc))
        print("#"*50)

        # write to log
        model.tensorboard_handler.writer.add_summary(summary, epoch)

        # Reset epoch time
        epoch_time = time.time()

    print("-"*50)

    # Total training time
    print("Total training time: {0:.2f}s".format(time.time() - total_time))
    loss, acc = sess.run([cost, accuracy], feed_dict={x: X_train,
                                                      y: Y_train,
                                                      keep_prob: 1.0})
    
   
    print("Training Accuracy: {0:.4f}".format(acc))
    print("Training Loss: {0:.4f}".format(loss))

    # If model not freezed, save the model
    if not freeze:
        model.saver.save(sess)

    # Run testing accuracy
    acc = sess.run(accuracy, feed_dict={x: X_test,
                                        y: Y_test,
                                        keep_prob: 1.0})
    print("Testing Accuracy: {0:.4f}".format(acc))
    print("-"*50)

