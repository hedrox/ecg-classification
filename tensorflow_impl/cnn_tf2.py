import time
import argparse

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, MaxPool1D, Dropout
from tensorflow.keras.metrics import CategoricalAccuracy

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from utils import get_labels, get_datasets, check_processed_dir_existance


par = argparse.ArgumentParser(description="ECG Convolutional " +
                                           "Neural Network implementation with Tensorflow 2.0")

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
n_inputs = 350
n_classes = len(heart_diseases)

check_processed_dir_existance()


class CNN:
    def __init__(self):
        self.datasets = get_datasets(heart_diseases, n_inputs)
        self.label_data = get_labels(self.datasets)
        self.callbacks = []

        # Initialize callbacks
        tensorboard_logs_path = "tensorboard_data/cnn/"
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_path,
                                                     histogram_freq=1, write_graph=True,
                                                     embeddings_freq=1)

        # load_weights_on_restart will read the filepath of the weights if it exists and it will
        # load the weights into the model
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="saved_models/cnn/model.hdf5",
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         load_weights_on_restart=restore_model)

        self.callbacks.extend([tb_callback, cp_callback])

        self.set_data()
        self.define_model()

    def set_data(self):
        dataset_len = []
        for dataset in self.datasets:
            dataset_len.append(len(dataset))

        # validation on 10% of the training data
        validation_size = 0.1

        print("Validation percentage: {}%".format(validation_size*100))
        print("Total samples: {}".format(sum(dataset_len)))
        print("Heart diseases: {}".format(', '.join(heart_diseases)))

        concat_dataset = np.concatenate(self.datasets)

        self.split_data(concat_dataset, validation_size)

        # Reshape input so that we can feed it to the conv layer
        self.X_train = tf.reshape(self.X_train, shape=[-1, n_inputs, 1])
        self.X_test = tf.reshape(self.X_test, shape=[-1, n_inputs, 1])
        self.X_val = tf.reshape(self.X_val, shape=[-1, n_inputs, 1])

        if verbose:
            print("X_train shape: {}".format(self.X_train.shape))
            print("Y_train shape: {}".format(self.Y_train.shape))
            print("X_test shape: {}".format(self.X_test.shape))
            print("Y_test shape: {}".format(self.Y_test.shape))
            print("X_val shape: {}".format(self.X_val.shape))
            print("Y_val shape: {}".format(self.Y_val.shape))

    def define_model(self):

         inputs = tf.keras.Input(shape=(n_inputs, 1), name='input')

         # 64 filters, 10 kernel size
         x = Conv1D(64, 10, activation='relu')(inputs)
         x = MaxPool1D()(x)
         x = BatchNormalization()(x)

         x = Conv1D(128, 10, activation='relu')(x)
         x = MaxPool1D()(x)
         x = BatchNormalization()(x)

         x = Conv1D(128, 10, activation='relu')(x)
         x = MaxPool1D()(x)
         x = BatchNormalization()(x)

         x = Conv1D(256, 10, activation='relu')(x)
         x = MaxPool1D()(x)
         x = BatchNormalization()(x)

         x = Flatten()(x)
         x = Dense(1024, activation='relu', name='dense_1')(x)
         x = BatchNormalization()(x)
         x = Dropout(dropout)(x)

         x = Dense(2048, activation='relu', name='dense_2')(x)
         x = BatchNormalization()(x)
         x = Dropout(dropout)(x)

         outputs = Dense(n_classes, activation='softmax', name='predictions')(x)

         self.cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
         optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
         accuracy = CategoricalAccuracy()
         self.cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                                metrics=[accuracy])

    def split_data(self, dataset, validation_size):
        """
        Suffle then split training, testing and validation sets
        """

        # In order to use statify in train_test_split we can't use one hot encodings,
        # so we convert to array of labels
        label_data = np.argmax(self.label_data, axis=1)

        # Splitting the dataset into train and test datasets
        res = train_test_split(dataset, label_data,
                               test_size=validation_size, shuffle=True,
                               stratify=label_data)

        self.X_train, self.X_test, self.Y_train, self.Y_test = res

        # From the training dataset we further split it to obtain the validation dataset
        res = train_test_split(self.X_train, self.Y_train,
                               test_size=validation_size, stratify=self.Y_train)

        self.X_train, self.X_val, self.Y_train, self.Y_val = res

        # Convert the array of labels back into one hot encodings to be able to do training
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test)
        self.Y_val = tf.keras.utils.to_categorical(self.Y_val)

    def get_data(self):
        return (self.X_train, self.X_test, self.X_val,
                self.Y_train, self.Y_test, self.Y_val)


def main():
    # Construct model
    model = CNN()
    X_train, X_test, X_val, Y_train, Y_test, Y_val = model.get_data()

    # Set start time
    total_time = time.time()

    print("-"*50)
    if restore_model:
        print("Restoring model: {}".format('saved_models/cnn/model.hdf5'))

    # Train
    model.cnn_model.fit(X_train, Y_train, batch_size=batch_size,
                        epochs=epochs, validation_data=(X_val, Y_val),
                        callbacks=model.callbacks)

    print("-"*50)

    # Total training time
    print("Total training time: {0:.2f}s".format(time.time() - total_time))

    # Test
    model.cnn_model.evaluate(X_test, Y_test, batch_size=batch_size)
    print("-"*50)
    print("Testing results:")
    y_pred = model.cnn_model.predict(X_test, batch_size=batch_size)

    # The following scikit-learn methods only accept array of labels, not one hot encodings
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(Y_test, axis=1)

    # Precision and recall could also be done as callbacks in the evaluate or fit function
    print("Precision: {}".format(precision_score(y_true, y_pred, average='micro')))
    print("Recall: {}".format(recall_score(y_true, y_pred, average='micro')))
    print("Confusion matrix: \n{}".format(confusion_matrix(y_true, y_pred, labels=[0,1,2])))
    disease_indexes = list(range(len(heart_diseases)))
    print("Indexes {} correspond to labels {}".format(disease_indexes, [x for x in heart_diseases]))

    print("-"*50)

if __name__ == "__main__":
    main()
