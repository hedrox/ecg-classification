from keras.datasets import Convolution1D
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten
from data import process_data
import numpy  as np
import os

if not os.listdir('datasets/processed'):
    process_data()

arrhy_data = np.loadtxt(open('datasets/processed/arrhythmia.csv', 'r'), skiprows=1)
malignant_data = np.loadtxt(open('datasets/processed/malignant-ventricular-ectopy.csv', 'r'), skiprows=1)
arrhy_data = arrhy_data[:len(malignant_data)]
arrhy_len = len(arrhy_data)/500

i = 0
X_train = []
y_train = []
for _ in range(arrhy_len):
    X_train.append(np.asarray(arrhy_data[i:i+500]))
    y_train.append(0)
    X_train.append(np.asarray(malignant_data[i:i+500]))
    y_train.append(1)
    i += 500

validation_size = 0.1  * len(X_train)
X_test = X_train[:-validation_size]
y_test = y_train[:-validation_size]

nb_filters = 32
nb_epoch = 10
batch_size = 8

model = Sequential()
model.add(Convolution1D(nb_filters, 3, input_shape=(500,), activation='relu'))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filters, 3, activation='relu'))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filters, 3, activation='relu'))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2), activation='softmax')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, validation_data=(X_test, y_test))