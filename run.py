from keras.datasets import Convolution1D
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten
import os
import csv
from collections import defaultdict

dataset_files = os.listdir('datasets')

data = defaultdict(list)

for record in dataset_files:
    with open(record, 'r') as record_file:
        reader = csv.reader(record_file)
        # skip headers
        reader.next()
        reader.next()
        for row in reader:
            data[record].append(float(row[1]))

nb_filters = 32

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