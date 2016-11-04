from keras.datasets import Convolution1D
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten
import os
import csv
import random
from collections import defaultdict

data = []

for subdir, _, files in os.walk('datasets'):
    label = 0 if 'arrhythmia' in subdir else 1
    for record in files:
        if record.endswith('.csv'):
            with open(record, 'r') as record_file:
                reader = csv.reader(record_file)
                # skip headers
                reader.next()
                reader.next()
                record_data = []
                for row in reader:
                    record_data.append(float(row[1]))
                data.append((record_data, label))

random.shuffle(data)

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