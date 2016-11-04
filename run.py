from keras.datasets import Convolution1D
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten
import os
import csv
import random
from collections import defaultdict

def process_data():
    with open('datasets/preprocessed.csv', 'w') as write_record_file:
        csvwriter = csv.writer(write_record_file, delimiter=',')
        csvwriter.writerow(['label', 'value'])
        for subdir, _, files in os.walk('datasets/raw'):
            label = 0 if 'arrhythmia' in subdir else 1
            for record in files:
                if record.endswith('.csv'):
                    with open('{}/{}'.format(subdir, record), 'r') as read_record_file:
                        reader = csv.reader(read_record_file)
                        # skip headers
                        reader.next()
                        reader.next()
                        for row in reader:
                            csvwriter.writerow([label, row[1]])

process_data()
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