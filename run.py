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