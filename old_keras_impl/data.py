import os
import csv
import random

def process_data():
    listdir = os.listdir('datasets/raw')
    for d in listdir:
        with open('datasets/processed/{}.csv'.format(d), 'w') as write_record_file:
            csvwriter = csv.writer(write_record_file, delimiter=',')
            csvwriter.writerow(['value'])
            for record in os.listdir('datasets/raw/{}'.format(d)):
                if record.endswith('.csv'):
                    with open('datasets/raw/{}/{}'.format(d,record), 'r') as read_record_file:
                        reader = csv.reader(read_record_file)
                        # skip headers
                        reader.next()
                        reader.next()
                        for row in reader:
                            csvwriter.writerow([row[1]])