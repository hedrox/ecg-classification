#!/usr/bin/env python

import os
import csv
import random
import subprocess
import argparse

par = argparse.ArgumentParser(description="Download and process Physionet Datasets")

par.add_argument("-dl", nargs="+",
                 dest="dataset_list",
                 default=[],
                 choices=["nsrdb", "apnea-ecg", "mitdb", "afdb", "svdb"],
                 help="The list of datasets to download")

args = par.parse_args()
dataset_list = args.dataset_list


def fetch_data():
    """
    nsrdb normal sinus rhythm
    apnea
    mitdb arrhythmia
    afdb atrial fibrillation
    svdb supraventricular arrhythmia 
    """

    physionet = {
        "nsrdb": ["16265", "16272", "16273", "16420", "16483", "16539", "16773",
                  "16786", "16795", "17052", "17453", "18177", "18184", "19088",
                  "19090", "19093", "19140", "19830"],
        "apnea-ecg": ["a01", "a01er", "a01r", "a02", "a02er", "a02r", "a03",
                      "a03er", "a03r", "a04", "a04er", "a04r", "a05", "a06",
                      "a07", "a08", "a09", "a10", "a11", "a12", "a13", "a14",
                      "a15", "a16", "a17", "a18", "a19", "a20", "b01", "b01er",
                      "b01r", "b02", "b03", "b04", "b05", "c01", "c01er", "c01r",
                      "c02", "c02er", "c02r", "c03", "c03er", "c03r", "c04",
                      "c05", "c06", "c07", "c08", "c09", "c10", "x01", "x02",
                      "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
                      "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18",
                      "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26",
                      "x27", "x28", "x29", "x30", "x31", "x32", "x33", "x34", "x35"],
        "mitdb": ["100", "101", "102", "103", "104", "105", "106", "107", "108",
                  "109", "111", "112", "113", "114", "115", "116", "117", "118",
                  "119", "121", "122", "123", "124", "200", "201", "202", "203",
                  "205", "207", "208", "209", "210", "212", "213", "214", "215",
                  "217", "219", "220", "221", "222", "223", "228", "230", "231",
                  "232", "233", "234"],
        "afdb": ["04015", "04043", "04048", "04126", "04746", "04908", "04936",
                 "05091", "05121", "05261", "06426", "06453", "06995", "07162",
                 "07859", "07879", "07910", "08215", "08219", "08378", "08405",
                 "08434", "08455"],
        "svdb": ["800", "801", "802", "803", "804", "805", "806", "807", "808",
                 "809", "810", "811", "812", "820", "821", "822", "823", "824",
                 "825", "826", "827", "828", "829", "840", "841", "842", "843",
                 "844", "845", "846", "847", "848", "849", "850", "851", "852",
                 "853", "854", "855", "856", "857", "858", "859", "860", "861",
                 "862", "863", "864", "865", "866", "867", "868", "869", "870",
                 "871", "872", "873", "874", "875", "876", "877", "878", "879",
                 "880", "881", "882", "883", "884", "885", "886", "887", "888",
                 "889", "890", "891", "892", "893", "894"]
    }

    dataset_dir = "datasets/raws"
    
    def check_folder_existance():
        if not os.path.isdir(dataset_dir):
            print("Directory {} not found".format(dataset_dir))
            print("Creating now...")
            os.makedirs(dataset_dir)

        for database in physionet:
            folder = os.path.join(dataset_dir, database)
            if not os.path.isdir(folder):
                print("Directory {} not found".format(folder))
                print("Creating now...")
                os.makedirs(folder)

    def rdsamp_installed():
        try:
            subprocess.call(["rdsamp", "-h"], stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL)
            return True
        except OSError as e:
            if e.errno == os.errno.ENOENT:
                print("rdsamp not installed")
                return False

        print("rdsamp installed check failed")
        return False

    def remove_unwanted_datasets():
        if dataset_list:
            unwanted_ds = physionet.keys() - dataset_list
            for ds in unwanted_ds:
                physionet.pop(ds, None)


    remove_unwanted_datasets()
    check_folder_existance()
    if not rdsamp_installed():
        sys.exit(1)

    for database, samples in physionet.items():
        print("Downloading {}".format(database))
        database_dir = os.path.join(dataset_dir, database)
        for sample in samples:
            csv_file_path = os.path.join(database_dir, sample) + ".csv"
            if os.path.exists(csv_file_path):
                print("File {} exists. Skipping download...".format(csv_file_path))
                continue

            sample_path = os.path.join(database, sample)
            cmd = ("rdsamp -r {} -c -H -f 0" +
                   " -t 60 -v -pe > {}").format(sample_path, csv_file_path)
            try:
                print("Downloading with command {}...".format(cmd))
                subprocess.check_call(cmd, shell=True)
            except Exception as e:
                print("Failed to execute command: {} with exception: {}".format(cmd, e))
                if os.path.exists(csv_file_path):
                    os.remove(csv_file_path)

        if os.path.isdir(database_dir) and not os.listdir(database_dir):
            cmd = "rm -rf {}".format(database_dir)
            subprocess.check_call(cmd, shell=True)
    print("Done")


def process_data():
    print("Processing data...")
    raw_dir = "datasets/raws"
    processed_dir = "datasets/processed"
    
    ecg_dirs = os.listdir(raw_dir)
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for ecg_name in ecg_dirs:
        print("Processing {}".format(ecg_name))
        processed_csv = os.path.join(processed_dir, ecg_name) + ".csv"
        with open(processed_csv, 'w') as write_processed_file:
            csvwriter = csv.writer(write_processed_file, delimiter=',')
            record_dir = os.path.join(raw_dir, ecg_name)
            for record in os.listdir(record_dir):
                if record.endswith('.csv'):
                    record_path = os.path.join(record_dir, record)
                    with open(record_path) as read_raw_file:
                        reader = csv.reader(read_raw_file)
                        # skip headers
                        reader.__next__()
                        reader.__next__()
                        for row in reader:
                            csvwriter.writerow([row[1]])
    print("Done")


fetch_data()
process_data()
