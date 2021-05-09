# ECG classification
This project aims to use machine learning algorithms(emphasis on Deep Learning) to classify/detect anomalies in ECG signals.


### Requirements
1. Python 3
2. tensorflow/tensorflow-gpu (cnn.py tested with 1.13.1 and cnn_tf2.py tested with 2.5.0)
3. wfdb (tested with version 10.6.2)


### Wfdb dataset setup
 - You can find the full wfdb documentation [here](https://www.physionet.org/content/wfdb)
 - After downloading the wfdb tool and installing it, add the bin/rdsamp to your PATH for the fetch_data.py script to use it

### Usage
 - After the wfdb command is available generate the datasets using:
    ```sh
    $ python fetch_data.py
    ```
 - Install tensorflow and dependencies using the below command(installs tf without gpu support) or you can find other ways of installing [here](https://www.tensorflow.org/install/)
    ```sh
    $ pip install -r requirements.txt
    ```
 - Run CNN model using Tensorflow 1:
    ```sh
    $ python cnn.py
    ```
 - Run CNN model using Tensorflow 2:
    ```sh
    $ python cnn_tf2.py
    ```


### Remarks
* Installing and setting up wfdb is probably the most error prone thing in using this project. Make sure that you have access to rdsamp in the current shell before executing fetch_data.py
* Old implementation can be found in the old_keras_impl directory, but as far as I know it doesn't work.
* Still needs more optimization(try different hyperparameters)
* You can find the Tensorflow 2.0 implementation in tensorflow_impl/cnn_tf2.py


### TODO
- [ ] Add RNN/ResNet models
- [x] Add precision and recall metrics
- [ ] Add more diseases
- [x] Tensorflow 2.0

