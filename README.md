# ECG classification
This project aims to use machine learning algorithms(emphasis on Deep Learning) to classify/detect anomalies in ECG signals.

### Requirements
1. Python 3
2. tensorflow/tensorflow-gpu (tested with version 1.13.1)
3. wfdb (tested with version 10.6.1)

### Usage
 - If you need to download the datasets install [wfdb](https://www.physionet.org/physiotools/wfdb.shtml)
 - Get datasets:
    ```sh
    $ python fetch_data.py
    ```
 - Install tensorflow and dependencies using the below command(installs tf without gpu support) or you can find other ways of installing [here](https://www.tensorflow.org/install/)
    ```sh
    $ pip install -r requirements.txt
    ```
 - Run CNN model:
    ```sh
    $ python cnn.py
    ```

### Remarks
* Old implementation can be found in the old_keras_impl directory, but as far as I know it doesn't work.
* Still needs more optimization(try different hyperparameters)


### TODO
- [ ] Add RNN/ResNet models
- [x] Add precision and recall metrics
- [ ] Add more diseases
- [x] Tensorflow 2.0

