# Heartbeat classification

In this study, I wanted to see the process of classification with respect to ECG Hearbeat data.
The dataset being used is the CG heartbeat categorization [data](https://www.kaggle.com/datasets/shayanfazeli/heartbeat?datasetId=29414&searchQuery=torch) The dataset is composed of two collection of heartbeat
signals derived from two famous datasets in hearbeat classification.

This dataset has been used in exploring heartbeat classification using deep neural network architectures, and observing
some of the capabilities of transfer learning on it. The signals correspond to electrocardiogram (ECG) shapes of
heartbeats for the normal case and the cases affected by different arrhythmias and myocardial infarction. These signals
are preprocessed and segmented, with each segment corresponding to a heartbeat.



# Content
## Arrhythmia Dataset

    Number of Samples: 109446
    Number of Categories: 5
    Sampling Frequency: 125Hz
    Data Source: Physionet's MIT-BIH Arrhythmia Dataset
    Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]

## The PTB Diagnostic ECG Database

    Number of Samples: 14552
    Number of Categories: 2
    Sampling Frequency: 125Hz
    Data Source: Physionet's PTB Diagnostic Database

> Remark: All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 188.


# How to run ?
```python
poetry run python -m heart
```

> Quick note: if you want to adjust the parameters, please refer to the utils folder, which contains , constants.py and
constants_helper.py
