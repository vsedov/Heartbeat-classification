import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger as log
from scipy.signal import resample as rs
from sklearn.utils import resample

from heart.core import hc, hp
from heart.data.getdata import fetch_data


class HeartBeatData:

    def __init__(self):
        self.data = {
            type_name: pd.read_csv(path, header=None)
            for type_name, path in fetch_data().items()
            if path.endswith(".csv")
        }
        self.labels = {
            0: 'N - Normal Beat',
            1: 'S - Supraventricular premature or ectopic beat',
            2: 'V - Premature ventricular contraction',
            3: 'F - Fusion of ventricular and normal beat',
            4: 'Q - Unclassified beat'
        }

        self.dataset = pd.concat([self.data["train"], self.data["test"]], axis=0, sort=True).reset_index(drop=True)
        self.resample_data()

    def log_data(self):
        for data_name, csv_load in self.data.items():
            log.info(f"Name=>{data_name}=>Amount=>{len(csv_load)}")

    def create_labels(self):
        """
        Creating: is dependent on the categories that it contains:
        There are Five classes , with the following unique idetnifiers:

            N - Normal beat
            S - Supraventricular premature or ectopic beat (atrial or nodal)
            V - Premature ventricular contraction
            F - Fusion of ventricular and normal beat
            Q - Unclassifiable beat

        Due to how kaggle works , and the dataset it self , all the samples that we have are cropped and reduced down
        Why ?
        Because Kaggle ... Kaggle dataset is set for all dims to be around 188 , so we use 187

        """
        labels = self.dataset.iloc[:, -1].astype('category').map(self.labels)

        return labels, np.array(self.dataset.iloc[:, :-1])

    def generate_subplot(self, figure, gs, obs, row, col, title):
        axis = figure.add_subplot(gs[row, col])
        axis.plot(np.linspace(0, 1, 187), obs)
        axis.set_title(title)

    def show_data(self):

        labels, last_col = self.create_labels()
        index_list = {name: labels.index[labels == name_type]
                      for name, name_type in self.labels.items()}

        fig = plt.figure(figsize=(12, 8))
        fig.subplots_adjust(hspace=.5, wspace=.001)
        gs = fig.add_gridspec(5, 3)
        for i in range(4):
            self.generate_plot(fig, gs, last_col[index_list[i][0]], i, 0, self.labels[i])
        hp.save(fig, "TA_VA-TL-VL", "Before normalisation")

    def show_pre_resampled_data(self):
        labels, _ = self.create_labels()
        log.info(f"\n{labels.value_counts()}")
        return labels.value_counts()
        # What this shows is the data is unbalanced, we cannot work with this .
        # What we end up having is overfitting on certain classes.

    def resample_data(self):
        """
        Resample Data:
            The data is very poorly distributed.
        ──────────────────────────────────────────────────────────────────────
        | INFO     | __main__:show_pre_resampled_data:78 -
            N - Normal Beat                                   90589
            Q - Unclassified beat                              8039
            V - Premature ventricular contraction              7236
            S - Supraventricular premature or ectopic beat     2779
            F - Fusion of ventricular and normal beat           803
            Name: 187, dtype: int64
        ──────────────────────────────────────────────────────────────────────
        Convert to :
        ──────────────────────────────────────────────────────────────────────
        | INFO     | __main__:resample_data:108 -
            N - Normal Beat                                   10000
            S - Supraventricular premature or ectopic beat    10000
            V - Premature ventricular contraction             10000
            F - Fusion of ventricular and normal beat         10000
            Q - Unclassified beat                             10000
            dtype: int64
        """
        labels_resampled = pd.Series([], dtype="float64")
        obs_resampled = None

        labels, last_col = self.create_labels()
        index_list = {name: labels.index[labels == name_type]
                      for name, name_type in self.labels.items()}

        for k, v in index_list.items():
            index_list[k] = resample(v, replace=True, n_samples=10000, random_state=1024)
            labels_resampled = pd.concat([labels_resampled, labels.iloc[index_list[k]]])

            obs_resampled = last_col[index_list[k], :] if obs_resampled is None else np.concatenate(
                (obs_resampled, last_col[index_list[k], :]))

        # this is our normalised data
        log.info(labels_resampled.value_counts())
        self.labels_resampled, self.obs_resampled = labels_resampled, obs_resampled


class Augmentation:

    def add_gaussian_noise(self, data):
        return data + np.random.normal(0, 0.005, 187)

    def stretch(self, x):
        strretcher = int(187 * (1 + (random.random() - 0.5) / 3))
        y = rs(x, strretcher)
        if strretcher < 187:
            y_ = np.zeros(shape=(187,))
            y_[:strretcher] = y
        else:
            y_ = y[:187]
        return y_

    def amp(self, data):
        alpha = (random.random() - 0.5)
        return data * -alpha * data + (1+alpha)

    def add_amplify_and_stretch_noise(self, x):
        new_y = self.amp(x)
        new_y = self.stretch(new_y)
        return new_y


class HeartBeatModify(HeartBeatData, Augmentation):
    """
    Due to the fact that we have repeating data now, its important that we add noise to this dataset, before we actively load anything.
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    This class will Exactly that.
    """

    def __init__(self):
        super().__init__()
        self.obs_resampled_with_noise = np.array([self.add_gaussian_noise(obs) for obs in self.obs_resampled])
        self.obs_resampled_with_noise_extra = np.array(
            [self.add_amplify_and_stretch_noise(obs) for obs in self.obs_resampled])

    def plot_augmented_data(self):
        n_index = 0
        obs_resampled, obs_resampled_with_noise_1, obs_resampled_with_noise_2 = self.obs_resampled, self.obs_resampled_with_noise, self.obs_resampled_with_noise_extra
        fig = plt.figure(figsize=(15, 15))
        fig.subplots_adjust(hspace=.5, wspace=.001)
        gs = fig.add_gridspec(5, 3)

        for index, v in enumerate(self.labels.values()):
            self.generate_subplot(fig, gs, obs_resampled[n_index], index, 0, f"normal-{v[:15]}")
            self.generate_subplot(fig, gs, obs_resampled_with_noise_1[n_index], index, 1, f"Gaussian_blue-{v[:15]}")
            self.generate_subplot(fig, gs, obs_resampled_with_noise_2[n_index], index, 2, f'stech-amp-{v[:15]}')
            n_index += 10000
        title = 'Side-by-side Comparison of Original and Two Data Augmentation Versions of Beat Observations Per Class'
        hp.save(fig, "Augmented_image_compare-2", title)
