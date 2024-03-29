import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger as log
from scipy.signal import resample as rs
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset

from heart.core import hp
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
        # new_y = self.stretch(new_y)
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
        obs_resampled, obs_resampled_with_noise, obs_resampled_with_noise_2 = self.obs_resampled, self.obs_resampled_with_noise, self.obs_resampled_with_noise_extra
        fig = plt.figure(figsize=(15, 15))
        fig.subplots_adjust(hspace=.5, wspace=.001)
        gs = fig.add_gridspec(5, 3)

        for index, v in enumerate(self.labels.values()):
            self.generate_subplot(fig, gs, obs_resampled[n_index], index, 0, f"normal-{v[:15]}")
            self.generate_subplot(fig, gs, obs_resampled_with_noise[n_index], index, 1, f"Gaussian_blue-{v[:15]}")
            self.generate_subplot(fig, gs, obs_resampled_with_noise_2[n_index], index, 2, f'stech-amp-{v[:15]}')
            n_index += 10000
        title = 'Side-by-side Comparison of Original and Two Data Augmentation Versions of Beat Observations Per Class'
        hp.save(fig, "Augmented_image_compare-2", title)

    def conversion(self, data, target):
        return TensorDataset(torch.from_numpy(data), torch.from_numpy(target))

    def data_loader(self):
        batch_size = 32
        num_train = len(self.obs_resampled)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        # split of data , and the ammoutn of data that we want to be validation and test data
        # 0.5 atm
        split = int(np.floor(0.2 * num_train))
        train_idx, test_valid_idx = indices[split:], indices[:split]
        # the validation split will be based on 0.5
        test_valid_split = int(len(test_valid_idx) * 0.5)
        test_idx, valid_idx = test_valid_idx[:test_valid_split], test_valid_idx[test_valid_split:]

        factorised_resample = pd.factorize(self.labels_resampled.astype('category'))[0]

        def parser(data, y):
            return [(self.conversion(data[index_type], y[index_type]), index_name) for index_name, index_type in {
                "train": train_idx,
                "valid": valid_idx,
                "test": test_idx
            }.items()]

        return {
            ds_type: list(
                map(
                    lambda x:
                    {x[1]: DataLoader(x[0], shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True)},
                    container))
            for ds_type, container in {
                "level_1": parser(self.obs_resampled_with_noise, factorised_resample),
                "level_2": parser(self.obs_resampled_with_noise_extra, factorised_resample),
            }.items()
        }

    def plot_ae_dataset(self, train_n, train_a):
        for i in range(8):
            series = train_n[i]
            plt.subplot(2, 4, i + 1)
            plt.plot(series.tolist())
            plt.title("Normal ECG Example {}".format(i + 1))
            plt.xlabel("Time")

        plt.show()

        for i in range(8):
            series = train_a[i]
            plt.subplot(2, 4, i + 1)
            plt.plot(series.tolist())
            plt.title("Abnormal ECG Example {}".format(i + 1))
            plt.xlabel("Time")
        plt.show()

        pass

    def auto_encoder_dataset(self):
        """AutoEncoder, dataset mainly based on autoencoders."""
        self.dataset[self.dataset[187] == 0]
        data = self.obs_resampled.squeeze()

        factorised_resample = pd.factorize(self.labels_resampled.astype('category'))[0]
        # log.info(factorised_resample)
        # split data normal and abnormal
        normal_data = data[factorised_resample == 0]
        abnormal_data = data[factorised_resample == 1]
        # log.info(normal_data)
        # log.info(abnormal_data)

        trainN = normal_data[:int(0.8 * len(normal_data))]
        testN = normal_data[int(0.8 * len(normal_data)):]
        trainA = abnormal_data[:int(0.8 * len(abnormal_data))]
        testA = abnormal_data[int(0.8 * len(abnormal_data)):]

        return {
            "level_1":
                DataLoader(
                    TensorDataset(torch.from_numpy(trainN), torch.from_numpy(trainA)), shuffle=True, batch_size=32),
            "level_2":
                DataLoader(
                    TensorDataset(torch.from_numpy(testN), torch.from_numpy(testA)), shuffle=True, batch_size=32)
        }
