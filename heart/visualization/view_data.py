import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger as log

from heart.core import hc
from heart.data.getdata import fetch_data


class ViewDataSet:

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

    def abnormal_normal_split(self, data):
        return (data[data[187] == 0], data[data[187] != 0])

    def log_data(self):
        for data_name, csv_load in self.data.items():
            log.info(f"Name=>{data_name}=>Amount=>{len(csv_load)}")

    def create_labels(self):
        """
        Creating Lables: is dependent on the categories that it contains:
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

    def show_data(self):
        labels, last_col = self.create_labels()
        index_list = {name: labels.index[labels == name_type]
                      for name, name_type in self.labels.items()}

        fig = plt.figure(figsize=(12, 8))
        fig.subplots_adjust(hspace=.5, wspace=.001)

        def generate_plot(figure, last_col, index, title):
            axis = figure.add_subplot(fig.add_gridspec(5, 1)[index, 0])
            axis.plot(np.linspace(0, 1, 187), last_col)
            axis.set_title(title)

        for i in range(4):
            generate_plot(fig, last_col[index_list[i][0]], i, self.labels[i])
        file_path = f"{hc.DIR}reports/data/raw/"
        filename = f"{file_path}TA_VA-TL-VL"

        plt.savefig(f'{filename}.png')
