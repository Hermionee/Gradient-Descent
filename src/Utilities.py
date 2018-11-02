import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Metrics(object):

    @staticmethod  # Root Mean Square Error
    def rmse(predictions, test_y):
        return np.sqrt(Metrics.mse(predictions, test_y))

    @staticmethod  # Mean Square Error
    def mse(predictions, test_y):
        return np.mean((predictions - test_y) ** 2)


def create_matrix(movies_users):
    movie_matrix = movies_users.pivot(index='userId', columns='movieId', values='rating')
    return movie_matrix


def load_data(filename):
    parent_folder = os.path.abspath(os.path.join(__file__, '../..'))
    data_folder = os.path.join(parent_folder, "data")
    file_path = os.path.join(data_folder, filename)
    return pd.read_csv(file_path, delimiter=',')


def visualize(matrix):
    # make a color map of fixed colors
    cmap = mpl.colors.ListedColormap(['white', 'blue', 'purple', 'red', 'orange', 'green'])
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    img = plt.imshow(matrix, interpolation='nearest',
                     cmap=cmap, norm=norm)
    plt.colorbar(img, cmap=cmap,
                 norm=norm, boundaries=bounds, ticks=[-5, 0, 5])
    plt.show()
