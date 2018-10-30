import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_matrix(movies_users):
    movie_matrix = movies_users.pivot(index='userId', columns='movieId', values='rating').fillna(0)
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

    # tell imshow about color map so that only set colors are used
    img = plt.imshow(matrix, interpolation='nearest',
                     cmap=cmap, norm=norm)

    # make a color bar
    plt.colorbar(img, cmap=cmap,
                 norm=norm, boundaries=bounds, ticks=[-5, 0, 5])
    plt.show()


class MovieRecommendationSystem(object):
    def __init__(self, movies_users):
        self.ratings_matrix = create_matrix(movies_users)


if __name__ == '__main__':
    file_name = "movie_ratings.csv"
    movie_ratings = load_data(file_name)
    movie_recommendation_sys = MovieRecommendationSystem(movie_ratings)
    visualize(np.array(movie_recommendation_sys.ratings_matrix))
    print(movie_ratings.head())
