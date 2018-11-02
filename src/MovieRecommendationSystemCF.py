from operator import itemgetter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

import Utilities


class MovieRecommendationSystemCF(object):
    def __init__(self, movies_users):
        self.ratings_matrix = Utilities.create_matrix(movies_users)
        print("matrix_created!")
        self.users_count = len(self.ratings_matrix.index)  # 610 users
        self.movies_count = len(self.ratings_matrix.columns)  # 9724 movies
        self.ratings = np.count_nonzero(self.ratings_matrix.values)  # 100836 non zero ratings.
        # missing 5M+ ratings.
        print("begin calculating similarity!")
        # The preprocessing takes forever, so pickle it to do it only once.
        try:
            self.user_based_similarity = pd.read_pickle("obj/user_based_similarity.pkl")
        except IOError:
            self.user_based_similarity = self.calculate_distances(similarity_type="u")
            self.user_based_similarity.to_pickle("obj/user_based_similarity.pkl")
        try:
            self.movie_based_similarity = pd.read_pickle("obj/movie_based_similarity.pkl")
        except IOError:
            self.movie_based_similarity = self.calculate_distances(similarity_type="m")
            self.movie_based_similarity.to_pickle("obj/movie_based_similarity.pkl")
        print("done calculating similarity!")
        self.users_rating_mean = self.ratings_matrix.mean(axis=1)
        self.movies_rating_mean = self.ratings_matrix.mean(axis=0)

    def calculate_distances(self, similarity_type):
        # 1. determine the type of similarity that needs to be calculated:
        if similarity_type == "m":
            matrix = self.ratings_matrix
            similarity_matrix = pd.pivot(self.ratings_matrix.columns, self.ratings_matrix.columns,
                                         np.ones(self.movies_count))
        else:
            matrix = self.ratings_matrix.transpose()
            similarity_matrix = pd.pivot(self.ratings_matrix.index, self.ratings_matrix.index,
                                         np.ones(self.users_count))

        # 2. Calculate the cosine distances between each 2 movies/users, depending on the chosen type
        for x in matrix:
            for y in matrix:
                if x != y:
                    x_matrix = matrix[x]
                    y_matrix = matrix[y]
                    similarity_matrix[x][y] = cosine(x_matrix, y_matrix)
            similarity_matrix = similarity_matrix.reindex(
                similarity_matrix['Value'].sort_values(by=x, ascending=False).index)
        # 3. Return similarity matrix
        return similarity_matrix

    def predict_review(self, x):
        distances = []
        for xn in self.ratings_matrix:
            if xn != x:
                distances.append(cosine(xn, x))
        distances = sorted(distances, key=itemgetter(0))
        y = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        k = 0
        while k < self.K:
            row = distances[k]
            point = row[1]
            y[point[1]] += 1
            k += 1
        return max(y, key=y.get)

    def predictions(self, similarity_type):
        if similarity_type == "m":
            predictions = self.ratings_matrix.dot(self.movie_based_similarity) / np.array(
                [np.abs(self.movie_based_similarity).sum(axis=1)])
        else:
            matrix = self.ratings_matrix.transpose()
            ratings_difference = (matrix - self.users_rating_mean[:, np.newaxis])
            predictions = self.users_rating_mean[:, np.newaxis] + self.user_based_similarity.dot(
                ratings_difference) / np.array([np.abs(self.user_based_similarity).sum(axis=1)])
        return predictions

    def evaluate(self, predictions):
        # RMSE
        return Utilities.Metrics.rmse(predictions, self.ratings_matrix)


if __name__ == '__main__':
    file_name = "movie_ratings.csv"
    movie_ratings = Utilities.load_data(file_name)
    print(movie_ratings.head())

    cross_validation_set = Utilities.split_data(data=movie_ratings, ratio=0.75, n_splits=5)
    for training, testing in cross_validation_set:
        # use the matrix for your prediction functions calls if you like.
        training_matrix = Utilities.create_matrix(training)
        testing_matrix = Utilities.create_matrix(testing)
        Utilities.visualize(np.array(training_matrix))
        # movie_recommendation_sys = MovieRecommendationSystemCF(training_matrix)
