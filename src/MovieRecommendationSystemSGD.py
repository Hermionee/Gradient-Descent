import Utilities


class MovieRecommendationSystemSGD(object):
    pass


if __name__ == '__main__':
    file_name = "movie_ratings.csv"
    movie_ratings = Utilities.load_data(file_name)

    # This is the code for the k fold cross validation, note that you have to add your prediction functions calls inside the
    # loop, because every iteration is going to be on a new training/testing set.
    cross_validation_set = Utilities.split_data(data=movie_ratings, ratio=0.75, n_splits=5)
    for training, testing in cross_validation_set:
        # use the matrix for your prediction functions calls if you like.
        training_matrix = Utilities.create_matrix(training)
        testing_matrix = Utilities.create_matrix(testing)

"""
def SGD(data):
    '''Learn the vectors p_u and q_i with SGD.
       data is a dataset containing all ratings + some useful info (e.g. number
       of items/users).
    '''

    n_factors = 10  # number of factors
    alpha = .01  # learning rate
    n_epochs = 10  # number of iteration of the SGD procedure

    # Randomly initialize the user and item factors.
    p = np.random.normal(0, .1, (data.n_users, n_factors))
    q = np.random.normal(0, .1, (data.n_items, n_factors))

    # Optimization procedure
    for _ in range(n_epochs):
        for u, i, r_ui in data.all_ratings():
            err = r_ui - np.dot(p[u], q[i])
            # Update vectors p_u and q_i
            p[u] += alpha * err * q[i]
            q[i] += alpha * err * p[u]

       1  2  3
    1 [5, 3, 2]
    2 [3, 5, 4]

    for _ in range(n_epochs):
        for i in data.all_ratings():
            for j in data.all_ratings()[i]:
                r_ij = r[i][j] 
                err = r_ij - np.dot(u[i], v[j])
                # Update vectors p_u and q_i
                u[i] += alpha * err * v[j]
                v[j] += alpha * err * u[i]

    for (_ in range(n_epochs)) and f' > 0.0000001 * if :
        for i in data.all_ratings():
            for j in data.all_ratings()[i]:
                r_ij = r[i][j] 
                err = r_ij - np.dot(u[i], v[j])
                # Update vectors p_u and q_i
                u[i] += alpha * err * v[j]
                v[j] += alpha * err * u[i]

"""
