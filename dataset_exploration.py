#################################################################
#
#   __author__ = 'yanhe'
#
#   dataset_exploration:
#       calculate the statistical parameters in training set
#
#################################################################


import csv
import numpy as np
import scipy.sparse as sparse
import scipy.spatial.distance as distance
from numpy import linalg as la
import rating_matrix


# explore the training set
def explorer():
    train_mtx = rating_matrix.matrix_transfer()
    [row, col] = train_mtx.shape
    # ***** part 1.1.1: statistics *****
    rating_one = np.where(train_mtx == 1)
    print 'movie with rating 1: ', rating_one[0].size, '\n'
    rating_three = np.where(train_mtx == 3)
    print 'movie with rating 3: ', rating_three[0].size, '\n'
    rating_five = np.where(train_mtx == 5)
    print 'movie with rating 5: ', rating_five[0].size, '\n'
    rating_avg = np.sum(train_mtx) / np.count_nonzero(train_mtx)
    print 'movie rating average: ', rating_avg, '\n'

    # ***** part 1.1.2: user_id 4321 *****
    curuser = train_mtx[:, 4321]
    movie_num = np.count_nonzero(curuser)
    print 'number of movie rated: ', movie_num, '\n'
    rating_one_num = np.where(curuser == 1)
    print 'movie with rating 1: ', rating_one_num[0].size, '\n'
    rating_three_num = np.where(curuser == 3)
    print 'movie with rating 3: ', rating_three_num[0].size, '\n'
    rating_five_num = np.where(curuser == 5)
    print 'movie with rating 5: ', rating_five_num[0].size, '\n'
    rating_avg_score = np.sum(curuser) / np.count_nonzero(curuser)
    print 'movie rating average: ', rating_avg_score, '\n'

    # ***** part 1.1: movie_id 3 ****
    curmovie = train_mtx[3, :]
    user_num = np.count_nonzero(curmovie)
    print 'number of user rated: ', user_num, '\n'
    rating_one_user = np.where(curmovie == 1)
    print 'movie with rating 1: ', rating_one_user[0].size, '\n'
    rating_three_user = np.where(curmovie == 3)
    print 'movie with rating 3: ', rating_three_user[0].size, '\n'
    rating_five_user = np.where(curmovie == 5)
    print 'movie with rating 5: ', rating_five_user[0].size, '\n'
    rating_avg_user = np.sum(curmovie) / np.count_nonzero(curmovie)
    print 'movie rating average: ', rating_avg_user, '\n'

    # ***** part 1.2.1: user_id 4321 *****
    userquery = train_mtx[:, 4321]
    user_dot_sim = []
    user_cos_sim = []
    for col_idx in range(0, col):
        user_dot_sim.append(np.dot(userquery, train_mtx[:, col_idx]))
        if la.norm(train_mtx[:, col_idx]) == 0:
            user_cos_sim.append(0)
        else:
            user_cos_sim.append(distance.cosine(userquery, train_mtx[:, col_idx]))
    knn_user_dot_sim = np.argsort(user_dot_sim)[-5:]
    knn_user_cos_sim = np.argsort(user_cos_sim)[-5:]
    print 'top 5 NNs of user 4321 by dot product similarity: ', knn_user_dot_sim
    print 'top 5 NNs of user 4321 by cosine similarity: ', knn_user_cos_sim

    # ***** part 1.2.2: movie_id 3 *****
    moviequery = train_mtx[3, :]
    movie_dot_sim = []
    movie_cos_sim = []
    for row_idx in range(0, row):
        movie_dot_sim.append(np.dot(moviequery, train_mtx[row_idx, :]))
        if la.norm(train_mtx[row_idx, :]) == 0:
            movie_cos_sim.append(0)
        else:
            movie_cos_sim.append(distance.cosine(moviequery, train_mtx[row_idx, :]))
    knn_movie_dot_sim = np.argsort(movie_dot_sim)[-5:]
    knn_movie_cos_sim = np.argsort(movie_cos_sim)[-5:]
    print 'top 5 NNs of movie 3 by dot product similarity: ', knn_movie_dot_sim
    print 'top 5 NNs of movie 3 by cosine similarity: ', knn_movie_cos_sim

# use this line to execute the main function
if __name__ == "__main__":
    explorer()
