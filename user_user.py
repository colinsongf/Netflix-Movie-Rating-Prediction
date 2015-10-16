#################################################################
#
#   __author__ = 'yanhe'
#
#   user_user:
#       predict rating with user-user similarity
#
#################################################################

import numpy as np
import rating_matrix
import user_sim
import pred_set


# ***** Experiment 1 *****
# predict rating for each movie-user pair
def user_rating_pred(pair_path, k, option):
    pair = pred_set.pred_pair(pair_path)
    train_mtx = rating_matrix.matrix_transfer(2)
    user_sim_mtx = []
    pred_list = []
    user_zero_vec = np.where(~train_mtx.any(axis=0))[0]
    if option == 1 or option == 2:
        user_sim_mtx = user_sim.user_dot_sim(train_mtx)
    if option == 3 or option == 4:
        # add a bias to the all zero column vectors
        train_mtx[:, [user_zero_vec]] = 0.001
        user_sim_mtx = user_sim.user_cos_sim(train_mtx)

    # TODO: weighted mean need refine
    for row in pair:
        pred_rating = 0
        movie_id = row[0]
        user_id = row[1]
        user_sim_list = user_sim_mtx[user_id]
        # top k+1 nearest neighbors
        user_knn_list = np.argsort(user_sim_list)[::-1][0: k+1]
        # TODO: if two sim equals, small user_id comes first
        if user_id in user_knn_list:
            position = np.where(user_knn_list == user_id)
            user_knn_list = np.delete(user_knn_list, position)
        else:
            user_knn_list = np.delete(user_knn_list, len(user_knn_list) - 1)

        if option == 1 or option == 3:
            pred_rating = np.sum(np.take(train_mtx[movie_id, :], user_knn_list.tolist())) / float(k) + 3
        # TODO: problem exists, what if weighted sum is zero
        if option == 2 or option == 4:
            user_knn_sim = user_sim_list[user_knn_list]
            if np.sum(user_knn_sim) != 0:
                weight = user_knn_sim / np.sum(user_knn_sim)
                pred_rating = np.sum(np.multiply(np.take(train_mtx[movie_id, :], user_knn_list.tolist()), weight)) + 3
            else:
                pred_rating = 3.0

        pred_list.append(pred_rating)
    # output the result
    file_writer(pred_list)
    return pred_list


# write the result into txt file
def file_writer(pred_list):
    # write the ranking result into txt file
    f = open('eval/user_dev_pred.txt', 'w')
    num = len(pred_list)
    for idx in range(0, num):
        f.write("{}\n".format(pred_list[idx]))
