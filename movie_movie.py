#################################################################
#
#   __author__ = 'yanhe'
#
#   movie_movie:
#       predict rating with movie-movie similarity
#
#################################################################

import numpy as np
import rating_matrix
import item_sim
import pred_set


# ***** Experiment 2 *****
# movie-movie similarity prediction
def item_rating_pred(pair_path, k, option):
    pair = pred_set.pred_pair(pair_path)
    train_mtx = rating_matrix.matrix_transfer(2)
    item_zero_vec = np.where(~train_mtx.any(axis=0))[0]
    item_sim_mtx = []
    pred_list = []
    if option == 1 or option == 2:
        item_sim_mtx = item_sim.item_dot_sim(train_mtx)
    if option == 3 or option == 4:
        train_mtx[:, [item_zero_vec]] = 0.001
        item_sim_mtx = item_sim.item_cos_sim(train_mtx)

    for row in pair:
        pred_rating = 0
        movie_id = row[0]
        user_id = row[1]
        item_sim_list = item_sim_mtx[movie_id]
        # top k+1 nearest neighbors
        item_knn_list = np.argsort(item_sim_list)[::-1][0: k+1]
        if movie_id in item_knn_list:
            position = np.where(item_knn_list == movie_id)
            item_knn_list = np.delete(item_knn_list, position)
        else:
            item_knn_list = np.delete(item_knn_list, len(item_knn_list) - 1)

        if option == 1 or option == 3:
            pred_rating = np.sum(np.take(train_mtx[:, user_id], item_knn_list.tolist())) / float(k) + 3
        if option == 2 or option == 4:
            item_knn_sim = item_sim_list[item_knn_list]
            if np.sum(item_knn_sim) != 0:
                weight = item_knn_sim / np.sum(item_knn_sim)
                pred_rating = np.sum(np.multiply(np.take(train_mtx[:, user_id], item_knn_list.tolist()), weight)) + 3
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

