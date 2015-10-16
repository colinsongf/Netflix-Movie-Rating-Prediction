#################################################################
#
#   __author__ = 'yanhe'
#
#   bipartite_clustering:
#       implement bipartite clustering algorithm
#
#################################################################

import numpy as np
import user_sim
import item_sim
import pred_set
import bipartite_clustering
import pred_result


# ***** Experiment 4 *****
def bipartite_pred(user_k, item_k, pair_path, category, k, option):
    ui_dict = bipartite_clustering.bipartite(user_k, item_k)
    if category == 'user':
        # combine with Experiment 1
        user_dict = ui_dict[0]
        user_pair = {}
        for key, value in user_dict.items():
            for ele in value:
                user_pair[ele] = key
        bi_user_rating_pred(user_pair, ui_dict[1], pair_path, k, option)
    if category == 'movie':
        # combine with Experiment 2
        item_dict = ui_dict[2]
        item_pair = {}
        for key, value in item_dict.items():
            for ele in value:
                item_pair[ele] = key
        bi_item_rating_pred(item_pair, ui_dict[3], pair_path, k, option)


# user based rating prediction with bipartite clustering
def bi_user_rating_pred(user_pair, train_mtx, pair_path, k, option):
    pair = pred_set.pred_pair(pair_path)
    # train_mtx = rating_matrix.matrix_transfer(2)
    user_sim_mtx = []
    pred_list = []
    # user_zero_vec = np.where(~train_mtx.any(axis=0))[0]
    if option == 1 or option == 2:
        user_sim_mtx = user_sim.user_dot_sim(train_mtx)
    if option == 3 or option == 4:
        # add a bias to the all zero column vectors
        # train_mtx[:, [user_zero_vec]] = 0.001
        user_sim_mtx = user_sim.user_cos_sim(train_mtx)

    # TODO: weighted mean need refine
    for row in pair:
        pred_rating = 0
        movie_id = row[0]
        user_id = row[1]
        key = user_pair[user_id]
        user_sim_list = user_sim_mtx[key]
        # top k+1 nearest neighbors
        user_knn_list = np.argsort(user_sim_list)[::-1][0: k]
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
    pred_result.file_writer(pred_list)
    return pred_list


# item based rating prediction with bipartite clustering
def bi_item_rating_pred(item_pair, train_mtx, pair_path, k, option):
    pair = pred_set.pred_pair(pair_path)
    item_sim_mtx = []
    pred_list = []
    if option == 1 or option == 2:
        item_sim_mtx = item_sim.item_dot_sim(train_mtx)
    if option == 3 or option == 4:
        item_zero_vec = np.where(~train_mtx.any(axis=1))[0]
        train_mtx[[item_zero_vec], 0] = 0.001
        item_sim_mtx = item_sim.item_cos_sim(train_mtx)

    for row in pair:
        pred_rating = 0
        movie_id = row[0]
        user_id = row[1]
        key = item_pair[movie_id]
        item_sim_list = item_sim_mtx[key]
        # top k+1 nearest neighbors
        item_knn_list = np.argsort(item_sim_list)[::-1][0: k]
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
    pred_result.file_writer(pred_list)
    return pred_list
