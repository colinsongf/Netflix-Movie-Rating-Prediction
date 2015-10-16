#################################################################
#
#   __author__ = 'yanhe'
#
#   pcc_based:
#       calculate user-user similarity
#
#################################################################

import numpy as np
import rating_matrix
import user_sim
import movie_sim
import pred_set
import pred_result


# ***** Experiment 3 *****
# PCC based method, user bias standardization
def pcc_user_rating_pred(pair_path, k, option):
    pair = pred_set.pred_pair(pair_path)
    train_mtx = rating_matrix.matrix_transfer(2)
    user_zero_vec = np.where(~train_mtx.any(axis=0))[0]
    # add a bias to the all zero column vectors
    train_mtx[:, [user_zero_vec]] = 0.001
    # user rating standardization
    pcc_mtx = train_mtx - np.sum(train_mtx, axis=0) / len(train_mtx)
    pcc_mtx /= np.linalg.norm(train_mtx, axis=0)
    user_sim_mtx = []
    pred_list = []
    if option == 1 or option == 2:
        user_sim_mtx = user_sim.user_dot_sim(pcc_mtx)
    if option == 3 or option == 4:
        user_sim_mtx = user_sim.user_cos_sim(pcc_mtx)

    # TODO: weighted mean need refine
    for row in pair:
        # pred_rating = 0
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

        pred_rating = np.sum(np.take(train_mtx[movie_id, :], user_knn_list.tolist())) / float(k) + 3
        pred_list.append(pred_rating)
    # output the result
    pred_result.file_writer(pred_list)
    return pred_list


# PCC based method, movie bias standardization
def pcc_item_rating_pred(pair_path, k, option):
    pair = pred_set.pred_pair(pair_path)
    train_mtx = rating_matrix.matrix_transfer(2)
    item_zero_vec = np.where(~train_mtx.any(axis=0))[0]
    # add a bias to the all zero column vectors
    train_mtx[:, [item_zero_vec]] = 0.001
    pcc_mtx = np.transpose(train_mtx)
    # user rating standardization
    pcc_mtx = pcc_mtx - np.sum(pcc_mtx, axis=0) / len(pcc_mtx)
    pcc_mtx /= np.linalg.norm(pcc_mtx, axis=0)
    pcc_mtx = np.transpose(pcc_mtx)
    item_sim_mtx = []
    pred_list = []
    if option == 1 or option == 2:
        item_sim_mtx = movie_sim.item_dot_sim(pcc_mtx)
    if option == 3 or option == 4:
        train_mtx[:, [item_zero_vec]] = 0.001
        item_sim_mtx = movie_sim.item_cos_sim(pcc_mtx)

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
    pred_result.file_writer(pred_list)
    return pred_list
