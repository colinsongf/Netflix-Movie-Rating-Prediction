#################################################################
#
#   __author__ = 'yanhe'
#
#   bipartite_clustering:
#       find k clusters for users and items
#
#################################################################


import sys
import numpy as np
import scipy.spatial.distance as distance
import random
import rating_matrix


# row wise k-means for train_mtx
def k_means(train_mtx, k_cluster):
    zero_vec = np.where(~train_mtx.any(axis=1))[0]
    train_mtx[[zero_vec], 0] = 0.001
    row_num = train_mtx.shape[0]
    pick = random.sample(range(row_num), k_cluster)
    center_mtx = train_mtx[pick, :]
    num_of_round = 0
    max_sum_cos_dis = 0 - sys.maxint
    k_dict = {}
    while num_of_round < 50:
        num_of_round += 1
        cos_sim_mtx = np.subtract(1.0, distance.cdist(train_mtx, center_mtx, 'cosine'))
        cos_sim_sum = cos_sim_mtx.max(axis=1).sum()
        max_idx_of_row = cos_sim_mtx.argmax(axis=1)
        # TODO: improve this part
        for idx in range(0, len(max_idx_of_row)):
            if idx in pick:
                if pick.index(idx) in k_dict:
                    k_dict[pick.index(idx)].append(idx)
                else:
                    k_dict[pick.index(idx)] = [idx]
            else:
                if max_idx_of_row[idx] in k_dict:
                    k_dict[max_idx_of_row[idx]].append(idx)
                else:
                    k_dict[max_idx_of_row[idx]] = [idx]
        # update the k cluster center
        for idx_k in range(0, len(center_mtx)):
            center_mtx[idx_k] = train_mtx[k_dict[idx_k], :].mean(axis=0)
        # check if converged
        if cos_sim_sum > max_sum_cos_dis and cos_sim_sum - max_sum_cos_dis > 5:
            max_sum_cos_dis = cos_sim_sum
            # print max_sum_cos_dis
        if cos_sim_sum > max_sum_cos_dis and cos_sim_sum - max_sum_cos_dis <= 5:
            max_sum_cos_dis = cos_sim_sum
            # print max_sum_cos_dis
            break
    return k_dict


def bipartite(user_k, item_k):
    train_mtx_ori = rating_matrix.matrix_transfer(2)
    [row, col] = train_mtx_ori.shape
    train_mtx = np.transpose(train_mtx_ori)
    num_of_round = 0
    user_dict = {}
    item_dict = {}
    train_mtx_p = []
    train_mtx_pp = []

    while num_of_round < 5:
        num_of_round += 1
        # step 1
        user_dict = k_means(train_mtx, user_k)
        # step 2
        train_mtx_p = np.zeros((row, user_k))
        for cluster in user_dict:
            train_mtx_p[:, cluster] = np.asarray(train_mtx_ori[:, user_dict.get(cluster)].mean(axis=1)).reshape(row)
        # step 3
        item_dict = k_means(train_mtx_p, item_k)
        # step 4
        train_mtx_pp = np.zeros((item_k, col))
        for cluster in item_dict:
            train_mtx_pp[cluster, :] = np.asarray(train_mtx_ori[item_dict.get(cluster), :].mean(axis=0)).reshape(col)
        # step 5
        train_mtx = np.transpose(train_mtx_pp)

    print 'bipartite finished.'
    user_item_dict = (user_dict, train_mtx_p, item_dict, train_mtx_pp)
    return user_item_dict
