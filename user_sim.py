#################################################################
#
#   __author__ = 'yanhe'
#
#   user_sim:
#       calculate user-user similarity
#       option 1: dot product sim & mean
#       option 2: dot product sim & weighted mean
#       option 3: cos sim & mean
#       option 4: cos sim & weighted mean
#
#################################################################


import rating_matrix
import numpy as np
import scipy.spatial.distance as distance
import numpy.linalg as la


def calculate_sim(movie_id, user_id, k, option):
    train_mtx = rating_matrix.matrix_transfer(2)
    [row, col] = train_mtx.shape
    # user_dot_sim = []
    user_cos_sim = []
    user_query = train_mtx[:, user_id]
    # knn_user = np.concatenate((knn_user_dot_sim, knn_user_cos_sim), axis=0)

    ################################
    if option == 1:
        # for col_idx in range(0, col):
        #     user_dot_sim.append(np.dot(user_query, train_mtx[:, col_idx]))
        user_dot_sim = np.sum(np.transpose(train_mtx) * user_query, axis=1)
        # find the k nearest neighbors
        knn_user_dot_sim = np.argsort(user_dot_sim)[::-1][0: k]
        pred_rating = np.sum(np.take(train_mtx[movie_id, :], knn_user_dot_sim.tolist())) / float(k) + 3
        return pred_rating

    ################################
    if option == 2:
        user_dot_sim = np.sum(np.transpose(train_mtx) * user_query, axis=1)
        # find the k nearest neighbors
        knn_user_dot_sim = np.argsort(user_dot_sim)[::-1][0: k]

    ################################
    if option == 3:
        for col_idx in range(0, col):
            if la.norm(train_mtx[:, col_idx]) == 0:
                user_cos_sim.append(0)
            else:
                user_cos_sim.append(distance.cosine(user_query, train_mtx[:, col_idx]))
        # find the k nearest neighbors
        knn_user_cos_sim = np.argsort(user_cos_sim)[::-1][0: k]
        pred_rating = np.sum(np.take(train_mtx[movie_id, :], knn_user_cos_sim.tolist())) / float(k) + 3
        return pred_rating

    ################################
    if option == 4:
        for col_idx in range(0, col):
            if la.norm(train_mtx[:, col_idx]) == 0:
                user_cos_sim.append(0)
            else:
                user_cos_sim.append(distance.cosine(user_query, train_mtx[:, col_idx]))
        # find the k nearest neighbors
        knn_user_cos_sim = np.argsort(user_cos_sim)[::-1][0: k]
    #################################################################


# user sim, column wise
def user_dot_sim(train_mtx, k):
    [row, col] = train_mtx.shape
    user_dot_sim_dict = {}
    for col_idx in range(0, col):
        user_query = train_mtx[:, col_idx]
        user_dot_sim_val = np.sum(np.transpose(train_mtx) * user_query, axis=1)
        # find the k nearest neighbors
        knn_user_dot_sim = np.argsort(user_dot_sim_val)[::-1][0: k+1]
        if col_idx in knn_user_dot_sim:
            position = np.where(knn_user_dot_sim == col_idx)
            knn_user_dot_sim = np.delete(knn_user_dot_sim, position)
            user_dot_sim_dict[col_idx] = knn_user_dot_sim
        else:
            knn_user_dot_sim = np.delete(knn_user_dot_sim, len(knn_user_dot_sim) - 1)
            user_dot_sim_dict[col_idx] = knn_user_dot_sim
    return user_dot_sim_dict


# use this line to execute the main function
# if __name__ == "__main__":
#     calculate_sim(5, 5, 50, 3)
