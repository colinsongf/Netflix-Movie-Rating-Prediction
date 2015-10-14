#################################################################
#
#   __author__ = 'yanhe'
#
#   user_sim:
#       calculate user-user similarity
#
#################################################################


import numpy as np
import scipy.spatial.distance as distance


# user dot product sim, column wise
def user_dot_sim(train_mtx):
    dot_sim_mtx = np.dot(np.transpose(train_mtx), train_mtx)
    return dot_sim_mtx


# user cosine sim, column wise
def user_cos_sim(train_mtx):
    # [row, col] = train_mtx.shape
    # zero_vector = np.where(~train_mtx.any(axis=0))[0]
    # nonzero_train_mtx = np.delete(train_mtx, zero_vector, 1)
    cos_sim_mtx = 1 - distance.cdist(np.transpose(train_mtx), np.transpose(train_mtx), 'cosine')
    return cos_sim_mtx


# use this line to execute the main function
# if __name__ == "__main__":
#     calculate_sim(5, 5, 50, 3)
