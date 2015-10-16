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
    cos_sim_mtx = 1 - distance.cdist(np.transpose(train_mtx), np.transpose(train_mtx), 'cosine')
    return cos_sim_mtx
