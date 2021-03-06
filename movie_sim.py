#################################################################
#
#   __author__ = 'yanhe'
#
#   movie_sim:
#       calculate movie-movie similarity
#
#################################################################

import numpy as np
import scipy.spatial.distance as distance


# user dot product sim, row wise
def item_dot_sim(train_mtx):
    dot_sim_mtx = np.dot(train_mtx, np.transpose(train_mtx))
    return dot_sim_mtx


# user cosine sim, row wise
def item_cos_sim(train_mtx):
    cos_sim_mtx = (2 - distance.cdist(train_mtx, train_mtx, 'cosine')) / 2
    return cos_sim_mtx
