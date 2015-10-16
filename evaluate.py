#################################################################
#
#   __author__ = 'yanhe'
#
#   evaluate:
#       predict rating in the dev set and test set
#       option 1: dot product sim & mean
#       option 2: dot product sim & weighted mean
#       option 3: cos sim & mean
#       option 4: cos sim & weighted mean
#
#################################################################


import user_user
import movie_movie
import pcc_based
import bipartite_based


# set parameters
pair_path = "HW4_data/dev.csv"
k = 10
option = 1

# ***** Experiment 1 *****
user_user.user_rating_pred(pair_path, k, option)

# ***** Experiment 2 *****
movie_movie.item_rating_pred(pair_path, k, option)

# ***** Experiment 3 *****
# standardization on user
pcc_based.pcc_user_rating_pred(pair_path, k, option)
# standardization on movie
pcc_based.pcc_item_rating_pred(pair_path, k, option)

# ***** Experiment 4 *****
# set the number of clusters
user_cluster = 2000
movie_cluster = 1500
# set category to be 'user' or 'movie'
category = 'user'
bipartite_based.bipartite_pred(user_cluster, movie_cluster, pair_path, category, k, option)
